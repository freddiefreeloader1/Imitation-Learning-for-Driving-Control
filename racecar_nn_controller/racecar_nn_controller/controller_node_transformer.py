import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from lar_msgs.msg import CarControlStamped, CarStateStamped
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose2D

from racecar_nn_controller.cartesian_to_curvilinear import pose_to_curvi
from racecar_nn_controller.transform_to_track_frame import transform_to_track, wrap_to_pi

from ament_index_python.packages import get_package_share_directory

import ipdb

import os

import torch
import torch.nn as nn
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from collections import deque
    

class TransformerModel(nn.Module):
    def __init__(self, state_dim, action_dim, nhead, num_encoder_layers, num_decoder_layers, d_model, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Embedding layers for states and actions
        self.state_embedding = nn.Linear(state_dim, d_model)
        self.action_embedding = nn.Linear(action_dim, d_model)

        # Transformer Encoder and Decoder
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward)

        # Final output layer to predict the 5th action
        self.fc_out = nn.Linear(d_model, action_dim)
    
    def forward(self, states, actions):
        # Embed states and actions
        state_embed = self.state_embedding(states)  # (batch_size, seq_length, d_model)
        action_embed = self.action_embedding(actions)  # (batch_size, seq_length, d_model)

        # Prepare input for transformer
        state_embed = state_embed.permute(1, 0, 2)  # (seq_length, batch_size, d_model)
        action_embed = action_embed.permute(1, 0, 2)  # (seq_length, batch_size, d_model)
        # ipdb.set_trace()
        # Forward pass through transformer

        print(state_embed)

        print(action_embed)

        transformer_output = self.transformer(state_embed, action_embed)  # (seq_length, batch_size, d_model)
        # Output the predicted 5th action (output of the last sequence step)
        output = self.fc_out(transformer_output[-1, :, :])  # (batch_size, action_dim)
        return output
      
class RacecarNNControllerTransformer(Node):
    def __init__(self):
        super().__init__('racecar_nn_controller')
        track_path = "/root/ros2_ws/install/lar_utils/share/lar_utils/config/track/sampled/la_track.yaml"
        with open(track_path, "r") as file:
            self.track_shape_data = yaml.safe_load(file)

        self.track_data = []
        self.subscription_track = self.create_subscription(
            Pose2D,
            '/sim/track/pose2d',
            self.track_callback,
            1)

        self.subscription = self.create_subscription(
            CarStateStamped,
            '/sim/car/state',
            self.listener_callback,
            1)

        self.publisher_ = self.create_publisher(CarControlStamped, '/sim/car/set/control', 1)

        # Load the trained transformer model
        package_share_directory = get_package_share_directory('racecar_nn_controller')
        model_path = os.path.join(package_share_directory, 'models', 'model_transformer.pth')
        scaler_params_path = os.path.join(package_share_directory, 'models', 'scaling_params_2.json')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the transformer model
        print(torch.version.__version__)
        self.model = TransformerModel(state_dim= 5, action_dim=2, 
                         d_model=64, nhead=4, num_encoder_layers=1, 
                         num_decoder_layers=0, dim_feedforward=128)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()


        with open(scaler_params_path, 'r') as f:
            scaling_params = json.load(f)

        mean_state = scaling_params['mean'][0]
        scale_state = scaling_params['std'][0]
        self.mean_action = scaling_params['mean'][1]
        self.scale_action = scaling_params['std'][1]

        self.scaler = StandardScaler()
        self.scaler.mean_ = mean_state
        self.scaler.scale_ = scale_state

        self.action_scaler = StandardScaler()
        self.action_scaler.mean_ = self.mean_action
        self.action_scaler.scale_ = self.scale_action
        

        # Buffer to store sequences of 5 states and 4 actions
        self.state_buffer = deque(maxlen=5)
        self.action_buffer = deque(maxlen=4)

        # print("Transformer model used for neural network control running on", self.device, ":\n")
        # print(self.model)

    def track_callback(self, msg):
        self.track_data = [msg.x, msg.y, msg.theta]
        self.destroy_subscription(self.subscription_track)

    def listener_callback(self, msg):
        state_values = np.array([[msg.pos_x, msg.pos_y, msg.vel_x, msg.vel_y, msg.turn_angle, np.clip(msg.turn_rate, -1, 1)]], dtype=np.float32)
        local_state = transform_to_track(state_values[0], self.track_data)
        curvilinear_pose = pose_to_curvi(self.track_shape_data, local_state)
        state = np.concatenate([curvilinear_pose, [local_state[3], local_state[4]]], axis=None)

        # state_transformed = self.scaler.transform([state])
        self.state_buffer.append(state)
        if len(self.action_buffer) < 4:
            self.update_action_buffer(0.0, 0.0)
        

        if len(self.state_buffer) == 5 and len(self.action_buffer) == 4:
            states = torch.tensor(self.state_buffer, dtype=torch.float32).unsqueeze(0).to(self.device)
            actions = torch.tensor(self.action_buffer, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                predicted_action = self.model(states, actions).cpu().clamp(min=-1.0, max=1.0).numpy().flatten()
                self.update_action_buffer(predicted_action[0], predicted_action[1])
                # predicted_action = predicted_action * self.scale_action + self.mean_action

            control_msg = CarControlStamped()
            control_msg.header.stamp = self.get_clock().now().to_msg()
            control_msg.throttle = float(predicted_action[0])
            control_msg.steering = float(predicted_action[1])
            self.publisher_.publish(control_msg)

    def update_action_buffer(self, throttle, steering):
        action = [throttle, steering]
        # action = self.action_scaler.transform([action])[0]
        self.action_buffer.append(action)


def main(args=None):
    rclpy.init(args=args)
    racecar_nn_controller = RacecarNNControllerTransformer()
    rclpy.spin(racecar_nn_controller)
    racecar_nn_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
