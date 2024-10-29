import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from lar_msgs.msg import CarControlStamped, CarStateStamped
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose2D

from racecar_nn_controller.cartesian_to_curvilinear import pose_to_curvi
from racecar_nn_controller.transform_to_track_frame import transform_to_track, wrap_to_pi

from ament_index_python.packages import get_package_share_directory

import os

import torch
import torch.nn as nn
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler
import json


class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, input_weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(SimpleNet, self).__init__()
        self.input_weights = nn.Parameter(torch.tensor(input_weights), requires_grad=False)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.LeakyReLU()
        #self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):
        x = self.input_weights * x

        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        #x = self.dropout(x)
        
        x = self.fc3(x)     
        return x
        
      
class RacecarNNController(Node):
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

        # Load the trained model
        package_share_directory = get_package_share_directory('racecar_nn_controller')
        model_path = os.path.join(package_share_directory, 'models', 'model_noisy_9_64.pth')
        scaler_params_path = os.path.join(package_share_directory, 'models', 'scaling_params_noisy_9.json')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = SimpleNet(input_size=5, hidden_size=64, output_size=2)
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

        self.track_flag = False

        print("Model used for neural network control running on", self.device, ":\n")
        print(self.model)
    
    def track_callback(self, msg):
        self.track_data = [msg.x, msg.y, msg.theta]
        self.track_flag = True
        self.destroy_subscription(self.subscription_track)
    	
    def listener_callback(self, msg):

        if self.track_flag:
            
            state_values = np.array([msg.pos_x, msg.pos_y, msg.vel_x, msg.vel_y, wrap_to_pi(msg.turn_angle),  np.clip(msg.turn_rate,-1,1)], dtype = np.float32)

            local_state = transform_to_track(state_values, self.track_data)
                    
            curvilinear_pose = pose_to_curvi(self.track_shape_data, local_state)
                
            state = np.concatenate([curvilinear_pose, [local_state[3], local_state[4]]], axis=None)
            state_transformed = self.scaler.transform([state])

            state_transformed = torch.Tensor(state_transformed).to(self.device)
            
            with torch.no_grad():
                action = self.model(state_transformed.unsqueeze(0)).cpu().numpy().flatten()
                action = action * self.scale_action + self.mean_action

            control_msg = CarControlStamped()
            control_msg.header.stamp = self.get_clock().now().to_msg()
            control_msg.throttle = float(action[0])
            control_msg.steering = float(action[1])
            self.publisher_.publish(control_msg)

def main(args=None):
    rclpy.init(args=args)
    racecar_nn_controller = RacecarNNController()
    rclpy.spin(racecar_nn_controller)
    racecar_nn_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

