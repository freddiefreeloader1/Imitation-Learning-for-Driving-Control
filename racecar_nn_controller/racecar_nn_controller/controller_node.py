import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from lar_msgs.msg import CarControlStamped, CarStateStamped
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose2D

from racecar_nn_controller.cartesian_to_curvilinear import pose_to_curvi
from racecar_nn_controller.transform_to_track_frame import transform_to_track, wrap_to_pi
from racecar_nn_controller.pure_pursuit_controller import pure_pursuit_controller

from ament_index_python.packages import get_package_share_directory

import os

import torch
import torch.nn as nn
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import time
import pandas as pd

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

        track_path = "/home/la-user/ros2_ws/src/racecar_nn_controller/racecar_nn_controller/la_track.yaml"
        with open(track_path, "r") as file:
            self.track_shape_data = yaml.safe_load(file)

        self.track_shape_data['track']['xCoords'] = np.array(self.track_shape_data['track']['xCoords']) - self.track_shape_data['track']['x_init']
        self.track_shape_data['track']['yCoords'] = np.array(self.track_shape_data['track']['yCoords']) - self.track_shape_data['track']['y_init']

        self.mean_traj = pd.read_feather('/home/la-user/ros2_ws/src/racecar_nn_controller/racecar_nn_controller/mean_trajectory.feather')

        # Convert the DataFrame to a dictionary
        mean_traj_dict = self.mean_traj.rename(columns={
            'x': 'xCoords',
            'y': 'yCoords',
            's': 'arcLength'
        }).to_dict(orient='list') 

        mean_traj_dict = {"track": mean_traj_dict}

        self.mean_traj = mean_traj_dict



        self.last_s = None
        self.lap_start_time = None
        self.lap_time = None
        self.lap_times = []

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
        model_path = os.path.join(package_share_directory, 'models', 'model_noisy_18_64.pth')
        scaler_params_path = os.path.join(package_share_directory, 'models', 'scaling_params_noisy_18.json')
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
        self.ave_flag = True

        print("Model used for neural network control running on", self.device, ":\n")
        print(self.model)


        self.log_data = {}
        self.pure_pursuit_log_data = {}

        # Store the previous `s` value for lap detection
        self.prev_s = None

        # Counter for lap numbers
        self.lap_counter = 0
        self.max_laps = 50

        # Data to track during a lap
        self.current_lap_data = {
            's': [],
            'e': [],
            'dtheta': [],
            'vx': [],
            'vy': [],
            'steering': [],
            'throttle': [],
            "x": [],
            "y": [],
            "heading_angle":[],
            "omega": []
        }

        self.current_pure_pursuit_lap_data = {
            's': [],
            'e': [],
            'dtheta': [],
            'vx': [],
            'vy': [],
            'steering': [],
            'throttle': [],
            "x": [],
            "y": [],
            "heading_angle":[],
            "omega": []
        }

        # Timer for running for 2 minutes (120 seconds)
        self.timer = self.create_timer(1.0, self.check_lap_number)

        # Set a shutdown flag
        self.shutdown_flag = False

        self.Kdd = 0.30

    
    def track_callback(self, msg):
        self.track_data = [msg.x, msg.y, msg.theta]
        self.track_flag = True
        self.destroy_subscription(self.subscription_track)
    	
    def listener_callback(self, msg):

        if self.track_flag:
            
            state_values = np.array([msg.pos_x, msg.pos_y, msg.vel_x, msg.vel_y, wrap_to_pi(msg.turn_angle),  np.clip(msg.turn_rate,-1,1)], dtype = np.float32)

            local_state = transform_to_track(state_values, self.track_data)
                    
            curvilinear_pose = pose_to_curvi(self.track_shape_data, local_state, bring_to_origin=False)
                
            state = np.concatenate([curvilinear_pose, [local_state[3], local_state[4]]], axis=None)

            s = state[0]
            e = state[1]
            dtheta = state[2]
            vx = state[3]
            vy = state[4]

            ##################################################################### calculate pure pursuit command as well but dont give it as command
            v = np.sqrt(local_state[3]**2 + local_state[4]**2)
            deltaS = self.Kdd * v


            # CHANGE THE REFERENCE TRACK TO THE MEAN TRACK HERE LATER
            steering_pure_pursuit, _, _ = pure_pursuit_controller(local_state[0], local_state[1], local_state[2], self.mean_traj["track"], deltaS, s, offset=False, offset_value=0)
            
            steering_pure_pursuit = np.clip(steering_pure_pursuit / np.deg2rad(17), -1, 1)

            ######################################################################

            state_transformed = self.scaler.transform([state])

            state_transformed = torch.Tensor(state_transformed).to(self.device)


            
            with torch.no_grad():
                action = self.model(state_transformed.unsqueeze(0)).cpu().numpy().flatten()
                action = action * self.scale_action + self.mean_action

            throttle = float(action[0])
            steering = float(action[1])

            control_msg = CarControlStamped()
            control_msg.header.stamp = self.get_clock().now().to_msg()
            control_msg.throttle = throttle
            control_msg.steering = steering
            self.publisher_.publish(control_msg)

            # Track values during the lap
            self.current_lap_data['s'].append(s)
            self.current_lap_data['e'].append(e)
            self.current_lap_data['dtheta'].append(dtheta)
            self.current_lap_data['vx'].append(vx)
            self.current_lap_data['vy'].append(vy)
            self.current_lap_data['steering'].append(steering)
            self.current_lap_data['throttle'].append(throttle)
            self.current_lap_data['x'].append(local_state[0])
            self.current_lap_data['y'].append(local_state[1])
            self.current_lap_data['heading_angle'].append(local_state[2])
            self.current_lap_data['omega'].append(local_state[-1])



            self.current_pure_pursuit_lap_data['s'].append(s)
            self.current_pure_pursuit_lap_data['e'].append(e)
            self.current_pure_pursuit_lap_data['dtheta'].append(dtheta)
            self.current_pure_pursuit_lap_data['vx'].append(vx)
            self.current_pure_pursuit_lap_data['vy'].append(vy)
            self.current_pure_pursuit_lap_data['steering'].append(steering_pure_pursuit)
            self.current_pure_pursuit_lap_data['throttle'].append(throttle)
            self.current_pure_pursuit_lap_data['x'].append(local_state[0])
            self.current_pure_pursuit_lap_data['y'].append(local_state[1])
            self.current_pure_pursuit_lap_data['heading_angle'].append(local_state[2])
            self.current_pure_pursuit_lap_data['omega'].append(local_state[-1])


            if self.last_s is not None and self.last_s > 10.5 and s <= 0.05:  # s has crossed zero (lap completed)
                if self.lap_start_time is not None:
                    self.lap_time = time.time() - self.lap_start_time
                    print(f"Lap time: {self.lap_time:.2f} seconds")
                    self.lap_times.append(self.lap_time)
                self.lap_start_time = time.time()  # Restart lap timer

                self.lap_counter += 1
                self.get_logger().info(f"Lap {self.lap_counter} completed.")

                # Save the current lap data under "trajectory_<lap_number>"
                self.log_data[f"trajectory_{self.lap_counter}"] = self.current_lap_data
                self.pure_pursuit_log_data[f"trajectory_{self.lap_counter}"] = self.current_pure_pursuit_lap_data

                # Reset current lap data for the next lap
                self.current_lap_data = {
                    's': [],
                    'e': [],
                    'dtheta': [],
                    'vx': [],
                    'vy': [],
                    'steering': [],
                    'throttle': [],
                    "x": [],
                    "y": [],
                    "heading_angle":[],
                    "omega": []
                }

                self.current_pure_pursuit_lap_data = {
                    's': [],
                    'e': [],
                    'dtheta': [],
                    'vx': [],
                    'vy': [],
                    'steering': [],
                    'throttle': [],
                    "x": [],
                    "y": [],
                    "heading_angle":[],
                    "omega": []
                }

            if len(self.lap_times) == 10 and self.ave_flag == True:
                self.ave_flag = False
                print("Average Lap Time")
                print(np.mean(self.lap_times))

            self.last_s = s

    def check_lap_number(self):
        if not self.shutdown_flag and self.lap_counter > self.max_laps:
            self.get_logger().info("Lap number reached, saving data and shutting down.")
            self.shutdown_flag = True

            # Save data to a Feather file at the end of the 2 minutes
            if self.log_data:
                df = pd.DataFrame.from_dict(self.log_data)
                df_pp = pd.DataFrame.from_dict(self.pure_pursuit_log_data)
                log_path = f'/home/la-user/Imititation-Learning-for-Driving-Control/model18_dist.feather'
                log_path_pp = f'/home/la-user/Imititation-Learning-for-Driving-Control/model18_dist_pure_pursuit.feather'

                df.to_feather(log_path)
                df_pp.to_feather(log_path_pp)

                self.get_logger().info(f"Original data saved to: {log_path}")
                self.get_logger().info(f"Pure Pursuit Data saved to: {log_path_pp}")
            
            # Shut down the node after saving data
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    racecar_nn_controller = RacecarNNController()
    rclpy.spin(racecar_nn_controller)
    racecar_nn_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

