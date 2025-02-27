import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from lar_msgs.msg import CarControlStamped, CarStateStamped
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Joy

from racecar_nn_controller.cartesian_to_curvilinear import pose_to_curvi, wrap_to_pi
from racecar_nn_controller.transform_to_track_frame import transform_to_track
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


# define the NN architecture
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

        # frequency of adding noise to states in runtime
        self.declare_parameter('noise_frequency', 1) 
        self.noise_frequency = self.get_parameter('noise_frequency').value


        # sim or real? 
        self.declare_parameter('mode', "sim") 
        self.mode = self.get_parameter('mode').value

        self.message_counter = 0

        # open the track
        track_path = "/home/la-user/ros2_ws/src/racecar_nn_controller/racecar_nn_controller/la_track.yaml"
        with open(track_path, "r") as file:
            self.track_shape_data = yaml.safe_load(file)

        self.track_shape_data['track']['xCoords'] = np.array(self.track_shape_data['track']['xCoords']) - self.track_shape_data['track']['x_init']
        self.track_shape_data['track']['yCoords'] = np.array(self.track_shape_data['track']['yCoords']) - self.track_shape_data['track']['y_init']

        
        # extract the mean trajectory
        self.mean_traj = pd.read_feather('/home/la-user/ros2_ws/src/racecar_nn_controller/racecar_nn_controller/mean_trajectory.feather')

        # Convert the DataFrame to a dictionary
        mean_traj_dict = self.mean_traj.rename(columns={
            'x': 'xCoords',
            'y': 'yCoords',
            's': 'arcLength'
        }).to_dict(orient='list') 

        mean_traj_dict = {"track": mean_traj_dict}

        self.mean_traj = mean_traj_dict


        # for keeping track of lap numbers and durations 
        self.last_s = None
        self.lap_start_time = None
        self.lap_time = None
        self.lap_times = []

        self.track_data = []

        # subscription to the joy node
        self.subscription_joy = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            1)

        self.joy_command = 0

        print("The mode is: ", self.mode)


        # create subscriptions
        if self.mode == "sim":
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
        else:
            self.subscription_track = self.create_subscription(
            Pose2D,
            '/mocap/track/pose2d',
            self.track_callback,
            1)

            self.subscription = self.create_subscription(
            CarStateStamped,
            '/car/state',
            self.listener_callback,
            1)

            self.publisher_ = self.create_publisher(CarControlStamped, '/car/set/control', 1)

        # Load the trained model
        package_share_directory = get_package_share_directory('racecar_nn_controller')
        model_path = os.path.join(package_share_directory, 'models', 'model_noisy_45_64.pth')
        scaler_params_path = os.path.join(package_share_directory, 'models', 'scaling_params_noisy_45.json')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        # select which ones are included in the trained model
        self.include_omega = True
        self.include_prev_action = True

        # this should always stay false
        self.wrap_omega = False

        # whether to add noise or not in runtime
        self.add_noise = True


        # init the model
        if self.include_omega and self.include_prev_action:
            self.model = SimpleNet(input_size=8, hidden_size=64, output_size=2, input_weights=[5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        elif not self.include_omega and self.include_prev_action:
            self.model = SimpleNet(input_size=7, hidden_size=64, output_size=2, input_weights=[5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0])
        elif self.include_omega and not self.include_prev_action:
            self.model = SimpleNet(input_size=6, hidden_size=64, output_size=2, input_weights=[5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
        elif not self.include_omega and not self.include_prev_action:
            self.model = SimpleNet(input_size=5, hidden_size=64, output_size=2)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        with open(scaler_params_path, 'r') as f:
            scaling_params = json.load(f)

        # for normalization of states and actions 

        mean_state = scaling_params['mean'][0]
        scale_state = scaling_params['std'][0]

        self.mean_action = scaling_params['mean'][1]
        self.scale_action = scaling_params['std'][1]

        self.scaler_action = StandardScaler()
        self.scaler_action.mean_ = self.mean_action 
        self.scaler_action.scale_ = self.scale_action

        self.prev_action = np.array([0.0, 0.0])

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

        # Timer for checking the current lap number 
        self.timer = self.create_timer(1.0, self.check_lap_number)

        # Set a shutdown flag
        self.shutdown_flag = False



        # pure pursuit lookahead parameter
        self.Kdd = 0.30

    
    def joy_callback(self, msg):
        self.joy_command = msg.axes[1]

    def track_callback(self, msg):
        self.track_data = [msg.x, msg.y, msg.theta]
        self.track_flag = True
        self.destroy_subscription(self.subscription_track)
    	
    def listener_callback(self, msg):

        if self.track_flag:

            self.message_counter += 1


            # get raw state
            state_values = np.array([msg.pos_x, msg.pos_y, msg.vel_x, msg.vel_y, wrap_to_pi(msg.turn_angle),  msg.turn_rate], dtype = np.float32)

            # transform to track frame
            local_state = transform_to_track(state_values, self.track_data)

            noise_weight_coeff = 0.01

            noised_local_state = local_state.copy()

            # apply noise to the local states
            if self.message_counter % self.noise_frequency == 0 and self.add_noise:
                noised_local_state[0] = noised_local_state[0] + np.random.normal(0, 0.5) * noise_weight_coeff
                noised_local_state[1] = noised_local_state[1] + np.random.normal(0, 0.5) * noise_weight_coeff
                noised_local_state[2] = noised_local_state[2] + np.random.normal(0, 2) * noise_weight_coeff
                noised_local_state[-1] = noised_local_state[-1] + np.random.normal(0, 1.5) * noise_weight_coeff

            
            # transform to curvilinear frame
            curvilinear_pose_noised = pose_to_curvi(self.track_shape_data, noised_local_state, bring_to_origin=False)
            curvilinear_pose_real = pose_to_curvi(self.track_shape_data, local_state, bring_to_origin=False)

            # wrap omega should be false, it is from one of the previous experiments
            if self.include_omega:
                if self.wrap_omega:
                    state = np.concatenate([curvilinear_pose_noised, [local_state[3], local_state[4]], wrap_to_pi(local_state[-1])], axis=None)
                    state[0] = curvilinear_pose_real[0]
                else:
                    state = np.concatenate([curvilinear_pose_noised, [local_state[3], local_state[4]],local_state[-1]], axis=None)
                    state[0] = curvilinear_pose_real[0]

                # noise coeffs for different variables
                if self.include_prev_action:
                    noise_coeff = [0.5, 0.7, 0.5, 0.2, 0.1, 0.1, 0.005, 0.04]
                else:
                    noise_coeff = [0.5, 0.7, 0.1, 0.2, 0.1, 0.1]
            else:
                state = np.concatenate([curvilinear_pose_noised, [local_state[3], local_state[4]]], axis=None)
                state[0] = curvilinear_pose_real[0]
                if self.include_prev_action:
                    noise_coeff = [0.5, 0.7, 0.1, 0.2, 0.1, 0.005, 0.04]
                else:
                    noise_coeff = [1, 0.7, 0.5, 0.2, 0.1]

            s = curvilinear_pose_real[0]
            e = curvilinear_pose_real[1]
            dtheta = curvilinear_pose_real[2]
            vx = state[3]
            vy = state[4]
            omega = local_state[-1]

            state[2] = wrap_to_pi(state[2])

            if self.include_prev_action:
                state = np.concatenate((state, self.prev_action))

            if self.add_noise:
                state = state + np.random.normal(0, 0.2, len(state)) * noise_coeff

            ##################################################################### calculate pure pursuit command as well but dont give it as command
            v = np.sqrt(local_state[3]**2 + local_state[4]**2)
            deltaS = self.Kdd * v


            # CHANGE THE REFERENCE TRACK TO THE MEAN TRACK HERE LATER
            steering_pure_pursuit, _, _ = pure_pursuit_controller(local_state[0], local_state[1], local_state[2], self.mean_traj["track"], deltaS, s, offset=False, offset_value=0)
            
            steering_pure_pursuit = np.clip(steering_pure_pursuit / np.deg2rad(17), -1, 1)

            ###################################################################### transform the state to normalize it

            state_transformed = self.scaler.transform([state])

            state_transformed = torch.Tensor(state_transformed).to(self.device)

            # calculate the actions 
            with torch.no_grad():
                action = self.model(state_transformed.unsqueeze(0)).cpu().numpy().flatten()
                action = action * self.scale_action + self.mean_action

            throttle = float(action[0]) 

            # throttle = max(throttle, 0.12)

            # throttle = min(throttle, self.joy_command)

            throttle = np.clip(throttle, -0.6, 0.25)

            steering = np.clip(float(action[1]), -1, 1)

            self.prev_action = np.array([throttle, steering])

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
            self.current_lap_data['omega'].append(omega)



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
            self.current_pure_pursuit_lap_data['omega'].append(omega)


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

            # Save data to a Feather file if max laps is achieved
            if self.log_data:
                df = pd.DataFrame.from_dict(self.log_data)
                df_pp = pd.DataFrame.from_dict(self.pure_pursuit_log_data)
                log_path = f'/home/la-user/Imititation-Learning-for-Driving-Control/Obtained Model Data/model45_dist_wrapped.feather'
                log_path_pp = f'/home/la-user/Imititation-Learning-for-Driving-Control/Obtained Model Data/model45_dist_pure_pursuit_wrapped.feather'

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

