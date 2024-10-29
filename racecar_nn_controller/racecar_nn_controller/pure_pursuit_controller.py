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

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_target_point(x, y, vehicle_theta, trajectory, deltaS, s):

    distances = np.sqrt((trajectory['xCoords'] - x)**2 + (trajectory['yCoords'] - y)**2)

    direction_vectors = np.array([trajectory['xCoords'] - x, trajectory['yCoords'] - y]).T
    
    forward_indices = [
        i for i in range(len(direction_vectors))
        if trajectory['arcLength'][i] > s + deltaS
    ]

    if not forward_indices:
        target_idx = np.argmin(distances)
        return trajectory['xCoords'][target_idx], trajectory['yCoords'][target_idx]

    if not forward_indices:
        target_idx = forward_indices[-1]
        return trajectory['xCoords'][target_idx], trajectory['yCoords'][target_idx]

    target_idx = forward_indices[0]
    return trajectory['xCoords'][target_idx], trajectory['yCoords'][target_idx]

def pure_pursuit_controller(x, y, heading_angle, trajectory, deltaS, s):

    target_x, target_y = compute_target_point(x, y, heading_angle, trajectory, deltaS, s)

    dx = target_x - x
    dy = target_y - y
    target_angle = np.arctan2(dy, dx)

    heading_error = target_angle - heading_angle
   
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    
    wheelbase = 0.098

    lookahead_distance = np.sqrt(dx**2 + dy**2)
    
    steering_angle = np.arctan2(2.0 * wheelbase * np.sin(heading_error), lookahead_distance)
    
    return steering_angle, target_x, target_y
    
      
class PurePursuitControllerNode(Node):
    def __init__(self):
        super().__init__('racecar_nn_controller')

        track_path = "/root/ros2_ws/install/lar_utils/share/lar_utils/config/track/sampled/la_track.yaml"
        with open(track_path, "r") as file:
            self.track_shape_data = yaml.safe_load(file)

        # bring to origin

        self.track_shape_data['track']['xCoords'] = np.array(self.track_shape_data['track']['xCoords']) - self.track_shape_data['track']['x_init']
        self.track_shape_data['track']['yCoords'] = np.array(self.track_shape_data['track']['yCoords']) - self.track_shape_data['track']['y_init']

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

        self.track_flag = False

        self.Kdd = 0.08

    
    def track_callback(self, msg):
        self.track_data = [msg.x, msg.y, msg.theta]
        self.track_flag = True
        self.destroy_subscription(self.subscription_track)
    	
    def listener_callback(self, msg):

        if self.track_flag:
            
            state_values = np.array([msg.pos_x, msg.pos_y, msg.vel_x, msg.vel_y, wrap_to_pi(msg.turn_angle),  np.clip(msg.turn_rate,-1,1)], dtype = np.float32)

            local_state = transform_to_track(state_values, self.track_data)
                    
            curvilinear_pose = pose_to_curvi(self.track_shape_data, local_state, bring_to_origin = False)
                
            state = np.concatenate([curvilinear_pose, [local_state[3], local_state[4]]], axis=None)

            s = state[0]
            v = np.sqrt(local_state[3]**2 + local_state[4]**2)
            deltaS = self.Kdd * v

            steering, _ , _ = pure_pursuit_controller(local_state[0], local_state[1], local_state[2], self.track_shape_data['track'], deltaS, s)
            
            throttle = np.random.normal(0.19, 0.09)

            control_msg = CarControlStamped()
            control_msg.header.stamp = self.get_clock().now().to_msg()
            control_msg.throttle = float(throttle)
            control_msg.steering = float(steering)
            self.publisher_.publish(control_msg)

def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_controller = PurePursuitControllerNode()
    rclpy.spin(pure_pursuit_controller)
    pure_pursuit_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

