import rclpy
from rclpy.node import Node
from lar_msgs.msg import CarControlStamped, CarStateStamped
from geometry_msgs.msg import Pose2D, Vector3Stamped
from racecar_nn_controller.cartesian_to_curvilinear import pose_to_curvi, curvi_to_pose_offset
from racecar_nn_controller.transform_to_track_frame import transform_to_track, wrap_to_pi
import numpy as np
import pandas as pd
import yaml

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_target_point(x, y, trajectory, deltaS, s, offset, offset_value):
    distances = np.sqrt((trajectory['xCoords'] - x)**2 + (trajectory['yCoords'] - y)**2)
    direction_vectors = np.array([trajectory['xCoords'] - x, trajectory['yCoords'] - y]).T

    forward_indices = [
        i for i in range(len(direction_vectors))
        if trajectory['arcLength'][i] > s + deltaS
    ]

    if not forward_indices:
        target_idx = np.argmin(distances)
        return trajectory['xCoords'][target_idx], trajectory['yCoords'][target_idx]

    target_idx = forward_indices[0]

    if offset:
        x, y = curvi_to_pose_offset(trajectory, [trajectory['arcLength'][target_idx], offset_value], bring_to_origin=False)
    else:
        x = trajectory['xCoords'][target_idx]
        y = trajectory['yCoords'][target_idx]

    return x, y

def pure_pursuit_controller(x, y, heading_angle, trajectory, deltaS, s, offset, offset_value):
    target_x, target_y = compute_target_point(x, y, trajectory, deltaS, s, offset, offset_value)

    dx = target_x - x
    dy = target_y - y
    target_angle = np.arctan2(dy, dx)

    heading_error = target_angle - heading_angle

    if heading_error > np.pi:
        heading_error -= 2 * np.pi
    elif heading_error < -np.pi:
        heading_error += 2 * np.pi

    wheelbase = 0.098
    lookahead_distance = np.sqrt(dx**2 + dy**2)
    steering_angle = np.arctan2(2.0 * wheelbase * np.sin(heading_error), lookahead_distance)

    return steering_angle, target_x, target_y

class PurePursuitControllerNode(Node):
    def __init__(self):
        super().__init__('racecar_nn_controller')

        track_path = "/home/la-user/ros2_ws/src/racecar_nn_controller/racecar_nn_controller/la_track.yaml"
        
        with open(track_path, "r") as file:
            self.track_shape_data = yaml.safe_load(file)

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
        self.control_ref_publisher = self.create_publisher(Vector3Stamped, "/diag/control_ref", 10)

        self.track_flag = False
        self.Kdd = 0.30
        self.offset_value = 0
        self.offset = True

        # Dictionary to store lap data: keys will be 'trajectory_1', 'trajectory_2', etc.
        self.log_data = {}

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
            "x":[],
            "y":[],
            "heading_angle":[]
        }

        # Timer for running for 2 minutes (120 seconds)
        self.timer = self.create_timer(1.0, self.check_lap_number)

        # Set a shutdown flag
        self.shutdown_flag = False

    def offset_change(self):
        self.offset_value = np.random.uniform(-0.35, 0)

    def track_callback(self, msg):
        self.track_data = [msg.x, msg.y, msg.theta]
        self.track_flag = True
        self.destroy_subscription(self.subscription_track)
    
    def listener_callback(self, msg):
        if self.track_flag:
            state_values = np.array([msg.pos_x, msg.pos_y, msg.vel_x, msg.vel_y, wrap_to_pi(msg.turn_angle), np.clip(msg.turn_rate,-1,1)], dtype = np.float32)
            local_state = transform_to_track(state_values, self.track_data)
            curvilinear_pose = pose_to_curvi(self.track_shape_data, local_state, bring_to_origin=False)
            state = np.concatenate([curvilinear_pose, [local_state[3], local_state[4]]], axis=None)

            s = state[0]
            e = state[1]
            dtheta = state[2]
            vx = state[3]
            vy = state[4]

            v = np.sqrt(local_state[3]**2 + local_state[4]**2)
            deltaS = self.Kdd * v

            steering, lookahead_x, lookahead_y = pure_pursuit_controller(local_state[0], local_state[1], local_state[2], self.track_shape_data['track'], deltaS, s, offset=self.offset, offset_value=self.offset_value)
            
            throttle = 0.17
            steering = np.clip(steering / np.deg2rad(17), -1, 1)

            control_msg = CarControlStamped()
            control_msg.header.stamp = self.get_clock().now().to_msg()
            control_msg.throttle = float(throttle)
            control_msg.steering = float(steering)
            self.publisher_.publish(control_msg)

            control_ref_msg = Vector3Stamped()
            control_ref_msg.header.stamp = control_msg.header.stamp
            control_ref_msg.vector.x = lookahead_x
            control_ref_msg.vector.y = lookahead_y
            self.control_ref_publisher.publish(control_ref_msg)

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

            if self.prev_s is not None and self.prev_s > 10.5 and s < 0.05:
                self.lap_counter += 1
                self.offset_change()

                self.get_logger().info(f"Lap {self.lap_counter} completed. Changing offset value to {self.offset_value}.")

                # Save the current lap data under "traj_<lap_number>"
                self.log_data[f"trajectory_{self.lap_counter}"] = self.current_lap_data.copy()

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
                    "heading_angle":[]
                }

            self.prev_s = s

    def check_lap_number(self):
        if not self.shutdown_flag and self.lap_counter > self.max_laps:
            self.get_logger().info("Lap number reached, saving data and shutting down.")
            self.shutdown_flag = True

            # Save data to a Feather file at the end of the 2 minutes
            if self.log_data:
                df = pd.DataFrame.from_dict(self.log_data)
                log_path = f'/home/la-user/Imititation-Learning-for-Driving-Control/pure_pursuit_artificial_df.feather'
                df.to_feather(log_path)
                self.get_logger().info(f"Data saved to: {log_path}")
            
            # Shut down the node after saving data
            rclpy.shutdown()

    def shutdown_callback(self):
        # This method is triggered during the shutdown process, but we don't need it here.
        pass

def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_controller = PurePursuitControllerNode()
    rclpy.spin(pure_pursuit_controller)
    pure_pursuit_controller.destroy_node()

if __name__ == '__main__':
    main()
