import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from plot_track import plot_track
import yaml
from cartesian_to_curvilinear_for_one import pose_to_curvi
from matplotlib.animation import FuncAnimation

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_target_point(x, y, vehicle_theta, trajectory, deltaS, s):

    heading_vector = np.array([np.cos(vehicle_theta), np.sin(vehicle_theta)])

    distances = np.sqrt((trajectory['xCoords'] - x)**2 + (trajectory['yCoords'] - y)**2)

    direction_vectors = np.array([trajectory['xCoords'] - x, trajectory['yCoords'] - y]).T
    
    forward_indices = [
        i for i in range(len(direction_vectors))
        if trajectory['arcLength'][i] > s + deltaS
    ]

    if not forward_indices:
        target_idx = np.argmin(distances)
        return trajectory['xCoords'][target_idx], trajectory['yCoords'][target_idx]

    forward_points = np.array([trajectory['xCoords'][i], trajectory['yCoords'][i]] for i in forward_indices)
    forward_distances = distances[forward_indices]

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


class KinematicBicycleSimulator:
    def __init__(self, state, track_data=None, trajectory=None):
        self.wheelbase = 0.098
        self.l_r = 0.051
        self.steering_angle_max = np.deg2rad(12.0)
        self.speed_gain = 8.0
        
        self.tau_T = 0.02
        self.tau_delta = 0.02

        self.state = state 
        self.state_dot = np.zeros_like(state)
        self.throttle_actual = 0.0
        self.steering_actual = 0.0
        self.track_data = track_data
        self.trajectory = trajectory        

        self.kp_v = 0.9
        self.desired_velocity = 1.0 

    def set_desired_velocity(self, desired_velocity):
        self.desired_velocity = desired_velocity
    
    def actuator_dynamics(self, steering_desired, throttle_desired, dt):
        throttle_dot = (throttle_desired - self.throttle_actual) / self.tau_T
        self.throttle_actual += throttle_dot * dt
        
        steering_dot = (steering_desired - self.steering_actual) / self.tau_delta
        self.steering_actual += steering_dot * dt
        
        return self.steering_actual, self.throttle_actual

    def velocity_controller(self, current_velocity):
        velocity_error = self.desired_velocity - current_velocity
        throttle_output = self.kp_v * velocity_error
    
        throttle_output = np.clip(throttle_output, 0.0, 1.0)
        
        return throttle_output
    
    def dynamics(self, state, steering, throttle, dt=0.005):

        steering_angle =  self.steering_angle_max * steering

        V = self.speed_gain * throttle

        theta = state[2]
        slip_angle = np.arctan((self.l_r / self.wheelbase) * np.tan(steering_angle))

        pos_x_dot = V * np.cos(theta + slip_angle)
        pos_y_dot = V * np.sin(theta + slip_angle)
        theta_dot = V / (self.wheelbase) * np.cos(slip_angle) * np.tan(steering_angle)

        self.state_dot = np.array([pos_x_dot, pos_y_dot, theta_dot])

        return np.array([pos_x_dot, pos_y_dot, theta_dot])

    def rk4_step(self, steering_desired, throttle_desired, dt, actuator_dynamics=True):
        if actuator_dynamics:
            steering, throttle = self.actuator_dynamics(steering_desired, throttle_desired, dt)
        else:
            steering = steering_desired
            throttle = throttle_desired
        
        state = self.state
        k1 = dt * self.dynamics(state, steering, throttle, dt)
        k2 = dt * self.dynamics(state + 0.5 * k1, steering, throttle, dt)
        k3 = dt * self.dynamics(state + 0.5 * k2, steering, throttle, dt)
        k4 = dt * self.dynamics(state + k3, steering, throttle, dt)
        
        self.state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return np.copy(self.state)

    def simulate(self, track_data=None, steps=100, dt=None, Kdd=1.0, throttle_data=None):
        path = []
        target_points = []
        
        for i in range(steps):
            x, y, heading_angle = self.state
            V = np.linalg.norm(self.state_dot[0:2])

            # Pure Pursuit for steering angle
            deltaS = Kdd * V

            s = pose_to_curvi(self.trajectory, self.state, bring_to_origin=False)[0]

            steering_angle, target_x, target_y = pure_pursuit_controller(x, y, heading_angle, self.trajectory['track'], deltaS, s)

            steering = np.clip(steering_angle/np.deg2rad(12), -1, 1)

            target_points.append((target_x, target_y))
            
            if throttle_data is not None:
                throttle_desired = throttle_data[i]
            else:
                throttle_desired = self.velocity_controller(V)

            print(f"Throttle Desired: {throttle_desired}")

            state = self.rk4_step(steering, throttle_desired, dt[i] if dt is not None else 0.005, actuator_dynamics=True)
            path.append(state)

        # Convert path and target points to numpy arrays for easier plotting
        path = np.array(path)
        target_points = np.array(target_points)
        
        return path, target_points

def animate_simulation(path, fig, ax, target_points, track_shape_data):

    line, = ax.plot([], [], 'b-', label="Simulated Path")
    target_scatter = ax.scatter([], [], color='red', label="Target Points")
    
    def init():
        line.set_data([], [])
        target_scatter.set_offsets(np.empty((0, 2)))
        return line, target_scatter

    def update(frame):
        line.set_data(path[:frame, 0], path[:frame, 1])
        target_scatter.set_offsets(target_points[:frame, :])
        return line, target_scatter

    ani = FuncAnimation(fig, update, frames=len(path), init_func=init, blit=True, interval=0.1)
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv("03_05_2024/driver_b_5/curvilinear_state.csv")

    track_file = os.path.join("04_05_2024/driver_b_7/cleaned_mocap_track_pose2d.csv")
    track_data = pd.read_csv(track_file)

    with open("la_track.yaml", "r") as file:
        track_shape_data = yaml.safe_load(file)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    plot_track(fig, ax, track_shape_data)

    throttle = data['throttle'].to_numpy()

    num_of_step = 1500

    state_0 = np.array([1.1, -0.5, -0.2])

    track_shape_data['track']['xCoords'] = np.array(track_shape_data['track']['xCoords']) - track_shape_data['track']['x_init']
    track_shape_data['track']['yCoords'] = np.array(track_shape_data['track']['yCoords']) - track_shape_data['track']['y_init']

    simulator = KinematicBicycleSimulator(state_0, track_data=track_data.iloc[0, :].to_numpy(), trajectory=track_shape_data)

    simulator.set_desired_velocity(1.0)

    path, target_points = simulator.simulate(track_data=track_data, steps=num_of_step, dt=None, Kdd = 0.25, throttle_data=throttle)

    animate_simulation(path, fig, ax , target_points, track_shape_data)






