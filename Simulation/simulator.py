import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from plot_track import plot_track
import yaml

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
    
def transform_to_track(state, track_data):
    track_x = track_data[0]
    track_y = track_data[1]
    track_angle = track_data[2]

    x = state[0]
    y = state[1]
    theta = state[2] 

    pos = np.array([x, y])
    track_pos = np.array([track_x, track_y]) 

    T_local_global = np.array([[np.cos(track_angle), np.sin(track_angle)],
                               [-np.sin(track_angle), np.cos(track_angle)]])
    
    pos_local = (T_local_global @ (pos - track_pos).T).T
    turn_angle_local = wrap_to_pi(theta - track_angle)

    return np.concatenate((pos_local, [turn_angle_local]))

class KinematicBicycleSimulator:
    def __init__(self, state, track_data=None):
        self.wheelbase = 0.098
        self.l_r = 0.051
        self.steering_angle_max = np.deg2rad(12.0)
        self.speed_gain = 8.0
        
        self.tau_T = 0.02
        self.tau_delta = 0.02

        self.state = state 
        self.throttle_actual = 0.0
        self.steering_actual = 0.0
        self.track_data = track_data

    def actuator_dynamics(self, steering_desired, throttle_desired, dt):
        # Throttle dynamics: first-order response
        throttle_dot = (throttle_desired - self.throttle_actual) / self.tau_T
        self.throttle_actual += throttle_dot * dt
        
        # Steering dynamics: first-order response
        steering_dot = (steering_desired - self.steering_actual) / self.tau_delta
        self.steering_actual += steering_dot * dt
        
        return self.steering_actual, self.throttle_actual
    
    def dynamics(self, state, steering, throttle, dt=0.005):
        steering_angle = self.steering_angle_max * steering
        V = self.speed_gain * throttle

        theta = state[2]
        slip_angle = np.arctan((self.l_r / self.wheelbase) * np.tan(steering_angle))

        pos_x_dot = V * np.cos(theta + slip_angle)
        pos_y_dot = V * np.sin(theta + slip_angle)
        theta_dot = V / (self.wheelbase) * np.sin(slip_angle)

        return np.array([pos_x_dot, pos_y_dot, theta_dot])

    def rk4_step(self, steering_desired, throttle_desired, dt):
        # Update actual steering and throttle using actuator dynamics

        steering, throttle = self.actuator_dynamics(steering_desired, throttle_desired, dt)
        # steering = steering_desired
        # throttle = throttle_desired

        state = self.state
        k1 = dt * self.dynamics(state, steering, throttle, dt)
        k2 = dt * self.dynamics(state + 0.5 * k1, steering, throttle, dt)
        k3 = dt * self.dynamics(state + 0.5 * k2, steering, throttle, dt)
        k4 = dt * self.dynamics(state + k3, steering, throttle, dt)
        
        self.state = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return np.copy(self.state)

    def simulate(self, throttle, steering, track_data=None, steps=100, dt = None):
        path = []
        for i in range(steps):
            state = self.rk4_step(steering[i], throttle[i], dt[i] if dt is not None else 0.005)
            path.append(state)
        return np.array(path)

if __name__ == "__main__":

    data = pd.read_csv("03_05_2024/driver_b_5/curvilinear_state.csv")

    track_file = os.path.join("04_05_2024/driver_b_7/cleaned_mocap_track_pose2d.csv")
    track_data = pd.read_csv(track_file)

    with open("la_track.yaml", "r") as file:
        track_shape_data = yaml.safe_load(file)

    num_of_step = 1500

    dt = data.iloc[1:num_of_step+1, 0].to_numpy() - data.iloc[0:num_of_step, 0].to_numpy()

    throttle = data['throttle'].to_numpy()[0:num_of_step]
    steering = data['steering'].to_numpy()[0:num_of_step]


    state_0 = np.array([data.iloc[0, 9], data.iloc[0, 10], data.iloc[0, 11]])

    simulator = KinematicBicycleSimulator(state_0, track_data = track_data.iloc[0, :].to_numpy())

    path = simulator.simulate(throttle, steering, track_data=track_data, steps=len(throttle), dt = dt)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    plot_track(fig, ax, track_shape_data)

    plt.plot(path[:, 0], path[:, 1])
    plt.plot(data.iloc[0:num_of_step, 9], data.iloc[0:num_of_step, 10])
    plt.legend(["Track","","", "Simulated", "Actual"])
    plt.show()

