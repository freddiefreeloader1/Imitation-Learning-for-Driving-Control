
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.signal import butter, filtfilt
from plot_track import plot_track
from filter_data import split_data
import yaml
import os 

from cartesian_to_curvilinear import find_closest_point, pose_to_curvi  


def plot_state(state, track_shape_data, animate=False):
    # Set up the figure and subplots
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    fig.tight_layout(pad=3.0)

    # Extract columns from the DataFrame
    columns = ['s', 'e', 'dtheta', 'vx', 'vy', 'omega', 'throttle', 'steering', 'curvature', 'x', 'heading_angle', 'total_curvature']

    # Plotting function for animation
    def update(frame):
        for i, column in enumerate(columns):
            ax = axes[i // 3, i % 3]
            ax.clear() 
            if column != 'x' and column != 'y' and column != 'curvature' and column != 'total_curvature':
                ax = axes[i//3, i%3]
                ax.plot(state.index[:frame], state[column].values[:frame])
                ax.set_title(column)
                ax.set_xlabel('time')
                ax.set_ylabel(column)
            elif column == 'x':
                ax.plot(state['x'][:frame], state['y'][:frame])
                plot_track(fig, ax, track_shape_data)
                ax.set_title('x-y')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
            elif column == 'curvature':
                ax.plot(track_shape_data['track']['curvature'][:frame])
                ax.set_title('curvature')
                ax.set_xlabel('time')
                ax.set_ylabel('curvature')
            elif column == 'total_curvature':
                ax.plot(track_shape_data['track']['tangentAngle'][:frame])
                ax.set_title('total curvature')
                ax.set_xlabel('time')
                ax.set_ylabel('total curvature')

        plt.tight_layout()

    if animate:
        # Set up animation
        num_frames = len(state['s'].values)  # Number of frames based on the length of the DataFrame
        ani = FuncAnimation(fig, update, frames=num_frames, interval=10, repeat=False)
        plt.show()
    else:
        # Static plotting (unchanged from original)
        for i, column in enumerate(columns):
            ax = axes[i // 3, i % 3]
            if column != 'x' and column != 'y' and column != 'curvature' and column != 'total_curvature':
                ax.plot(state.index, state[column])
                ax.set_title(column)
                ax.set_xlabel('time')
                ax.set_ylabel(column)
            elif column == 'x':
                ax.plot(state['x'], state['y'])
                plot_track(fig, ax, track_shape_data)
                ax.set_title('x-y')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
            elif column == 'curvature':
                ax.plot(track_shape_data['track']['curvature'])
                ax.set_title('curvature')
                ax.set_xlabel('index')
                ax.set_ylabel('curvature')
            elif column == 'total_curvature':
                ax.plot(track_shape_data['track']['tangentAngle'])
                ax.set_title('total curvature')
                ax.set_xlabel('index')
                ax.set_ylabel('total curvature')

        plt.tight_layout()
        plt.show()

def diffmid(data, time):
    data = np.asarray(data)
    time = np.asarray(time)

    mask = ~np.isnan(time) & ~np.isnan(data)
    data = data[mask]
    time = time[mask]

    diff_data = np.zeros_like(data)
    
    if len(data) < 3 or len(time) < 3:
        return diff_data, time  

    dt = time[2:] - time[:-2]  

    diff_data[1:-1] = (data[2:] - data[:-2]) / dt 
    
    return diff_data[1:-1], time[1:-1]


def unwrap_from(data, period, offset):  
    return np.unwrap(data + offset) - offset

def first_order_filter(time, data, tau):

    time = np.asarray(time)
    
    dt = np.diff(time, prepend=time[0])
    
    u_cmd = data.values  
    u = np.zeros((u_cmd.shape[0] + 1)) 
    u[0] = u_cmd[0]  

    for k in range(u_cmd.shape[0]):
        du = (1.0 / tau) * (u_cmd[k] - u[k])
        u[k+1] = u[k] + du * dt[k]
    
    u = u[1:]

    return u


def second_order_filter(time, signal, tau, K=1, c=1.3):

    dt = np.diff(time, prepend=time[0]) 
    n = len(signal)
    
    u1 = np.zeros(n)
    u2 = np.zeros(n)
    
    u1[0] = signal.iloc[0]  
    u2[0] = 0 
    
    w = 1 / tau  
    for k in range(1, n):
        err = K * signal.iloc[k] - u1[k-1]
                
        du1 = u2[k-1]  
        du2 = w**2 * err - 2 * c * w * u2[k-1]  
        
        # Update state variables using Euler integration (forward integration)
        u1[k] = u1[k-1] + du1 * dt[k]  
        u2[k] = u2[k-1] + du2 * dt[k] 
    
    return u1  

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_odometer(positions):
    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)  
    odometer = np.concatenate(([0], np.cumsum(distances))) 
    return odometer



def process_drivelog_csv(file_name_mocap, file_name_control, track_pos=[0, 0], track_angle=0, plot=True, track_data=None):
    # Load the CSV files
    mocap_data = pd.read_csv(file_name_mocap)
    control_data = pd.read_csv(file_name_control)

    track_pos = track_data[['x_raw', 'y_raw']].values[0]
    track_angle = track_data['theta_raw'].values[0]

    # Extract time, mocap position (from mocap_car_pose) and control inputs (from car_set_control)
    time = mocap_data['time']
    mocap_pos = mocap_data[['x_raw', 'y_raw']].values

    w = mocap_data['qw_raw'].values
    x = mocap_data['qx_raw'].values
    y = mocap_data['qy_raw'].values
    z = mocap_data['qz_raw'].values

    # Convert quaternion to yaw angle (in radians)
    turn_angle =  np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    throttle = control_data['throttle'].values
    steering = control_data['steering'].values

    # Transform the data to the local frame of the track
    print(f"Transforming onto track (pos: {track_pos}, angle: {np.rad2deg(track_angle)})")
    
    T_local_global = np.array([[np.cos(track_angle), np.sin(track_angle)],
                               [-np.sin(track_angle), np.cos(track_angle)]])
    
    pos_local = (T_local_global @ (mocap_pos - track_pos).T).T
    turn_angle_local = wrap_to_pi(turn_angle - track_angle)

    # Create a DataFrame for transformed data
    transformed_data = pd.DataFrame({
        'time': time,
        'x_local': pos_local[:, 0],
        'y_local': pos_local[:, 1],
        'turn_angle_local': turn_angle_local
    })

    # Compute velocities and additional quantities
    filtered_data = pd.DataFrame({
        'time': time,
        'x': transformed_data['x_local'],
        'y': transformed_data['y_local'],
        'theta': transformed_data['turn_angle_local']
    })

    filtered_velocities = pd.DataFrame(
        columns=['dx', 'dy', 'omega', 'dxy', 'time']
    )
      
    # Calculate velocities using finite differences
    filtered_velocities['dx'], vel_time = diffmid(filtered_data['x'], filtered_data['time'])
    filtered_velocities['dy'], _ = diffmid(filtered_data['y'], filtered_data['time'])
    filtered_velocities['omega'], _ = diffmid(unwrap_from(filtered_data['theta'] + np.pi, 2*np.pi, np.pi) - np.pi, filtered_data['time'])

    filtered_velocities['time'] = vel_time

    filtered_velocities.set_index('time', inplace=True)

    # Smooth the velocities (using a simple moving average as an example)
    # filtered_data['dx'] = uniform_filter1d(filtered_data['dx'], size=7)
    # filtered_data['dy'] = uniform_filter1d(filtered_data['dy'], size=7)
    # filtered_data['omega'] = uniform_filter1d(filtered_data['omega'], size=11)

    filtered_velocities['dx'] = filtered_velocities['dx'].rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    filtered_velocities['dy'] = filtered_velocities['dy'].rolling(window=7, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    filtered_velocities['omega'] = filtered_velocities['omega'].rolling(window=11, center=True).mean().fillna(method='bfill').fillna(method='ffill')

    # Compute velocity magnitude
    filtered_velocities['dxy'] = np.sqrt(filtered_velocities['dx']**2 + filtered_velocities['dy']**2)

    # compute odometer
    transformed_data['odom'] = compute_odometer(transformed_data[['x_local', 'y_local']].values)

    control_data['time'] = control_data['time'] + 0.03

    # unwrap the angle
    filtered_data['theta'] = unwrap_from(filtered_data['theta'] + np.pi, 2*np.pi, np.pi) - np.pi

    # syncronize different topics

    inputTT = control_data[['time', 'steering', 'throttle']].set_index('time')
    mocap_poseTT = filtered_data[['time', 'x', 'y', 'theta']].set_index('time')
    velTT = filtered_velocities


    mocap_poseTT = mocap_poseTT.reindex(inputTT.index, method='nearest')
    velTT = velTT.reindex(inputTT.index, method='nearest')

    syncedTT = pd.concat([mocap_poseTT, inputTT, velTT], axis=1)

    # Simulate actuator dynamics (applying filters)
    syncedTT['steering_filt'] = second_order_filter(syncedTT.index, syncedTT['steering'], 0.02)
    syncedTT['throttle_filt'] = first_order_filter(syncedTT.index, syncedTT['throttle'], 0.015)

    # wrap the turn angle again ASK THIS IF THIS IS CORRECT
    syncedTT['theta'] = wrap_to_pi(syncedTT['theta'])
    filtered_data['theta'] = wrap_to_pi(filtered_data['theta'])

    # Compute new velocities in the body frame

    v_world = np.vstack([syncedTT['dx'], syncedTT['dy']])
    v_body = np.zeros_like(v_world)
    
    for i in range(len(syncedTT)):
        angle = syncedTT['theta'].iloc[i]
        T_body_world = np.array([[np.cos(angle), np.sin(angle)],
                                [-np.sin(angle), np.cos(angle)]])
        v_body[:, i] = np.dot(T_body_world, v_world[:, i])

    # Add body frame velocities to syncedTT DataFrame
    syncedTT['vx'] = v_body[0, :]
    syncedTT['vy'] = v_body[1, :]
    syncedTT = syncedTT.drop(columns=['dx', 'dy'])


    if plot:
        lw = 1  

        fig, ax = plt.subplots(5, 1, figsize=(6, 10), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})
        fig.tight_layout(pad=3.0)

        # Plot raw data in xy-plane (Position)
        ax_pos = ax[0]
        ax_pos.set_title('Raw data + resampled + synced')
        ax_pos.plot(syncedTT['x'], syncedTT['y'], linewidth=lw, label='synced')
        ax_pos.plot(syncedTT['x'].iloc[0], syncedTT['y'].iloc[0], 'o', linewidth=lw, label='init')
        ax_pos.plot(filtered_data['x'], filtered_data['y'], linewidth=lw, label='mocap')
        ax_pos.set_xlabel('x (m)')
        ax_pos.set_ylabel('y (m)')
        ax_pos.grid(True)
        ax_pos.legend()

        # Plot turn angle (theta)
        ax[1].plot(syncedTT.index, syncedTT['theta'], linewidth=lw, label='synced')
        ax[1].plot(filtered_data['time'], filtered_data['theta'], linewidth=lw, label='mocap')
        ax[1].set_ylabel('Turn angle (rad)')
        ax[1].grid(True)
        ax[1].legend()

        # Plot velocity in body frame (vx and vy)
        ax[2].plot(syncedTT.index, syncedTT['vx'], 'r', linewidth=lw, label='body vx')
        ax[2].plot(syncedTT.index, syncedTT['vy'], '--r', linewidth=lw, label='body vy')
        ax[2].plot(filtered_data['time'], filtered_data['dx'], 'b', linewidth=lw, label='world vx')
        ax[2].plot(filtered_data['time'], filtered_data['dy'], '--b', linewidth=lw, label='world vy')
        ax[2].set_ylabel('v (body frame)')
        ax[2].grid(True)
        ax[2].legend()

        # Plot turn rate (dtheta/dt)
        ax[3].plot(syncedTT.index, syncedTT['dtheta'], linewidth=lw, label='turn rate')
        ax[3].plot(filtered_data['time'], filtered_data['omega'], linewidth=lw, label='mocap turn rate')
        ax[3].set_ylabel('Turn rate (rad/s)')
        ax[3].grid(True)
        ax[3].legend()

        # Plot filtered steering input
        ax[4].plot(syncedTT.index, syncedTT['steering'], linewidth=lw, label='synced')
        ax[4].plot(syncedTT.index, syncedTT['steering_filt'], linewidth=lw, label='filtered')
        ax[4].set_ylabel('Steering angle')
        ax[4].grid(True)
        ax[4].legend()

        # Show the figure
        plt.show()

    # rearrange the columns 
    syncedTT = syncedTT[['x', 'y', 'theta', 'vx', 'vy', 'omega', 'throttle', 'steering', 'throttle_filt', 'steering_filt']]

    # Assuming syncedTT is a DataFrame and its index represents time
    syncedTT = syncedTT[syncedTT.index >= 0]

    return syncedTT


def process_files_in_folder(track_data=None):
    all_states = {}
    trajectory_count = 0

    trajectory_folders = ['03_05_2024', '04_05_2024', '05_05_2024']

    for folder_path in trajectory_folders:
        for idx, folder_name in enumerate(os.listdir(folder_path)):
            if folder_name.startswith("driver_b"):

                
                folder_path_full = os.path.join(folder_path, folder_name)

                # Construct file paths for mocap_car_pose and car_set_control CSVs
                mocap_file = os.path.join(folder_path_full, "cleaned_mocap_car_pose.csv")
                control_file = os.path.join(folder_path_full, "cleaned_car_set_control.csv")
               
                if folder_path != '05_05_2024':
                    track_file = os.path.join(folder_path_full, "cleaned_mocap_track_pose2d.csv")
                    track_data = pd.read_csv(track_file)
                    
                # Process the data
                transformed_df = process_drivelog_csv(mocap_file, control_file, track_data=track_data, plot=False)

                # save the transformed data 
                # transformed_df.to_csv(os.path.join(folder_path_full, "cleaned_synced_data.csv"), index=True)

                with open("la_track.yaml", "r") as file:
                    track_shape_data = yaml.safe_load(file)


                curvilinear_pose = pose_to_curvi(track_shape_data, transformed_df)

                # Compute and filter the state data as before
                state = pd.DataFrame({
                    'time': transformed_df.index,
                    's': curvilinear_pose[:, 0],
                    'e': curvilinear_pose[:, 1],
                    'dtheta': wrap_to_pi(curvilinear_pose[:, 2]),
                    'vx': transformed_df['vx'],
                    'vy': transformed_df['vy'],
                    'omega': transformed_df['omega'],
                    'throttle': transformed_df['throttle_filt'],
                    'steering': transformed_df['steering_filt'],
                    'x': transformed_df['x'],
                    'y': transformed_df['y'],
                    'heading_angle': transformed_df['theta'],
                }).set_index('time')

                plot_state(state, track_shape_data)

                # state.to_csv(os.path.join(folder_path_full, "curvilinear_state.csv"), index=True)

                # Apply filtering/splitting logic
                filtered_trajectories_in_curvilinear = split_data(pd.concat([state.iloc[:, :6], state.iloc[:, -3:]], axis=1), state.iloc[:, 6:8], threshold=0.2, min_decreasing_steps=5)

                # for trajectory in filtered_trajectories_in_curvilinear:
                #     plot_state(trajectory, track_shape_data)

                for trajectory in filtered_trajectories_in_curvilinear:
                    trajectory_name = f"trajectory_{trajectory_count}"
                    print(f"Processing: {trajectory_name}")
                    if len(trajectory) > 150:
                        all_states[trajectory_name] = trajectory.to_dict(orient='list')
                    trajectory_count += 1 
                
                print(state)
                print(f"Processed: {folder_name}")

            output_path = "D:/all_trajectories.yaml"
            # with open(output_path, "w") as output_file:
            #     yaml.dump(all_states, output_file)

            print(f"All trajectories saved to: {output_path}")


if __name__ == "__main__":
    process_files_in_folder()