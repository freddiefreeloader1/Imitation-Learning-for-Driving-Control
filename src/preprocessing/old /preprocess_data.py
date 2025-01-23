import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from plot_track import plot_track
from filter_data import split_data
import yaml
import os 


from cartesian_to_curvilinear import find_closest_point, pose_to_curvi  

def diffmid(data, time):
    data = np.asarray(data)
    time = np.asarray(time)

    mask = ~np.isnan(time) & ~np.isnan(data)
    data = data[mask]
    time = time[mask]

    dt = time[2:] - time[:-2]  

    diff_data = (data[2:] - data[:-2]) / dt 
    
    time_mid = time[1:-1]

    return diff_data, time_mid

def unwrap_from(data, period, offset):  
    # check the matlab function here
    return np.unwrap(data + offset) - offset

def first_order_filter(time_series, signal_series, tau):
    time = time_series.values
    signal = signal_series.values

    dt = np.diff(time, prepend=time[0])
    alpha = dt / (tau + dt)
    filt_signal = np.zeros_like(signal)
    for i in range(1, len(signal)):
        filt_signal[i] = alpha[i] * signal[i] + (1 - alpha[i]) * filt_signal[i-1]
    
    return pd.Series(filt_signal, index=signal_series.index)

def second_order_filter(time, signal, tau):
    b, a = butter(2, 1 / tau, fs=1/np.mean(np.diff(time)))
    return filtfilt(b, a, signal)

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def compute_odometer(positions):
    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)  
    odometer = np.concatenate(([0], np.cumsum(distances))) 
    return odometer

def process_drivelog_csv(file_name, track_pos=[0, 0], track_angle=0, plot=True, track_data=None):
    data = pd.read_csv(file_name)

    # get the track pose right, read it from the other ros topic, 3rd day doesnt have it so use the second day


    # Extract data from the csv file
    time = data['time']
    mocap_pos = data[['mocap_pos_x', 'mocap_pos_y']].values
    state_vel = data[['state_vel_x', 'state_vel_y']].values
    turn_angle = data['turn_angle'].values
    turn_rate = data['turn_rate'].values
    throttle = data['throttle'].values
    steering = data['steering'].values


    # transform the data to the local frame of the track
    print(f"Transforming onto track (pos: {track_pos}, angle: {np.rad2deg(track_angle)})")
    
    T_local_global = np.array([[np.cos(track_angle), np.sin(track_angle)],
                               [-np.sin(track_angle), np.cos(track_angle)]])
    
    pos_local = (T_local_global @ (mocap_pos - track_pos).T).T
    turn_angle_local = wrap_to_pi(turn_angle - track_angle)
    


    transformed_data = pd.DataFrame({
        'time': time,
        'mocap_pos_x_local': pos_local[:, 0],
        'mocap_pos_y_local': pos_local[:, 1],
        'state_vel_x': state_vel[:, 0],
        'state_vel_y': state_vel[:, 1],
        'turn_angle_local': turn_angle_local,
        'turn_rate': turn_rate,
        'throttle': throttle,
        'steering': steering
    })


    # Filter the data
    filtered = {}
    filtered['raw'] = {
        'Time': time,
        'x': pos_local[:, 0],
        'y': pos_local[:, 1],
        'theta': turn_angle_local
    }

    # Compute mid-point differences to get velocities
    filtered['dmid'] = {}
    filtered['dmid']['dx'], _ = diffmid(filtered['raw']['x'], filtered['raw']['Time'])
    filtered['dmid']['dy'], _ = diffmid(filtered['raw']['y'], filtered['raw']['Time'])
    filtered['dmid']['dtheta'], filtered['dmid']['Time'] = diffmid(
        unwrap_from(filtered['raw']['theta'] + np.pi, 2*np.pi, np.pi) - np.pi, filtered['raw']['Time']
    )

    # Smooth velocities
    # double check the filter and compare with matlab, plot and overlay the two
    filtered['dsmooth'] = {}
    filtered['dsmooth']['Time'] = filtered['dmid']['Time']
    filtered['dsmooth']['dx'] = uniform_filter1d(filtered['dmid']['dx'], size=7)
    filtered['dsmooth']['dy'] = uniform_filter1d(filtered['dmid']['dy'], size=7)
    filtered['dsmooth']['dtheta'] = uniform_filter1d(filtered['dmid']['dtheta'], size=11)

    # Compute velocity magnitude
    filtered['dmid']['dxy'] = np.linalg.norm([filtered['dmid']['dx'], filtered['dmid']['dy']], axis=0)
    transformed_data['velocity'] = np.linalg.norm([transformed_data['state_vel_x'], transformed_data['state_vel_y']], axis=0)   

    # compute odometer readings
    # odo_raw should be coming from the mocap and odo should be coming from the kalman filter
    transformed_data['odo_raw'] = compute_odometer(data[['mocap_pos_x', 'mocap_pos_y']].values)
    
    # Compute odometer for car_state (state_vel_x, state_vel_y)
    transformed_data['odo'] = compute_odometer(transformed_data[['mocap_pos_x_local', 'mocap_pos_y_local']].values)

    # correct for input communication delay
    transformed_data['Time'] = transformed_data['time']  + 0.03

    # Unwrap turn angle
    transformed_data['turn_angle_local'] = unwrap_from(transformed_data['turn_angle_local'], 2*np.pi, np.pi) - np.pi
    
    inputTT = transformed_data[['Time', 'steering', 'throttle']].set_index('Time')
    mocap_poseTT = transformed_data[['Time', 'mocap_pos_x_local', 'mocap_pos_y_local', 'turn_angle_local']].set_index('Time')
    # use the differentially filtered velocities
    velTT = transformed_data[['Time', 'state_vel_x', 'state_vel_y', 'turn_rate']].set_index('Time')
    carStateTT = transformed_data[['Time', 'state_vel_x', 'state_vel_y']].rename(columns={'state_vel_x': 'velocity', 'state_vel_y': 'odo'}).set_index('Time')

    # Align DataFrames to the same time grid as inputTT

    # check the interpolation method
    mocap_poseTT = mocap_poseTT.reindex(inputTT.index).interpolate(method='linear')
    velTT = velTT.reindex(inputTT.index).interpolate(method='linear')
    carStateTT = carStateTT.reindex(inputTT.index).interpolate(method='linear')

    # Combine all the DataFrames
    syncedTT = pd.concat([mocap_poseTT, inputTT, velTT, carStateTT], axis=1)

    # Filter throttle and steering using first and second-order filters
    syncedTT['steering_filt'] = second_order_filter(syncedTT.index, syncedTT['steering'], tau=0.02)
    syncedTT['throttle_filt'] = first_order_filter(syncedTT.index, syncedTT['throttle'], tau=0.015)
    
    # Wrap the turn angles to stay within [-pi, pi]
    syncedTT['turn_angle_local'] = wrap_to_pi(syncedTT['turn_angle_local'])

    # Compute new velocities in the body frame
    v_world = np.vstack([syncedTT['state_vel_x'], syncedTT['state_vel_y']])
    v_body = np.zeros_like(v_world)
    
    for i in range(len(syncedTT)):
        angle = syncedTT['turn_angle_local'].iloc[i]
        T_body_world = np.array([[np.cos(angle), np.sin(angle)],
                                 [-np.sin(angle), np.cos(angle)]])
        v_body[:, i] = np.dot(T_body_world, v_world[:, i])

    # Add body frame velocities to syncedTT DataFrame
    syncedTT['vx'] = v_body[0, :]
    syncedTT['vy'] = v_body[1, :]
    syncedTT = syncedTT.drop(columns=['state_vel_x', 'state_vel_y'])

    # print(syncedTT.head())

    if plot:

        # plot everything

        fig, ax = plt.subplots(figsize=(10, 10))

        plot_track(fig, ax, track_data)

        ax.plot(syncedTT['mocap_pos_x_local'], syncedTT['mocap_pos_y_local'], 'k', label='Mocap position', linewidth=2)

        ax.set_xlabel('X coordinates')
        ax.set_ylabel('Y coordinates')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    return syncedTT


def process_files_in_folder(folder_path, track_data=None):
    all_states = {}

    for idx, filename in enumerate(os.listdir(folder_path)):
        if filename.startswith("driver_b"):
            file_path = os.path.join(folder_path, filename)
        
            transformed_df = process_drivelog_csv(file_path, track_data=track_data, plot=True)
            curvilinear_pose = pose_to_curvi(track_data, transformed_df)
            
            # ask if the theta should be wrapped to py before filtering? 

            # also append track frame x,y, heading angle, curvature of the track, total curvature (integral of curvature)(same as the tangent angle we are using)
            state = pd.DataFrame({
                's': curvilinear_pose[:, 0],
                'e': curvilinear_pose[:, 1],
                'dtheta': wrap_to_pi(curvilinear_pose[:, 2]),
                'vx': transformed_df['vx'],
                'vy': transformed_df['vy'],
                'omega': transformed_df['turn_rate'],
                'throttle': transformed_df['throttle_filt'],
                'steering': transformed_df['steering_filt'],
            })

            state = split_data(state.iloc[:, :6], state.iloc[:, -2:], threshold=0.5, min_decreasing_steps=5)
            print(state)

            all_states[f"trajectory_{idx}"] = state.to_dict(orient='list')
            
            print(f"Processed: {filename}")

    output_path = "all_trajectories.yaml"
    with open(output_path, "w") as output_file:
        yaml.dump(all_states, output_file)
    
    print(f"All trajectories saved to: {output_path}")

with open("la_track.yaml", "r") as file:
    track_data = yaml.safe_load(file)

process_files_in_folder("Dataset_global", track_data=track_data)