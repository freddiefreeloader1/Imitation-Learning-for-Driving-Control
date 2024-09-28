import os
import pandas as pd

base_folder = './'
date_folders = ['03_05_2024', '04_05_2024', '05_05_2024']

topics_to_read = {
    'mocap_car_pose': {
        'time': 'time',  
        'pose.position.x': 'x_raw',
        'pose.position.y': 'y_raw',
        'pose.orientation.w': 'qw_raw',
        'pose.orientation.x': 'qx_raw',
        'pose.orientation.y': 'qy_raw',
        'pose.orientation.z': 'qz_raw'
    },
    'mocap_track_pose2d': {
        'x': 'x_raw',
        'y': 'y_raw',
        'theta': 'theta_raw'
    },
    'car_state': {
        'pos_x': 'x',
        'pos_y': 'y',
        'vel_x': 'vx',
        'vel_y': 'vy',
        'turn_angle': 'theta',
        'turn_rate': 'rate'
    },
    'car_set_control': {
        'time': 'time',
        'steering': 'steering',
        'throttle': 'throttle'
    }
}

time_ref = 0 
time_scale = 1e9  
time_offset = 0  

for date_folder in date_folders:
    date_path = os.path.join(base_folder, date_folder)
    trajectory_folders = os.listdir(date_path)
    
    for trajectory in trajectory_folders:
        trajectory_path = os.path.join(date_path, trajectory)
        
        # Iterate through each topic's CSV file
        topic_idx = 0
        time_ref = 0
        for topic, var_map in topics_to_read.items():
            csv_file = os.path.join(trajectory_path, f"{topic.replace('/', '_')}.csv")
            
            if os.path.basename(csv_file).startswith("cleaned"):
                print(f"Skipping already cleaned file: {csv_file}")
                continue
            
            if os.path.exists(csv_file):
                print(f"Processing {csv_file}")
                
                # Load the CSV into a DataFrame
                try: 
                    df = pd.read_csv(csv_file)
                except pd.errors.EmptyDataError:
                    print(f"Empty CSV file: {csv_file}")
                    continue

                if 'header.stamp.sec' in df.columns and 'header.stamp.nanosec' in df.columns and topic_idx == 0:
                    time_scale = 1
                    df['header_stamp'] = (df['header.stamp.sec'] * 1e9 + df['header.stamp.nanosec'])
                    df['header_stamp'] = (df['header_stamp'] - time_ref) / (time_scale + time_offset)

                    time_ref = df['header_stamp'][0]
                    topic_idx += 1

                if 'header.stamp.sec' in df.columns and 'header.stamp.nanosec' in df.columns:
                    time_scale = 1e9
                    df['time'] = (df['header.stamp.sec'] * 1e9 + df['header.stamp.nanosec'])
                    df['time'] = (df['time'] - time_ref) / (time_scale + time_offset)

                # Filter the DataFrame to only include the necessary columns
                filtered_columns = {original: new for original, new in var_map.items() if original in df.columns}
                df_clean = df[list(filtered_columns.keys())].rename(columns=filtered_columns)
                
                # Save the cleaned DataFrame to a new CSV file
                cleaned_csv_file = os.path.join(trajectory_path, f"cleaned_{topic.replace('/', '_')}.csv")
                df_clean.to_csv(cleaned_csv_file, index=False)
                print(f"Saved cleaned data to {cleaned_csv_file}")
