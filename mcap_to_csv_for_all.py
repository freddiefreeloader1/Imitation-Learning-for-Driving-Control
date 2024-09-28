
import os
import sys
import pandas as pd
from mcap_ros2.reader import read_ros2_messages

def get_attributes(obj):
    return filter(lambda a: not a.startswith('__'), dir(obj))

def msg_to_dic(msg):
    dic = {'log_time_ns': msg.log_time_ns}
    
    def recursive_traversal(prefix, obj, depth):
        if depth > 5: return
        for attr in get_attributes(obj):
            field = getattr(obj, attr)
            
            if prefix == '':
                name = attr
            else:
                name = f'{prefix}.{attr}'

            if type(field) in [bool, int, float, str]:
                dic[name] = field
            elif type(field) not in [list]:
                recursive_traversal(name, field, depth + 1)
                
    recursive_traversal('', msg.ros_msg, 0)
    return dic

def process_mcap_files(base_dir):
    # List of topics to be saved to CSV
    topic_list = [
        '/car/state',
        '/car/set/control',
        '/mocap/car/pose',
        '/mocap/track/pose2d',
        '/car/state_acc',
    ]
    
    # Loop through each date folder
    for date_folder in os.listdir(base_dir):
        date_path = os.path.join(base_dir, date_folder)
        
        if os.path.isdir(date_path):
            # Loop through each trajectory folder inside the date folder
            for traj_folder in os.listdir(date_path):
                traj_path = os.path.join(date_path, traj_folder)
                
                if os.path.isdir(traj_path):
                    # Get all mcap files in the trajectory folder
                    for mcap_file in os.listdir(traj_path):
                        if mcap_file.endswith(".mcap"):
                            mcap_path = os.path.join(traj_path, mcap_file)
                            print(f"Processing file: {mcap_path}")

                            # Output directory: base_dir/date_folder/traj_folder
                            outdir = os.path.join(base_dir, date_folder, traj_folder)
                            if not os.path.exists(outdir):
                                os.makedirs(outdir)

                            topic_msgs = {key: [] for key in topic_list}
                            for msg in read_ros2_messages(mcap_path, topic_list):
                                topic_msgs[msg.channel.topic].append(msg_to_dic(msg))

                            for topic, msgs in topic_msgs.items():
                                df = pd.DataFrame(msgs)
                                if topic.startswith("/"):
                                    topic = topic[1:]  # Remove leading slash
                                save_path = f'{outdir}/{topic.replace("/", "_")}.csv'
                                df.to_csv(save_path, index=False)
                                print(f"Saved: {save_path}")
                        else:
                            print(f"Skipping non-mcap file: {mcap_file}")

if __name__ == '__main__':
    base_directory = "./" 
    process_mcap_files(base_directory)
