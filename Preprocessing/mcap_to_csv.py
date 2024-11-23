#
#
# Python dependencies:
# pip3 install pandas mcap-ros2-support
#
#
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

def main():

    # maps topic to csv file
    topic_list = [
        '/car/state',
        '/car/set/control',
        '/mocap/car/pose',
        '/mocap/track/pose2d',
        '/car/state_acc',
    ]

    import os
    current_dir = os.getcwd()

    mcap_in = sys.argv[1]

    if mcap_in.endswith(".mcap"):
        # Got a single file
        if os.path.isfile(mcap_in):
            # Absolute filepath given
            print('mcap file given: ' + mcap_in)
            mcaplist = [mcap_in]
        elif os.path.isfile(current_dir + "/" + mcap_in):
            # Relative filepath given
            mcap_in = current_dir + "/" + mcap_in
            mcaplist = [mcap_in]
            print('mcap file given: ' + mcap_in)
    else:
        # Got a directory
        if os.path.isdir(mcap_in):  
            # Absolute directory path given, keep mcap_in
            mcap_dir = mcap_in
            print('mcap directory given: ' + mcap_dir)
            mcaplist = os.listdir(mcap_dir)

        elif os.path.isdir(current_dir + "/" + mcap_in):
            # Relative directory path given
            mcap_dir = current_dir + "/" + mcap_in
            print('mcap directory given: ' + mcap_dir)
            mcaplist = os.listdir(mcap_dir)

    for mcap_file in mcaplist:
        if mcap_file.endswith(".mcap"):

            if 'mcap_dir' in locals():
                mcap_file = mcap_dir + "/" + mcap_file

            print('Parsing file: ' + mcap_file)
            mcap_filename = os.path.splitext(mcap_file)[0]
            
            outdir = mcap_filename
            if not os.path.exists(outdir): # Only process new mcaps
            
                topic_msgs = {key:[] for key in topic_list}
                for msg in read_ros2_messages(mcap_file, topic_list):
                    topic_msgs[msg.channel.topic].append(msg_to_dic(msg))
                
                os.makedirs(outdir)
                for topic, msgs in topic_msgs.items():
                    df = pd.DataFrame(msgs)
                    if topic.startswith("/"): topic = topic[1:]
                    save_path = f'{outdir}/{topic.replace("/", "_")}.csv'
                    df.to_csv(save_path, index=False)
                    print('-> ' + save_path)
                    
            else:
                print('-- ignored (exists already)')


if __name__ == '__main__':
    main()
