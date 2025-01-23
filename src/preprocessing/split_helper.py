import numpy as np
import pandas as pd


#####################################################    SPLIT TRAJECTORY BASED ON JUMPING CRITERIA
def split_data(data, u, threshold, min_decreasing_steps=5):
    S = data['s'].values
    num_points = len(S)

    jumps = np.where(np.abs(np.diff(S)) > threshold)[0]

    # Initialize a list to store each trajectory separately
    trajectory_struct = []
    start_idx = 0

    for jump_idx in jumps:
        end_idx = jump_idx + 1  

        segment_S = S[start_idx:end_idx]
        if not has_decreasing_segment(segment_S, min_decreasing_steps):
            full_state_segment = data.iloc[start_idx:end_idx, :]
            u_segment = u.iloc[start_idx:end_idx, :]
            concataneted_segment = pd.concat([full_state_segment, u_segment], axis=1)
            trajectory_struct.append(concataneted_segment)

        start_idx = end_idx

    # Handle the final segment
    if start_idx < num_points:
        segment_S = S[start_idx:]
        if not has_decreasing_segment(segment_S, min_decreasing_steps):
            full_state_segment = data.iloc[start_idx:, :]
            u_segment = u.iloc[start_idx:, :]
            concataneted_segment = pd.concat([full_state_segment, u_segment], axis=1)
            trajectory_struct.append(concataneted_segment)

    # Return a list of separate trajectory segments, each containing its own DataFrame
    return trajectory_struct


def has_decreasing_segment(segment_S, min_decreasing_steps):

    decreasing_count = 0

    for i in range(1, len(segment_S)):
        if segment_S[i] < segment_S[i - 1]:
            decreasing_count += 1
            if decreasing_count >= min_decreasing_steps:
                return True
        else:
            decreasing_count = 0

    return False
