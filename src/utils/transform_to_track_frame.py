import numpy as np

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def transform_to_track(state, track_data):
    track_x = track_data[0]
    track_y = track_data[1]
    track_angle = track_data[2]

    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    theta = state[4] 
    omega = state[5]  

    pos = np.array([x, y])
    track_pos = np.array([track_x, track_y]) 

    T_local_global = np.array([[np.cos(track_angle), np.sin(track_angle)],
                               [-np.sin(track_angle), np.cos(track_angle)]])
    
    pos_local = (T_local_global @ (pos - track_pos).T).T
    turn_angle_local = wrap_to_pi(theta - track_angle)

   
    vel = np.array([vx, vy])
    vel_local = (T_local_global @ vel.T).T

    T_body_world = np.array([[np.cos(turn_angle_local), np.sin(turn_angle_local)],
                             [-np.sin(turn_angle_local), np.cos(turn_angle_local)]])
    
    v_body = np.dot(T_body_world, vel_local)

    omega_local = omega  

    return np.concatenate((pos_local, v_body, [turn_angle_local, omega_local]))

# Example usage

state = np.array([1, 2, 3, 4, 5, 6])
track_data = np.array([0, 0, 0])

transformed_state = transform_to_track(state, track_data)
print(transformed_state)
