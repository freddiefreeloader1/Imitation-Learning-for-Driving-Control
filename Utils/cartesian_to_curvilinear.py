import numpy as np


def find_closest_point(path, pos):
    path_pos = np.array([np.array(path['track']['xCoords']) - path['track']['x_init'] , np.array(path['track']['yCoords']) - path['track']['y_init']])
    
    closest_idx = np.full(pos.shape[0], np.nan)
    closest_s = np.full(pos.shape[0], np.nan)

    for k in range(pos.shape[0]):
        distances = np.linalg.norm(pos[k, :].reshape(-1, 1) - path_pos, axis=0)
        closest_idx[k] = np.argmin(distances)
        closest_s[k] = path['track']['arcLength'][int(closest_idx[k])]

    return closest_s, closest_idx

def pose_to_curvi(path, pose):

    path_pose = np.array([np.array(path['track']['xCoords']) - path['track']['x_init'], np.array(path['track']['yCoords']) - path['track']['y_init'], 
                        path['track']['tangentAngle']])

    path_pose = path_pose[:, :path_pose.shape[1] // 2]

    pose_curvi = np.zeros_like(pose)
    closest_s, closest_idx = find_closest_point(path, pose.iloc[:, :2].to_numpy())

    pose_curvi[:, 0] = closest_s

    for k in range(pose.shape[0]):
        path_angle = path_pose[2, int(closest_idx[k])]
        T_local_global = np.array([
            [np.cos(path_angle), np.sin(path_angle)],
            [-np.sin(path_angle), np.cos(path_angle)]
        ])

        # ask about this part, why the only the first row of T_local_global is used
        e = T_local_global[1, :] @ (pose.iloc[k, :2].to_numpy() - path_pose[:2, int(closest_idx[k])])
        dtheta =  pose.iloc[k, 2] - path_pose[2, int(closest_idx[k])]
        
        pose_curvi[k, 1:3] = [e, dtheta]

    return pose_curvi