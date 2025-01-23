import numpy as np

def wrap_to_pi(angle):
    """
    Normalizes an angle (in radians) to the range [-pi, pi].
    
    Args:
        angle (float): Angle in radians to normalize.
    
    Returns:
        float: Normalized angle within the range [-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

def find_closest_point(path, pos, bring_to_origin = True):

    """
    Finds the closest point on a given path to a specified position.
    
    Args:
        path (dict): Dictionary containing the path details, including:
            - 'track': A sub-dictionary with:
                - 'xCoords': X-coordinates of the path.
                - 'yCoords': Y-coordinates of the path.
                - 'arcLength': Cumulative arc lengths along the path.
                - 'x_init': Initial X-coordinate of the path.
                - 'y_init': Initial Y-coordinate of the path.
        pos (array-like): 2D position [x, y] for which the closest point is sought.
        bring_to_origin (bool): If True, shifts the path coordinates to the origin based on the initial point.

    Returns:
        tuple: Closest arc length (`closest_s`) and the index of the closest point (`closest_idx`).
    """
    if bring_to_origin:
        path_pos = np.array([np.array(path['track']['xCoords']) - path['track']['x_init'] , np.array(path['track']['yCoords']) - path['track']['y_init']])
    else:
        path_pos = np.array([np.array(path['track']['xCoords']), np.array(path['track']['yCoords'])])

    closest_idx = 0
    closest_s = 0

    distances = np.linalg.norm(pos.reshape(-1, 1) - path_pos, axis=0)
    closest_idx = np.argmin(distances)
    closest_s = path['track']['arcLength'][int(closest_idx)]

    # get the middle point of the path['track']['arcLength'] 

    closest_s = closest_s % path['track']['arcLength'][int(len(path['track']['arcLength']) // 2)]

    return closest_s, closest_idx

def pose_to_curvi(path, pose, bring_to_origin = True):
    """
    Converts a Cartesian pose to curvilinear coordinates relative to a given path.
    
    Args:
        path (dict): Dictionary containing the path details (see `find_closest_point`).
        pose (array-like): 3D pose [x, y, theta] to convert, where theta is the orientation angle.
        bring_to_origin (bool): If True, shifts the path coordinates to the origin based on the initial point.
    
    Returns:
        list: Curvilinear coordinates [s, e, dtheta]:
            - s: Arc length along the path to the closest point.
            - e: Lateral deviation (perpendicular distance) to the path at the closest point.
            - dtheta: Difference in orientation angle between the pose and the path at the closest point.
    """
    if bring_to_origin:
        path_pose = np.array([np.array(path['track']['xCoords']) - path['track']['x_init'], np.array(path['track']['yCoords']) - path['track']['y_init'], 
                        path['track']['tangentAngle']])
    else:
        path_pose = np.array([np.array(path['track']['xCoords']), np.array(path['track']['yCoords']), 
                        path['track']['tangentAngle']])

    path_pose = path_pose[:, :path_pose.shape[1] // 2]

    pose_curvi = [0,0,0]
	
    closest_s, closest_idx = find_closest_point(path, pose[:2], bring_to_origin)

    pose_curvi[0] = closest_s

    
    path_angle = path_pose[2, int(closest_idx)]
    T_local_global = np.array([
                 [np.cos(path_angle), np.sin(path_angle)],
                 [-np.sin(path_angle), np.cos(path_angle)]
                  ])

    # ask about this part, why the only the first row of T_local_global is used
    e = T_local_global[1, :] @ (pose[:2] - path_pose[:2, int(closest_idx)])
    dtheta =  pose[2] - path_pose[2, int(closest_idx)]

    pose_curvi[1:3] = [e, dtheta]

    return pose_curvi
