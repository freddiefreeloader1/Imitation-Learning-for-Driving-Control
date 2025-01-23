import numpy as np


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def find_closest_point(path, pos, bring_to_origin = True):

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

def curvi_to_pose_offset(path, curvi_coords, bring_to_origin=True):
    s, e = curvi_coords
    
    if bring_to_origin:
        # Adjust path to the origin if needed
        path_pos = np.array([np.array(path['xCoords']) - path['x_init'], 
                             np.array(path['yCoords']) - path['y_init'], 
                             path['tangentAngle']])
    else:
        path_pos = np.array([np.array(path['xCoords']), 
                             np.array(path['yCoords']), 
                             path['tangentAngle']])

    closest_idx = np.argmin(np.abs(np.array(path['arcLength']) - s))
    x_s = path_pos[0, closest_idx]  
    y_s = path_pos[1, closest_idx]  
    theta_s = path_pos[2, closest_idx]  

    # Compute tangent vectors at consecutive points to estimate curvature
    prev_idx = max(closest_idx - 1, 0)  # Make sure we don't go out of bounds
    next_idx = min(closest_idx + 1, len(path['arcLength']) - 1)  # Ensure within bounds

    tangent_prev = np.array([path_pos[0, prev_idx] - x_s, path_pos[1, prev_idx] - y_s])
    tangent_next = np.array([path_pos[0, next_idx] - x_s, path_pos[1, next_idx] - y_s])

    # Compute the cross product to find the curvature direction
    cross_product = np.cross(tangent_prev, tangent_next)
    
    # If the cross product is zero, the path is straight
    if np.abs(cross_product) < 1e-6:  # Threshold for zero curvature
        # Apply the default outward normal (perpendicular to the tangent)
        normal_vector = np.array([-np.sin(theta_s), np.cos(theta_s)])  # 90-degree rotation of tangent
        curvature_sign = 1  # Default to outward direction (positive normal)
    else:
        # If there is curvature, the sign of the cross product gives us the direction
        curvature_sign = np.sign(cross_product)
        normal_vector = np.array([-np.sin(theta_s), np.cos(theta_s)])  # Normal to the tangent

    # Apply the lateral offset 'e' in the direction of the outward normal
    offset = normal_vector * curvature_sign * e
    
    x = x_s + offset[0]  # Apply offset in x direction
    y = y_s + offset[1]  # Apply offset in y direction


    return x, y

