import numpy as np

def is_point_in_arbitrary_shape(point, boundary_points):
    """
    Determine if a point is inside an arbitrary shape using the Ray Casting algorithm with sampled boundary points.
    
    Parameters:
    - point: (px, py) coordinates of the point to check.
    - boundary_points: List of (x, y) coordinates representing the sampled boundary points of the shape.
    
    Returns:
    - True if the point is inside the shape, False otherwise.
    """
    px, py = point
    n = len(boundary_points)
    inside = False

    # Loop through each edge of the shape (using boundary points)
    for i in range(n):
        x1, y1 = boundary_points[i]
        x2, y2 = boundary_points[(i + 1) % n]  # Wrap around to the first vertex

        # Check if the point is between the y-coordinates of the edge
        if min(y1, y2) < py <= max(y1, y2):
            # Calculate the x-coordinate of the intersection of the ray and the edge
            x_intersection = (py - y1) * (x2 - x1) / (y2 - y1) + x1
            
            if x_intersection > px:
                inside = not inside  # Toggle inside status

    return inside


def check_collision(trajectory_x, trajectory_y, left_track_x, left_track_y, right_track_x, right_track_y, collision_threshold=0.1):
    """
    Check for transitions (collisions) between the trajectory and the track boundaries.
    A transition happens when the trajectory crosses into or out of the inner or outer track boundaries.
    
    Parameters:
    - trajectory_x: x coordinates of the trajectory
    - trajectory_y: y coordinates of the trajectory
    - left_track_x, left_track_y: coordinates of the left (inner) track boundary
    - right_track_x, right_track_y: coordinates of the right (outer) track boundary
    - collision_threshold: minimum distance to track boundary to count as a collision (not used in transition detection)
    
    Returns:
    - num_inner_collisions: Number of transitions across the inner (left) boundary
    - num_outer_collisions: Number of transitions across the outer (right) boundary
    """
    num_inner_collisions = 0
    num_outer_collisions = 0

    # Initial states (assume the trajectory starts outside the boundaries)
    was_inside_inner = False
    was_outside_outer = False

    # Iterate over the trajectory points to detect transitions
    for tx, ty in zip(trajectory_x, trajectory_y):
        # Check if the point is inside the inner boundary (left track)
        is_inside_inner = is_point_in_arbitrary_shape((tx, ty), list(zip(left_track_x, left_track_y)))

        # Check if the point is outside the outer boundary (right track)
        is_outside_outer = not is_point_in_arbitrary_shape((tx, ty), list(zip(right_track_x, right_track_y)))

        # Check for transition across the inner boundary
        # if is_inside_inner and not was_inside_inner:
        #     num_inner_collisions += 1  # Transition from outside to inside the inner boundary
        if not is_inside_inner and was_inside_inner:
            num_inner_collisions += 1  # Transition from inside to outside the inner boundary

        # Check for transition across the outer boundary
        # if is_outside_outer and not was_outside_outer:
        #     num_outer_collisions += 1  # Transition from inside to outside the outer boundary
        if not is_outside_outer and was_outside_outer:
            num_outer_collisions += 1  # Transition from outside to inside the outer boundary

        # Update the previous states for the next point
        was_inside_inner = is_inside_inner
        was_outside_outer = is_outside_outer

    return num_inner_collisions, num_outer_collisions
