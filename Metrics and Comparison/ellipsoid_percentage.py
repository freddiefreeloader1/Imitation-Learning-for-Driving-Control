import numpy as np
import pandas as pd

def is_outside_ellipsoid_n_dim(noisy_vals, means, cov_matrix):

    noisy_vals = np.array(noisy_vals)
    means = np.array(means)
    
    diff = noisy_vals - means
    inv_cov = np.linalg.inv(cov_matrix)
    dist_n_dim = diff.T @ inv_cov @ diff
    return dist_n_dim > 16  # Assuming the threshold is 16 (for a 2-sigma threshold)

def find_closest_bucket_s(model_s):
    diffs = np.abs(np.array(bucket_data_mean_std['s']) - model_s)
    closest_index = np.argmin(diffs)
    return closest_index

def wrap_to_pi(angles):
    """
    Wrap angles to be between -pi and pi.
    """
    wrapped_angles = (angles + np.pi) % (2 * np.pi) - np.pi
    wrapped_angles = np.where(wrapped_angles >= np.pi, wrapped_angles - 2 * np.pi, wrapped_angles)
    wrapped_angles = np.where(wrapped_angles < -np.pi, wrapped_angles + 2 * np.pi, wrapped_angles)
    return wrapped_angles

# Load the model data
model_data = pd.read_feather('Obtained Model Data/model37_dist_wrapped.feather')
model_data = model_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
model_data = model_data.to_dict()

# Load the bucket data (mean, std, cov for the ellipsoid)
bucket_data_mean_std = pd.read_feather('Obtained Model Data/bucket_data_mean_std.feather')
bucket_data_mean_std = bucket_data_mean_std.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
bucket_data_mean_std = bucket_data_mean_std.to_dict()

s_values = list(bucket_data_mean_std['s'].values())
bucket_data_mean_std["s"] = s_values

all_s = []
all_e = []
all_dtheta = []
all_omega = []
all_x = []  # Assuming you have x position data in model_data
all_y = []  # Assuming you have y position data in model_data

for trajectory in model_data.values():
    all_s.extend(trajectory['s'])
    all_e.extend(trajectory['e'])
    all_dtheta.extend(trajectory['dtheta'])
    all_x.extend(trajectory['x'])  # Add x position data
    all_y.extend(trajectory['y'])  # Add y position data
    all_omega.extend(trajectory["omega"])

# Convert lists into arrays for easier handling
all_s = np.array(all_s)
all_e = np.array(all_e)
all_dtheta = np.array(all_dtheta)
all_x = np.array(all_x)
all_y = np.array(all_y)
all_omega = np.array(all_omega)

# Initialize counters for points outside the ellipsoid
outside_count = 0

# Loop through each point in the model data and check if it's outside the ellipsoid
for i in range(len(all_s)):
    # For the given model point, find the closest bucket index
    model_e = all_e[i]
    model_dtheta = wrap_to_pi(all_dtheta[i])
    
    # Get the relevant values for the ellipsoid (mean and covariance) for the closest bucket
    bucket_index = find_closest_bucket_s(all_s[i])  # Function to find closest bucket index
    mean_e = bucket_data_mean_std['mean_e'][bucket_index]
    mean_dtheta = bucket_data_mean_std['mean_dtheta'][bucket_index]
    cov_matrix_edtheta = np.array(bucket_data_mean_std['cov_e_dtheta'][bucket_index]).reshape(2, 2)
    
    # Check if the point is outside the ellipsoid
    if is_outside_ellipsoid_n_dim([model_e, model_dtheta], [mean_e, mean_dtheta], cov_matrix_edtheta):
        outside_count += 1

# Calculate the percentage of points outside the ellipsoid
total_points = len(all_s)
percentage_outside = (outside_count / total_points) * 100

print(f"Percentage of model points outside the ellipsoid: {percentage_outside:.2f}%")
