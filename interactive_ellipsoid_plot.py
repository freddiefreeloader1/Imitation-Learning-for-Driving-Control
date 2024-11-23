import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse

# Load the bucket mean and std data (assuming the bucket_data_mean_std is already loaded as a dictionary)
bucket_data_mean_std = pd.read_feather('bucket_data_mean_std.feather')
bucket_data_mean_std = bucket_data_mean_std.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
bucket_data_mean_std = bucket_data_mean_std.to_dict()

s_values = list(bucket_data_mean_std['s'].values())
bucket_data_mean_std["s"] = s_values


# Load the model data
model_data = pd.read_feather('model18_dist.feather')
model_data = model_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
model_data = model_data.to_dict()

all_s = []
all_e = []
all_dtheta = []

for trajectory in model_data.values():
    all_s.extend(trajectory['s'])
    all_e.extend(trajectory['e'])
    all_dtheta.extend(trajectory['dtheta'])

# Convert lists into arrays for easier handling
all_s = np.array(all_s)
all_e = np.array(all_e)
all_dtheta = np.array(all_dtheta)


model_data = {"s": all_s, "e": all_e, "dtheta": all_dtheta}

# Set up figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

# Initialize model_index globally
model_index = 0  # Set to the first model index initially

# Initialize the line object (empty at the start)
line, = ax.plot([], [], 'b-', label="Model trajectory line")
point, = ax.plot([], [], 'bo', label="Model data point")


def wrap_to_pi(angles):
    # Apply the wrapping logic element-wise for the entire array
    wrapped_angles = (angles + np.pi) % (2 * np.pi) - np.pi
    
    # Ensure the angle is properly within the range [-pi, pi)
    wrapped_angles = np.where(wrapped_angles >= np.pi, wrapped_angles - 2 * np.pi, wrapped_angles)
    wrapped_angles = np.where(wrapped_angles < -np.pi, wrapped_angles + 2 * np.pi, wrapped_angles)
    
    return wrapped_angles

# Function to plot the confidence ellipsoid and model data point for the closest s
def plot_ellipsoid_and_model(bucket_index, model_index):
    # Remove previous plot elements (ellipses and points)
    for patch in ax.patches:
        patch.remove()  # Remove ellipses
    
    # Extract relevant values for the selected bucket (for the closest s)
    mean_e = bucket_data_mean_std['mean_e'][bucket_index]
    mean_dtheta = bucket_data_mean_std['mean_dtheta'][bucket_index]
    std_e = bucket_data_mean_std['std_e'][bucket_index]
    std_dtheta = bucket_data_mean_std['std_dtheta'][bucket_index]

    cov_matrix = np.array(bucket_data_mean_std['cov_e_dtheta'][bucket_index]).reshape(2, 2)
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0]) * 180 / np.pi

    # Ellipse parameters (4 standard deviations for 99.99% confidence)
    width = 4 * std_e
    height = 4 * std_dtheta
    
    # Plot the ellipse at the mean location
    ellipse = Ellipse(xy=(mean_e, mean_dtheta), width=width, height=height, angle = angle, edgecolor='r', facecolor='none', linewidth=2)
    ax.add_patch(ellipse)
    
    # Plot the model data point on top of the ellipsoid
    model_e = model_data['e'][model_index]
    model_dtheta = wrap_to_pi(model_data['dtheta'][model_index])
    point.set_data(model_e, model_dtheta)
    
    # Set labels and title
    ax.set_title(f"Confidence Ellipsoid and Model Data for Bucket s={bucket_data_mean_std['s'][bucket_index]}")
    ax.set_xlabel("Mean e")
    ax.set_ylabel("Mean dtheta")
    
    # Fix the axis limits manually to be within a small range
    ax.set_xlim(-2, 2)  # Set x-axis limit between -1 and 1
    ax.set_ylim(-4, 4)  # Set y-axis limit between -1 and 1
    
# Function to find the closest s from model data to a given s in bucket data
def find_closest_bucket_s(model_s):
    # Calculate the absolute differences between model_s and all s values in bucket_data_mean_std
    diffs = np.abs(np.array(bucket_data_mean_std['s']) - model_s)
    # Find the index of the minimum difference (i.e., the closest s)
    closest_index = np.argmin(diffs)
    return closest_index

# Function to update the plot for the animation
def update(frame):
    model_index = frame
    model_s = model_data['s'][model_index]
    closest_bucket_index = find_closest_bucket_s(model_s)
    
    # Update the ellipsoid and model data point for this frame
    plot_ellipsoid_and_model(closest_bucket_index, model_index)
    
    # Update the trajectory line with the new model point
    line.set_data(model_data['e'][:model_index+1], wrap_to_pi(model_data['dtheta'][:model_index+1]))
    
    return line, point  # Return the updated objects

# Create the animation
ani = FuncAnimation(fig, update, frames=len(model_data['s']), interval=1, repeat=False)

# Show the plot
plt.show()
