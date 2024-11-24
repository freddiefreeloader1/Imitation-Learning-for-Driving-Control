import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import sys
import yaml
from matplotlib.widgets import Button

sys.path.append('Utils')

from plot_track import plot_track

# Load the bucket mean and std data (assuming the bucket_data_mean_std is already loaded as a dictionary)
bucket_data_mean_std = pd.read_feather('Obtained Model Data/bucket_data_mean_std.feather')
bucket_data_mean_std = bucket_data_mean_std.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
bucket_data_mean_std = bucket_data_mean_std.to_dict()

s_values = list(bucket_data_mean_std['s'].values())
bucket_data_mean_std["s"] = s_values

# Load the model data
model_data = pd.read_feather('Obtained Model Data/model22_dist_wrapped.feather')
model_data = model_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
model_data = model_data.to_dict()

all_s = []
all_e = []
all_dtheta = []
all_x = []  # Assuming you have x position data in model_data
all_y = []  # Assuming you have y position data in model_data

for trajectory in model_data.values():
    all_s.extend(trajectory['s'])
    all_e.extend(trajectory['e'])
    all_dtheta.extend(trajectory['dtheta'])
    all_x.extend(trajectory['x'])  # Add x position data
    all_y.extend(trajectory['y'])  # Add y position data

# Convert lists into arrays for easier handling
all_s = np.array(all_s)
all_e = np.array(all_e)
all_dtheta = np.array(all_dtheta)
all_x = np.array(all_x)
all_y = np.array(all_y)


model_data = {"s": all_s, "e": all_e, "dtheta": all_dtheta}

fig, (ax, ax_side) = plt.subplots(1, 2, figsize=(16, 6))

# Initialize model_index globally
model_index = 0  # Set to the first model index initially

# Initialize the line objects for both plots (empty at the start)
line, = ax.plot([], [], 'b-', label="Model trajectory line")
point, = ax.plot([], [], 'bo', label="Model data point")

line_side, = ax_side.plot([], [], 'g-', label="Model x/y trajectory")
point_side, = ax_side.plot([], [], 'go', label="Model x/y position")

with open("la_track.yaml", "r") as file:
    track_shape_data = yaml.safe_load(file)

# Set up the side plot
ax_side.set_xlim(np.min(all_x) - 0.1, np.max(all_x) + 0.1)  # Adjust x limits
ax_side.set_ylim(np.min(all_y) - 0.1, np.max(all_y) + 0.1)  # Adjust y limits
ax_side.set_xlabel("X Position")
ax_side.set_ylabel("Y Position")
ax_side.set_title("X-Y Trajectory of the Model")

plot_track(fig, ax_side, track_shape_data)

def distance_to_rotated_ellipsoid(point, ellipsoid_center, axes, angle):
    
    # Unpack input values
    x, y = point
    x0, y0 = ellipsoid_center
    a, b = axes
    
    # Step 1: Translate the point to the ellipsoid-centered coordinates
    x_translated = x - x0
    y_translated = y - y0
    
    # Step 2: Create the rotation matrix for the given angle
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    
    # Apply the inverse of the rotation to the translated point
    x_rotated = cos_theta * x_translated + sin_theta * y_translated
    y_rotated = -sin_theta * x_translated + cos_theta * y_translated
    
    # Step 3: Normalize the point with respect to the ellipsoid's axes (scaled by a and b)
    x_scaled = x_rotated / a
    y_scaled = y_rotated / b
    
    # Step 4: Compute the Euclidean distance from the origin (unit circle)
    distance_to_unit_circle = np.sqrt(x_scaled**2 + y_scaled**2)
    
    # Step 5: Adjust the distance based on the unit circle
    distance_to_surface = np.abs(distance_to_unit_circle - 1)
    
    # Step 6: Scale the result back to the ellipsoid's surface
    distance = distance_to_surface * np.sqrt(a**2 + b**2)
    
    return distance

def wrap_to_pi(angles):
    wrapped_angles = (angles + np.pi) % (2 * np.pi) - np.pi
    
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
    _, eigvecs = np.linalg.eig(cov_matrix)
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

    distance_to_ellipsoid = distance_to_rotated_ellipsoid([model_e, model_dtheta], [mean_e, mean_dtheta], [width, height], angle= angle)
    
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

def play_animation(event):
    ani.event_source.start()

def stop_animation(event):
    ani.event_source.stop()

def update(frame):
    model_index = frame
    model_s = model_data['s'][model_index]
    closest_bucket_index = find_closest_bucket_s(model_s)
    
    # Update the ellipsoid and model data point for this frame
    plot_ellipsoid_and_model(closest_bucket_index, model_index)
    
    # Update the trajectory line with the new model point
    line.set_data(model_data['e'][:model_index+1], wrap_to_pi(model_data['dtheta'][:model_index+1]))
    
    # Update the side plot with the x and y positions
    line_side.set_data(all_x[:model_index+1], all_y[:model_index+1])
    point_side.set_data(all_x[model_index], all_y[model_index])
    
    return line, point, line_side, point_side  # Return the updated objects


# Create the animation
ani = FuncAnimation(fig, update, frames=len(model_data['s']), interval=0.1, repeat=False)

ax_button = plt.axes([0.85, 0.01, 0.1, 0.075])  # Button position
button_start = Button(ax_button, 'Play', color='lightgoldenrodyellow', hovercolor='orange')
button_start.on_clicked(play_animation)

ax_button = plt.axes([0.75, 0.01, 0.1, 0.075])  # Button position
button_stop = Button(ax_button, 'Stop', color='lightgoldenrodyellow', hovercolor='orange')
button_stop.on_clicked(stop_animation)

# Show the plot
plt.show()
