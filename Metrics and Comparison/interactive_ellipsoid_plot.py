import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import sys
import yaml
from matplotlib.widgets import Button, Slider
from matplotlib.lines import Line2D
import re
from matplotlib.transforms import Bbox
import os

sys.path.append('Utils')

from plot_track import plot_track
from check_collision import check_collision

check_coll = True

# Load the bucket mean and std data (assuming the bucket_data_mean_std is already loaded as a dictionary)
bucket_data_mean_std = pd.read_feather('Obtained Model Data/bucket_data_mean_std.feather')
bucket_data_mean_std = bucket_data_mean_std.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
bucket_data_mean_std = bucket_data_mean_std.to_dict()


expert_data = pd.read_feather('Obtained Model Data/pure_pursuit_artificial_df.feather')
expert_data = expert_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
expert_data = expert_data.to_dict()


all_s_expert = []
all_e_expert = []
all_dtheta_expert = []
all_omega_expert = []
all_x_expert = []
all_y_expert = []
all_steering_expert = []
all_throttle_expert = []

for trajectory in expert_data.values():
    all_s_expert.extend(trajectory['s'])
    all_e_expert.extend(trajectory['e'])
    all_dtheta_expert.extend(trajectory['dtheta'])
    all_x_expert.extend(trajectory['x'])
    all_y_expert.extend(trajectory['y'])
    all_omega_expert.extend(trajectory["omega"])
    all_steering_expert.extend(trajectory["steering"])
    all_throttle_expert.extend(trajectory["throttle"])

all_s_expert = np.array(all_s_expert)
all_e_expert = np.array(all_e_expert)
all_dtheta_expert = np.array(all_dtheta_expert)
all_x_expert = np.array(all_x_expert)
all_y_expert = np.array(all_y_expert)
all_omega_expert = np.array(all_omega_expert)
all_steering_expert = np.array(all_steering_expert)
all_throttle_expert = np.array(all_throttle_expert)

added_data = pd.read_feather('Obtained Model Data/added_data_low_noise_Kdd05.feather')
added_data = added_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
added_data = added_data.to_dict()

all_s_added = []
all_e_added = []
all_dtheta_added = []
all_omega_added = []
all_x_added = []
all_y_added = []
all_steering_added = []
all_throttle_added = []

for trajectory in added_data.values():
    all_s_added.extend(trajectory['s'])
    all_e_added.extend(trajectory['e'])
    all_dtheta_added.extend(trajectory['dtheta'])
    all_x_added.extend(trajectory['x'])
    all_y_added.extend(trajectory['y'])
    all_omega_added.extend(trajectory["omega"])
    all_steering_added.extend(trajectory["steering"])
    all_throttle_added.extend(trajectory["throttle"])

all_s_added = np.array(all_s_added)
all_e_added = np.array(all_e_added)
all_dtheta_added = np.array(all_dtheta_added)
all_x_added = np.array(all_x_added)
all_y_added = np.array(all_y_added)
all_omega_added = np.array(all_omega_added)
all_steering_added = np.array(all_steering_added)
all_throttle_added = np.array(all_throttle_added)

s_values = list(bucket_data_mean_std['s'].values())
bucket_data_mean_std["s"] = s_values

# Load the model data

file_path = 'Obtained Model Data/model40_dist_wrapped.feather'

# Regular expression to find the number in the filename
model_number = int(re.search(r'model(\d+)', file_path).group(1))

model_data = pd.read_feather(file_path)
model_data = model_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
model_data = model_data.to_dict()

all_s = []
all_e = []
all_dtheta = []
all_omega = []
all_x = []
all_y = []
all_steering = []
all_throttle = []

for trajectory in model_data.values():
    all_s.extend(trajectory['s'])
    all_e.extend(trajectory['e'])
    all_dtheta.extend(trajectory['dtheta'])
    all_x.extend(trajectory['x'])
    all_y.extend(trajectory['y'])
    all_omega.extend(trajectory["omega"])
    all_steering.extend(trajectory["steering"])
    all_throttle.extend(trajectory["throttle"])

all_s = np.array(all_s)
all_e = np.array(all_e)
all_dtheta = np.array(all_dtheta)
all_x = np.array(all_x)
all_y = np.array(all_y)
all_omega = np.array(all_omega)
all_steering = np.array(all_steering)
all_throttle = np.array(all_throttle)

with open("la_track.yaml", "r") as file:
    track_shape_data = yaml.safe_load(file)

model_data = {"s": all_s, "e": all_e, "dtheta": all_dtheta, "omega": all_omega, "x": all_x, "y": all_y, "steering": all_steering, "throttle": all_throttle}

fig, axs = plt.subplots(2, 2, figsize=(20, 14))  # Adjust layout to 2x2 plots

# Initialize model_index globally
model_index = 0  # Set to the first model index initially

# Initialize the line objects for both plots (empty at the start)
line, = axs[0, 0].plot([], [], 'b-', label="Model trajectory line")
point, = axs[0, 0].plot([], [], 'bo', label="Model data point")

line1, = axs[0, 1].plot([], [], 'b-', label="Model trajectory line")
point1, = axs[0, 1].plot([], [], 'bo', label="Model data point")

line2, = axs[1, 1].plot([], [], 'b-', label="Model trajectory line")
point2, = axs[1, 1].plot([], [], 'bo', label="Model data point")

line_side, = axs[1, 0].plot([], [], 'g-', label="Model x/y trajectory")
point_side, = axs[1, 0].plot([], [], 'ro', label="Model x/y position")

axs[0, 0].set_xlabel(r'Mean $e_y$')  # LaTeX for e_y
axs[0, 0].set_ylabel(r'Mean $e_\Psi$')  # LaTeX for e_Psi

# Update labels for subplot at axs[0, 1]
axs[0, 1].set_xlabel(r'Mean $e_y$')  # LaTeX for e_y
axs[0, 1].set_ylabel(r'Mean $\delta$')  # LaTeX for delta

# Update labels for subplot at axs[1, 1]
axs[1, 1].set_xlabel(r'Mean $s$')  # LaTeX for s
axs[1, 1].set_ylabel(r'Mean throttle')

# Set the axis labels for the side plot (XY plot)
axs[1, 0].set_xlabel("X Position")
axs[1, 0].set_ylabel("Y Position")
axs[1, 0].set_title("X-Y Trajectory of the Model")


for traj in expert_data:
    if traj == "trajectory_1":
        axs[1, 0].plot(expert_data[traj]['x'], expert_data[traj]['y'], linewidth = 0.5, alpha = 0.1, color="r", label="Expert Trajectories")
    else:
        axs[1, 0].plot(expert_data[traj]['x'], expert_data[traj]['y'], linewidth = 0.5, alpha = 0.1, color="r")


left_track_x, left_track_y, right_track_x, right_track_y = plot_track(fig, axs[1, 0], track_shape_data)

# Set up the ellipses plots
for ax_row in axs[0, :]:
    ax_row.set_xlim(-2, 2)  # Set x-axis limit between -2 and 2 for ellipses
    ax_row.set_ylim(-6, 6)  # Set y-axis limit between -4 and 4 for ellipses

axs[1,1].set_xlim(-15,15)
axs[1,1].set_ylim(-1, 1)


if check_coll:
    inner_collisions, outer_collisions = check_collision(all_x, all_y, left_track_x, left_track_y, right_track_x, right_track_y)

    print(f'Number of collisions with inner boundary: {inner_collisions}')
    print(f'Number of collisions with outer boundary: {outer_collisions}')
else:
    inner_collisions, outer_collisions = 13, 0



def is_outside_ellipsoid_n_dim(noisy_vals, means, cov_matrix):

    noisy_vals = np.array(noisy_vals)
    means = np.array(means)
    
    diff = noisy_vals - means
    
    inv_cov = np.linalg.inv(cov_matrix)
    
    dist_n_dim = diff.T @ inv_cov @ diff
    
    return dist_n_dim > 16


def distance_to_rotated_ellipsoid(point, ellipsoid_center, axes, angle):
    # Same as your original function for calculating distance to the rotated ellipsoid
    x, y = point
    x0, y0 = ellipsoid_center
    a, b = axes
    x_translated = x - x0
    y_translated = y - y0
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    x_rotated = cos_theta * x_translated + sin_theta * y_translated
    y_rotated = -sin_theta * x_translated + cos_theta * y_translated
    x_scaled = x_rotated / a
    y_scaled = y_rotated / b
    distance_to_unit_circle = np.sqrt(x_scaled**2 + y_scaled**2)
    distance_to_surface = np.abs(distance_to_unit_circle - 1)
    distance = distance_to_surface * np.sqrt(a**2 + b**2)
    return distance

def wrap_to_pi(angles):
    wrapped_angles = (angles + np.pi) % (2 * np.pi) - np.pi
    wrapped_angles = np.where(wrapped_angles >= np.pi, wrapped_angles - 2 * np.pi, wrapped_angles)
    wrapped_angles = np.where(wrapped_angles < -np.pi, wrapped_angles + 2 * np.pi, wrapped_angles)
    return wrapped_angles


added_point_plot_handles_added = {
    'edtheta_added': None,
    'sthrottle_added': None,
    'omegasteering_added': None
}

added_point_plot_handles_expert = {
    'edtheta_expert': None,
    'sthrottle_expert': None,
    'omegasteering_expert': None
}


# Function to plot the confidence ellipsoids and model data point for the closest s
def plot_ellipsoid_and_model(bucket_index, model_index):
    # Remove previous plot elements (ellipses and points)
    for ax in axs[0, :]:
        for patch in ax.patches:
            patch.remove()  # Remove ellipses from all subplots

    for patch in axs[1,1].patches:
        patch.remove()

    if added_point_plot_handles_added['edtheta_added']:
        added_point_plot_handles_added['edtheta_added'].remove()
    if added_point_plot_handles_added['sthrottle_added']:
        added_point_plot_handles_added['sthrottle_added'].remove()
    if added_point_plot_handles_added['omegasteering_added']:
        added_point_plot_handles_added['omegasteering_added'].remove()

    if added_point_plot_handles_expert['edtheta_expert']:
        added_point_plot_handles_expert['edtheta_expert'].remove()
    if added_point_plot_handles_expert['sthrottle_expert']:
        added_point_plot_handles_expert['sthrottle_expert'].remove()
    if added_point_plot_handles_expert['omegasteering_expert']:
        added_point_plot_handles_expert['omegasteering_expert'].remove()

    # Extract relevant values for the selected bucket (for the closest s)
    mean_e = bucket_data_mean_std['mean_e'][bucket_index]
    mean_dtheta = bucket_data_mean_std['mean_dtheta'][bucket_index]
    mean_omega = bucket_data_mean_std['mean_omega'][bucket_index]
    mean_s = bucket_data_mean_std['mean_s'][bucket_index]
    mean_throttle = bucket_data_mean_std['mean_throttle'][bucket_index]
    mean_steering = bucket_data_mean_std['mean_steering'][bucket_index]

    std_e = bucket_data_mean_std['std_e'][bucket_index]
    std_dtheta = bucket_data_mean_std['std_dtheta'][bucket_index]
    std_omega = bucket_data_mean_std['std_omega'][bucket_index]
    std_s = bucket_data_mean_std['std_s'][bucket_index]
    std_throttle = bucket_data_mean_std['std_throttle'][bucket_index]
    std_steering = bucket_data_mean_std['std_steering'][bucket_index]

    
    # Covariance matrices for the new pairings
    cov_matrix_omega_steering = np.array(bucket_data_mean_std['cov_omega_steering'][bucket_index]).reshape(2, 2)
    cov_matrix_s_throttle = np.array(bucket_data_mean_std['cov_s_throttle'][bucket_index]).reshape(2, 2)

    # Ellipse (e-dtheta) angle and dimensions
    _, eigvecs_edtheta = np.linalg.eig(np.array(bucket_data_mean_std['cov_e_dtheta'][bucket_index]).reshape(2, 2))
    angle_edtheta = np.arctan2(eigvecs_edtheta[1, 0], eigvecs_edtheta[0, 0]) * 180 / np.pi
    width_edtheta = 4 * std_e
    height_edtheta = 4 * std_dtheta
    
    # Plot the e-dtheta ellipse on the first axis
    ellipse_edtheta = Ellipse(xy=(mean_e, mean_dtheta), width=width_edtheta, height=height_edtheta, 
                              angle=angle_edtheta, edgecolor='r', facecolor='none', linewidth=3)
    axs[0, 0].add_patch(ellipse_edtheta)

    # Plot the model data point on top of the e-dtheta ellipsoid
    model_e = model_data['e'][model_index]
    model_dtheta = wrap_to_pi(model_data['dtheta'][model_index])
    point.set_data(model_e, model_dtheta)

    if is_outside_ellipsoid_n_dim([model_e, model_dtheta], [mean_e, mean_dtheta], np.array(bucket_data_mean_std['cov_e_dtheta'][bucket_index]).reshape(2, 2)):
        point.set_marker('X') 
    else:
        point.set_marker('o')  
    point.set_data(model_e, model_dtheta)
    
    # Ellipse (omega-steering) angle and dimensions
    _, eigvecs_omega_steering = np.linalg.eig(cov_matrix_omega_steering)
    angle_omega_steering = np.arctan2(eigvecs_omega_steering[1, 0], eigvecs_omega_steering[0, 0]) * 180 / np.pi
    width_omega_steering = 4 * std_e
    height_omega_steering = 4 * std_steering
    
    # Plot the omega-steering ellipse
    ellipse_omega_steering = Ellipse(xy=(mean_e, mean_steering), width=width_omega_steering, height=height_omega_steering, 
                                      angle=angle_omega_steering, edgecolor='g', facecolor='none', linewidth=3)
    axs[0, 1].add_patch(ellipse_omega_steering)

    model_omega = model_data['e'][model_index]
    model_steering = model_data['steering'][model_index]
    point1.set_data(model_omega, model_steering)  # Update point for omega-steering plot

    # Ellipse (s-throttle) angle and dimensions
    _, eigvecs_s_throttle = np.linalg.eig(cov_matrix_s_throttle)
    angle_s_throttle = np.arctan2(eigvecs_s_throttle[1, 0], eigvecs_s_throttle[0, 0]) * 180 / np.pi
    width_s_throttle = 4 * std_s
    height_s_throttle = 4 * std_throttle
    
    # Plot the s-throttle ellipse
    ellipse_s_throttle = Ellipse(xy=(mean_s, mean_throttle), width=width_s_throttle, height=height_s_throttle, 
                                 angle=angle_s_throttle, edgecolor='b', facecolor='none', linewidth=3)
    axs[1, 1].add_patch(ellipse_s_throttle)

    model_throttle = model_data['throttle'][model_index]
    model_s= model_data['s'][model_index]
    point2.set_data(model_s, model_throttle)  # Update point for s-throttle plot

    # Bucket range calculation
    bucket_start = bucket_data_mean_std['s'][bucket_index]
    try:
        bucket_end = bucket_data_mean_std['s'][bucket_index + 1]
    except:
        bucket_start = bucket_data_mean_std['s'][0]
        bucket_end = bucket_data_mean_std['s'][1]

    indices_in_bucket_added = (all_s_added >= bucket_start) & (all_s_added < bucket_end)
    indices_in_bucket_expert = (all_s_expert >= bucket_start) & (all_s_expert < bucket_end)
    
    # Get the corresponding e, dtheta, omega values for added and expert data
    added_e = np.array(all_e_added)[indices_in_bucket_added]
    added_dtheta = np.array(all_dtheta_added)[indices_in_bucket_added]
    added_omega = np.array(all_omega_added)[indices_in_bucket_added]
    added_steering = np.array(all_steering_added)[indices_in_bucket_added]
    added_s = np.array(all_s_added)[indices_in_bucket_added]
    added_throttle = np.array(all_throttle_added)[indices_in_bucket_added]


    expert_e = np.array(all_e_expert)[indices_in_bucket_expert]
    expert_dtheta = np.array(all_dtheta_expert)[indices_in_bucket_expert]
    expert_omega = np.array(all_omega_expert)[indices_in_bucket_expert]
    expert_steering = np.array(all_steering_expert)[indices_in_bucket_expert]
    expert_s = np.array(all_s_expert)[indices_in_bucket_expert]
    expert_throttle = np.array(all_throttle_expert)[indices_in_bucket_expert]



    # Plot added points (omega-steering and s-throttle)
    added_point_plot_handles_added['edtheta_added'] = axs[0, 0].plot(added_e, added_dtheta, 'yX', alpha=0.05)[0]
    added_point_plot_handles_added['omegasteering_added'] = axs[0, 1].plot(added_e, added_steering, 'yX', alpha=0.05)[0]  
    # added_point_plot_handles_added['sthrottle_added'] = axs[1, 1].plot(added_s, added_throttle, 'yX', alpha=0.1)[0]

    # Plot expert points (omega-steering and s-throttle)
    added_point_plot_handles_expert['edtheta_expert'] = axs[0, 0].plot(expert_e, expert_dtheta, 'mX', alpha=0.01)[0]
    added_point_plot_handles_expert['omegasteering_expert'] = axs[0, 1].plot(expert_e, expert_steering, 'mX', alpha=0.01)[0] 
    # added_point_plot_handles_expert['sthrottle_expert'] = axs[1, 1].plot(expert_s, expert_throttle, 'mX', alpha=0.02)[0] 

    legend_handles = [
        Line2D([0], [0], marker='X', color='y', markerfacecolor='y', markersize=8, alpha=1.0, label='Added Points'),
        Line2D([0], [0], marker='X', color='m', markerfacecolor='m', markersize=8, alpha=1.0, label='Expert Points')
    ]

    # Add the legend with the custom handles
    axs[0, 1].legend(handles=legend_handles, labels=["Added Points", "Expert Points"])
    axs[0, 0].legend(handles=legend_handles, labels=["Added Points", "Expert Points"])
    axs[1, 1].legend(handles=legend_handles, labels=["Added Points", "Expert Points"])

    axs[1,1].set_xlim(bucket_end-0.5,bucket_start+0.5)
    axs[1,1].set_ylim(-1, 1)

    # Update the side plot with the x and y positions
    line_side.set_data(all_x[:model_index + 1], all_y[:model_index + 1])
    line_side.set_alpha(0.8)  # Set transparency (0.0 is fully transparent, 1.0 is fully opaque)
    line_side.set_linewidth(0.3)
    point_side.set_data(all_x[model_index], all_y[model_index])


    axs[1, 0].text(0.3, 0.95, f'Collisions: {inner_collisions + outer_collisions}', transform=axs[1, 0].transAxes, ha='right', va='top', fontsize=12, color='red')

    return line, point, line1, point1, line2, point2, line_side, point_side  # Return the updated objects



# Function to find the closest s from model data to a given s in bucket data
def find_closest_bucket_s(model_s):
    diffs = np.abs(np.array(bucket_data_mean_std['s']) - model_s)
    closest_index = np.argmin(diffs)
    return closest_index


def stop_animation(event):
    ani.event_source.stop()

ax_slider = plt.axes([0.1, 0.005, 0.6, 0.075]) 
slider = Slider(ax_slider, 'Frame', 0, len(all_s)-1, valinit=0, valstep=1)


# Declare glob_frame globally
glob_frame = 0

# Function to update the frame based on slider position
def update_slider(val):
    global glob_frame
    glob_frame = int(slider.val)  # Update the global frame based on slider position
    bucket_index = find_closest_bucket_s(model_data['s'][glob_frame])
    plot_ellipsoid_and_model(bucket_index, glob_frame)


    if glob_frame == len(all_s) - 1:
        save_frame(axs)

    return line, point, line1, point1, line2, point2, line_side, point_side

# Function to start the animation from the selected glob_frame
def play_animation(event):
    global glob_frame
    # Set the frame sequence to start from glob_frame
    ani.event_source.frame_seq = range(glob_frame, len(all_s))  # Start animation from glob_frame
    ani.event_source.start()

# Update the plots based on slider changes
slider.on_changed(update_slider)

# Now, connect the slider to the animation update
def update(frame):
    global glob_frame
    glob_frame = frame
    bucket_index = find_closest_bucket_s(model_data['s'][glob_frame])
    plot_ellipsoid_and_model(bucket_index, glob_frame)
    return line, point, line1, point1, line2, point2, line_side, point_side

def full_extent(ax, pad=0.1):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

def save_frame(axs):
    # Create the directory if it doesn't exist
    save_path = f'figures/model{model_number}'
    os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist

    # Save the full figure
    print("Saving full figure at the last step...")
    plt.savefig(f'{save_path}/ellipsoids_and_overlayed_trajectories_{model_number}.svg', format='svg')
    print("Full figure saved as SVG!")

    # Save each subplot separately using tight bounding box
    extent = axs[1, 0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    axs[1, 0].figure.savefig(f'{save_path}/X-Y_overlayed_model_{model_number}.svg', bbox_inches=extent)
    axs[1, 0].figure.savefig(f'{save_path}/X-Y_overlayed_model_{model_number}.png', bbox_inches=extent)

    extent = axs[0, 0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    axs[0, 0].figure.savefig(f'{save_path}/e_dtheta_model_{model_number}.svg', bbox_inches=extent)
    axs[0, 0].figure.savefig(f'{save_path}/e_dtheta_model_{model_number}.png', bbox_inches=extent)

    extent = axs[0, 1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    axs[0, 1].figure.savefig(f'{save_path}/e_steering_model_{model_number}.svg', bbox_inches=extent)
    axs[0, 1].figure.savefig(f'{save_path}/e_steering_model_{model_number}.png', bbox_inches=extent)

    extent = axs[1, 1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    axs[1, 1].figure.savefig(f'{save_path}/s_throttle_model_{model_number}.svg', bbox_inches=extent)
    axs[1, 1].figure.savefig(f'{save_path}/s_throttle_model_{model_number}.png', bbox_inches=extent)

    print("Frames saved successfully!")

# Create the animation
ani = FuncAnimation(fig, update, frames=len(all_s), interval=0.1, repeat=False)


# Play button position
ax_button_play = plt.axes([0.85, 0.00, 0.1, 0.075])  # Lowered position
button_start = Button(ax_button_play, 'Play', color='lightgoldenrodyellow', hovercolor='orange')
button_start.on_clicked(play_animation)

# Stop button position
ax_button_stop = plt.axes([0.75, 0.00, 0.1, 0.075])  # Lowered position
button_stop = Button(ax_button_stop, 'Stop', color='lightgoldenrodyellow', hovercolor='orange')
button_stop.on_clicked(stop_animation)


plt.show()
