import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import sys
import yaml
from matplotlib.widgets import Button, Slider
from matplotlib.lines import Line2D

sys.path.append('Utils')

from plot_track import plot_track

# Load the bucket mean and std data (assuming the bucket_data_mean_std is already loaded as a dictionary)
bucket_data_mean_std = pd.read_feather('Obtained Model Data/bucket_data_mean_std.feather')
bucket_data_mean_std = bucket_data_mean_std.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
bucket_data_mean_std = bucket_data_mean_std.to_dict()


expert_data = pd.read_feather('Obtained Model Data/all_trajectories_filtered.feather')
expert_data = expert_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
expert_data = expert_data.to_dict()


all_s_expert = []
all_e_expert  = []
all_dtheta_expert  = []
all_omega_expert  = []
all_x_expert  = []  # Assuming you have x position data in model_data
all_y_expert  = []  # Assuming you have y position data in model_data

for trajectory in expert_data.values():
    all_s_expert .extend(trajectory['s'])
    all_e_expert .extend(trajectory['e'])
    all_dtheta_expert .extend(trajectory['dtheta'])
    all_x_expert .extend(trajectory['x'])  # Add x position data
    all_y_expert .extend(trajectory['y'])  # Add y position data
    all_omega_expert .extend(trajectory["omega"])

# Convert lists into arrays for easier handling
all_s_expert  = np.array(all_s_expert)
all_e_expert  = np.array(all_e_expert)
all_dtheta_expert  = np.array(all_dtheta_expert)
all_x_expert  = np.array(all_x_expert)
all_y_expert  = np.array(all_y_expert)
all_omega_expert  = np.array(all_omega_expert)


added_data = pd.read_feather('Obtained Model Data/added_data_low_noise.feather')
added_data = added_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
added_data = added_data.to_dict()


all_s_added = []
all_e_added  = []
all_dtheta_added  = []
all_omega_added  = []
all_x_added  = []  # Assuming you have x position data in model_data
all_y_added  = []  # Assuming you have y position data in model_data

for trajectory in added_data.values():
    all_s_added .extend(trajectory['s'])
    all_e_added .extend(trajectory['e'])
    all_dtheta_added .extend(trajectory['dtheta'])
    all_x_added .extend(trajectory['x'])  # Add x position data
    all_y_added .extend(trajectory['y'])  # Add y position data
    all_omega_added .extend(trajectory["omega"])

# Convert lists into arrays for easier handling
all_s_added  = np.array(all_s_added )
all_e_added  = np.array(all_e_added )
all_dtheta_added  = np.array(all_dtheta_added )
all_x_added  = np.array(all_x_added )
all_y_added  = np.array(all_y_added )
all_omega_added  = np.array(all_omega_added)


s_values = list(bucket_data_mean_std['s'].values())
bucket_data_mean_std["s"] = s_values

# Load the model data
model_data = pd.read_feather('Obtained Model Data/model30_dist_wrapped.feather')
model_data = model_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
model_data = model_data.to_dict()

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

with open("la_track.yaml", "r") as file:
    track_shape_data = yaml.safe_load(file)

model_data = {"s": all_s, "e": all_e, "dtheta": all_dtheta, "omega": all_omega}

fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # Adjust layout to 2x2 plots

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

axs[0, 0].set_xlabel('Mean e')
axs[0, 0].set_ylabel('Mean dtheta')

axs[0, 1].set_xlabel('Mean e')
axs[0, 1].set_ylabel('Mean omega')

axs[1, 1].set_xlabel('Mean dtheta')
axs[1, 1].set_ylabel('Mean omega')

# Set the axis labels for the side plot (XY plot)
axs[1, 0].set_xlabel("X Position")
axs[1, 0].set_ylabel("Y Position")
axs[1, 0].set_title("X-Y Trajectory of the Model")

plot_track(fig, axs[1, 0], track_shape_data)

# Set up the ellipses plots
for ax_row in axs[0, :]:
    ax_row.set_xlim(-2, 2)  # Set x-axis limit between -2 and 2 for ellipses
    ax_row.set_ylim(-6, 6)  # Set y-axis limit between -4 and 4 for ellipses

axs[1,1].set_xlim(-2,2)
axs[1,1].set_ylim(-8, 8)


def is_outside_ellipsoid_n_dim(noisy_vals, means, cov_matrix):

    noisy_vals = np.array(noisy_vals)
    means = np.array(means)
    
    diff = noisy_vals - means
    
    inv_cov = np.linalg.inv(cov_matrix)
    
    dist_n_dim = diff.T @ inv_cov @ diff

    print(dist_n_dim)
    
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
    'eomega_added': None,
    'dthetaomega_added': None
}

added_point_plot_handles_expert = {
    'edtheta_expert': None,
    'eomega_expert': None,
    'dthetaomega_expert': None
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
    if added_point_plot_handles_added['eomega_added']:
        added_point_plot_handles_added['eomega_added'].remove()
    if added_point_plot_handles_added['dthetaomega_added']:
        added_point_plot_handles_added['dthetaomega_added'].remove()

    if added_point_plot_handles_expert['edtheta_expert']:
        added_point_plot_handles_expert['edtheta_expert'].remove()
    if added_point_plot_handles_expert['eomega_expert']:
        added_point_plot_handles_expert['eomega_expert'].remove()
    if added_point_plot_handles_expert['dthetaomega_expert']:
        added_point_plot_handles_expert['dthetaomega_expert'].remove()

    # Extract relevant values for the selected bucket (for the closest s)
    mean_e = bucket_data_mean_std['mean_e'][bucket_index]
    mean_dtheta = bucket_data_mean_std['mean_dtheta'][bucket_index]
    mean_omega = bucket_data_mean_std['mean_omega'][bucket_index]
    std_e = bucket_data_mean_std['std_e'][bucket_index]
    std_dtheta = bucket_data_mean_std['std_dtheta'][bucket_index]
    std_omega = bucket_data_mean_std['std_omega'][bucket_index]
    
    # Covariance matrices for each pair
    cov_matrix_edtheta = np.array(bucket_data_mean_std['cov_e_dtheta'][bucket_index]).reshape(2, 2)
    cov_matrix_eomega = np.array(bucket_data_mean_std['cov_e_omega'][bucket_index]).reshape(2, 2)
    cov_matrix_dthetaomega = np.array(bucket_data_mean_std['cov_dtheta_omega'][bucket_index]).reshape(2, 2)
    
    # Ellipse (e-dtheta) angle and dimensions
    _, eigvecs_edtheta = np.linalg.eig(cov_matrix_edtheta)
    angle_edtheta = np.arctan2(eigvecs_edtheta[1, 0], eigvecs_edtheta[0, 0]) * 180 / np.pi
    width_edtheta = 4 * std_e
    height_edtheta = 4 * std_dtheta
    
    # Plot the e-dtheta ellipse
    ellipse_edtheta = Ellipse(xy=(mean_e, mean_dtheta), width=width_edtheta, height=height_edtheta, 
                              angle=angle_edtheta, edgecolor='r', facecolor='none', linewidth=3)
    axs[0, 0].add_patch(ellipse_edtheta)

    # Plot the model data point on top of the ellipsoid
    model_e = model_data['e'][model_index]
    model_dtheta = wrap_to_pi(model_data['dtheta'][model_index])
    point.set_data(model_e, model_dtheta)

    if is_outside_ellipsoid_n_dim([model_e, model_dtheta], [mean_e, mean_dtheta], cov_matrix_edtheta):
        point.set_marker('X') 
    else:
        point.set_marker('o')  
    point.set_data(model_e, model_dtheta)
    
    # Now add the e-omega ellipsoid
    _, eigvecs_eomega = np.linalg.eig(cov_matrix_eomega)
    angle_eomega = np.arctan2(eigvecs_eomega[1, 0], eigvecs_eomega[0, 0]) * 180 / np.pi
    width_eomega = 4 * std_e
    height_eomega = 4 * std_omega
    
    # Plot the e-omega ellipse
    ellipse_eomega = Ellipse(xy=(mean_e, mean_omega), width=width_eomega, height=height_eomega, 
                             angle=angle_eomega, edgecolor='g', facecolor='none', linewidth=3)
    axs[0, 1].add_patch(ellipse_eomega)

    model_omega = model_data['omega'][model_index]
    point1.set_data(model_e, model_omega)  # Update point for e-omega plot

    # Now add the dtheta-omega ellipsoid
    _, eigvecs_dthetaomega = np.linalg.eig(cov_matrix_dthetaomega)
    angle_dthetaomega = np.arctan2(eigvecs_dthetaomega[1, 0], eigvecs_dthetaomega[0, 0]) * 180 / np.pi
    width_dthetaomega = 4 * std_dtheta
    height_dthetaomega = 4 * std_omega
    
    # Plot the dtheta-omega ellipse
    ellipse_dthetaomega = Ellipse(xy=(mean_dtheta, mean_omega), width=width_dthetaomega, 
                                  height=height_dthetaomega, angle=angle_dthetaomega, 
                                  edgecolor='b', facecolor='none', linewidth=3)
    axs[1, 1].add_patch(ellipse_dthetaomega)

    model_dtheta = wrap_to_pi(model_data['dtheta'][model_index])
    model_omega = model_data['omega'][model_index]
    point2.set_data(model_dtheta, model_omega)  # Update point for dtheta-omega plot

    bucket_start = bucket_data_mean_std['s'][bucket_index]
    try:
        bucket_end =  bucket_data_mean_std['s'][bucket_index+1]
    except:
        bucket_start = bucket_data_mean_std['s'][0]
        bucket_end =  bucket_data_mean_std['s'][1]

    indices_in_bucket_added = (all_s_added >= bucket_start) & (all_s_added < bucket_end)
    indices_in_bucket_expert = (all_s_expert >= bucket_start) & (all_s_expert < bucket_end)
    
    # Get the corresponding e, dtheta, omega values
    added_e = np.array(all_e_added)[indices_in_bucket_added]
    added_dtheta = np.array(all_dtheta_added)[indices_in_bucket_added]
    added_omega = np.array(all_omega_added)[indices_in_bucket_added]

    expert_e = np.array(all_e_expert)[indices_in_bucket_expert]
    expert_dtheta = np.array(all_dtheta_expert)[indices_in_bucket_expert]
    expert_omega = np.array(all_omega_expert)[indices_in_bucket_expert]

    added_point_plot_handles_added['edtheta_added'] = axs[0, 0].plot(added_e, added_dtheta, 'yX', alpha=0.1, label = "Added Points")[0]  
    added_point_plot_handles_added['eomega_added'] = axs[0, 1].plot(added_e, added_omega, 'yX', alpha=0.1)[0]  
    added_point_plot_handles_added['dthetaomega_added'] = axs[1, 1].plot(added_dtheta, added_omega, 'yX', alpha=0.1)[0] 


    added_point_plot_handles_expert['edtheta_expert'] = axs[0, 0].plot(expert_e, expert_dtheta, 'mX', alpha=0.02, label= "Expert Points")[0] 
    added_point_plot_handles_expert['eomega_expert'] = axs[0, 1].plot(expert_e, expert_omega, 'mX', alpha=0.02)[0] 
    added_point_plot_handles_expert['dthetaomega_expert'] = axs[1, 1].plot(expert_dtheta, expert_omega, 'mX', alpha=0.02)[0]  

    legend_handles = [
        Line2D([0], [0], marker='X', color='y', markerfacecolor='y', markersize=8, alpha=1.0, label='Added Points'),
        Line2D([0], [0], marker='X', color='m', markerfacecolor='m', markersize=8, alpha=1.0, label='Expert Points')
    ]

    # Add the legend with the custom handles
    axs[0, 0].legend(handles=legend_handles, labels=["Added Points", "Expert Points"])

    
    # Update the trajectory line with the new model point
    # line.set_data(model_data['e'][:model_index+1], wrap_to_pi(model_data['dtheta'][:model_index+1]))
    
    # Update the side plot with the x and y positions
    line_side.set_data(all_x[:model_index+1], all_y[:model_index+1])
    point_side.set_data(all_x[model_index], all_y[model_index])

    return line, point, line1, point1, line2, point2, line_side, point_side  # Return the updated objects

# Function to find the closest s from model data to a given s in bucket data
def find_closest_bucket_s(model_s):
    diffs = np.abs(np.array(bucket_data_mean_std['s']) - model_s)
    closest_index = np.argmin(diffs)
    return closest_index


def stop_animation(event):
    ani.event_source.stop()

ax_slider = plt.axes([0.1, 0.01, 0.6, 0.075])  # Position of the slider
slider = Slider(ax_slider, 'Frame', 0, len(all_s)-1, valinit=0, valstep=1)


# Declare glob_frame globally
glob_frame = 0

# Function to update the frame based on slider position
def update_slider(val):
    global glob_frame
    glob_frame = int(slider.val)  # Update the global frame based on slider position
    bucket_index = find_closest_bucket_s(model_data['s'][glob_frame])
    plot_ellipsoid_and_model(bucket_index, glob_frame)
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

# Create the animation
ani = FuncAnimation(fig, update, frames=len(all_s), interval=0.1, repeat=False)


# Create Play button
ax_button_play = plt.axes([0.85, 0.01, 0.1, 0.075])  # Button position
button_start = Button(ax_button_play, 'Play', color='lightgoldenrodyellow', hovercolor='orange')
button_start.on_clicked(play_animation)

# Create Stop button
ax_button_stop = plt.axes([0.75, 0.01, 0.1, 0.075])  # Button position
button_stop = Button(ax_button_stop, 'Stop', color='lightgoldenrodyellow', hovercolor='orange')
button_stop.on_clicked(stop_animation)

plt.show()
