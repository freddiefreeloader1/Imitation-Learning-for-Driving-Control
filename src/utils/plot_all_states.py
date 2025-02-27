import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import yaml
import sys
import re  # Regular expressions module
from matplotlib.ticker import ScalarFormatter

sys.path.append('Utils')

from plot_track import plot_track

def wrap_to_pi(angles):
    wrapped_angles = (angles + np.pi) % (2 * np.pi) - np.pi
    wrapped_angles = np.where(wrapped_angles >= np.pi, wrapped_angles - 2 * np.pi, wrapped_angles)
    wrapped_angles = np.where(wrapped_angles < -np.pi, wrapped_angles + 2 * np.pi, wrapped_angles)
    return wrapped_angles


# Load track data
with open("la_track.yaml", "r") as file:
    track_shape_data = yaml.safe_load(file)

# Select one model to plot (example: the model 'model45_dist_wrapped.feather')
model = 'model42_dist_wrapped.feather'

# Extract the model number from the filename using a regular expression
model_number_match = re.search(r'(\d+)', model)
model_number = model_number_match.group(1) if model_number_match else "expert"

# Read the model data
model_data = pd.read_feather(f'Obtained Model Data/{model}')

# Convert arrays to lists where necessary
model_data = model_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

# Convert the DataFrame to a dictionary
model_data = model_data.to_dict()

# Define the number of trajectories to include in the plots (change this value)
num_trajectories = 10  # Set to the number of trajectories you want to overlay

# Plot setup: create a GridSpec for the custom layout
fig = plt.figure(figsize=(16, 12))  # Adjust the overall figure size (wider aspect ratio)
gs = gridspec.GridSpec(1, 2, width_ratios=[0.4, 0.6])  # Adjust width ratios (more space for the right plot)

# Left plot: x vs y (Large plot)
ax_large = plt.subplot(gs[0])

# Right plot: 3x2 grid for other states (6 subplots)
gs_right = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[1], height_ratios=[1, 1, 1.5])  # Adjusting height ratios
axes = [plt.subplot(gs_right[i]) for i in range(6)]

# Define state labels with the requested symbols
state_labels = [r'$\mathbf{\delta}$', r'$\mathbf{e_y}$ (m)', r'$\mathbf{e_\Psi}$ (rad)', 
                r'$\mathbf{v_x^{body}}$ (m/s)', r'$\mathbf{v_y^{body}}$(m/s)', r'$\mathbf{\tau}$', 
                r'$\mathbf{x}$ (m)', r'$\mathbf{y}$ (m)', r'$\mathbf{s_p}$']


# Choose a single color for all trajectories
plot_color = 'g'  # Blue color for all trajectories

# Loop through each trajectory (up to num_trajectories)
for i, trajectory in enumerate(list(model_data.values())[:num_trajectories]):
    # Extract relevant states and s parameter for the trajectory
    s = np.array(trajectory["s"])[:-1]  # Remove the last element of s
    steering = np.array(trajectory["steering"])[:-1]  # Remove the last element of steering
    e = np.array(trajectory["e"])[:-1]  # Remove the last element of e
    dtheta = wrap_to_pi(np.array(trajectory["dtheta"])[:-1])  # Remove the last element of dtheta
    vx = np.array(trajectory["vx"])[:-1]  # Remove the last element of vx
    vy = np.array(trajectory["vy"])[:-1]  # Remove the last element of vy
    throttle = np.array(trajectory["throttle"])[:-1]  # Remove the last element of throttle
    x = np.array(trajectory["x"])[:-1]  # Remove the last element of x
    y = np.array(trajectory["y"])[:-1]  # Remove the last element of y

    # Overlay the states with respect to the 's' parameter using line plots
    axes[0].plot(s, steering, color=plot_color, alpha=0.5)  # Overlay all trajectories
    axes[1].plot(s, e, color=plot_color, alpha=0.5)  # Overlay all trajectories
    axes[2].plot(s, dtheta, color=plot_color, alpha=0.5)  # Overlay all trajectories
    axes[3].plot(s, vx, color=plot_color, alpha=0.5)  # Overlay all trajectories
    axes[4].plot(s, vy, color=plot_color, alpha=0.5)  # Overlay all trajectories
    axes[5].plot(s, throttle, color=plot_color, alpha=0.5)  # Overlay all trajectories

# Plot x vs y on the left-side plot (large plot)
for trajectory in list(model_data.values())[:num_trajectories]:
    x = np.array(trajectory["x"])[:-1]  # Remove the last element of x
    y = np.array(trajectory["y"])[:-1]  # Remove the last element of y
    ax_large.plot(x, y, color="m", alpha=0.5)

if model_number == "expert":
    ax_large.set_title(f'Expert (Controller Generated)', fontsize=20, fontweight='bold', fontname='sans-serif', pad=20)
    # ax_large.set_title(f'Expert', fontsize=20, fontweight='bold', fontname='sans-serif', pad=20)
else:
    # ax_large.set_title(f'Learned Controller from Controller Dataset', fontsize=18, fontweight='bold', fontname='sans-serif', pad=20)
    ax_large.set_title(f'Model 5', fontsize=18, fontweight='bold', fontname='sans-serif', pad=20)
ax_large.set_xlabel('x(m)', fontsize=14, fontname='sans-serif')
ax_large.set_ylabel('y(m)', fontsize=14, fontname='sans-serif')
ax_large.grid(False)

# Add a label to the plot legend for the runtime trajectories
ax_large.plot([], [], color='m', alpha=0.5, label='Runtime Trajectories')  # Empty plot to create legend entry
ax_large.legend(loc='upper right', fontsize=12)  # Add legend to the top-right corner

# Plot track data
left_track_x, left_track_y, right_track_x, right_track_y = plot_track(fig, ax_large, track_shape_data)

# Set titles and labels for each subplot (state vs s)
for i, ax in enumerate(axes[:6]):
    if model_number == "expert":
        ax.set_title(f'{state_labels[i]}', fontsize=16, fontweight='bold', fontname='sans-serif', pad=15)
    else:
        ax.set_title(f'{state_labels[i]}', fontsize=16, fontweight='bold', fontname='sans-serif', pad=15)
    ax.set_xlabel(r'$s_p$', fontsize=14, fontname='sans-serif')
    # ax.set_ylabel(state_labels[i], fontsize=14, fontname='sans-serif')
    ax.grid(True)

    # Apply the ScalarFormatter to format y-axis ticks with two significant digits
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.tick_params(axis='y', labelsize=12)  # Adjust label size if needed

    # Manually adjust the y-tick formatting to have 2 significant digits
    def format_ticks(value, pos):
        return f'{value:.2g}'  # Format ticks with 2 significant digits

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_ticks))

# Adjust layout and spacing
plt.subplots_adjust(left=0.08, right=0.92, top=0.9, bottom=0.1, hspace=0.5, wspace=0.4)  # Manually adjust spacing

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# Save the figure
fig.savefig("figures/model42_with_states.svg")
