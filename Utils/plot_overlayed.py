import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.append('Utils')

from plot_track import plot_track

def wrap_to_pi(angles):
    wrapped_angles = (angles + np.pi) % (2 * np.pi) - np.pi
    wrapped_angles = np.where(wrapped_angles >= np.pi, wrapped_angles - 2 * np.pi, wrapped_angles)
    wrapped_angles = np.where(wrapped_angles < -np.pi, wrapped_angles + 2 * np.pi, wrapped_angles)
    return wrapped_angles

# Load expert data
expert_data = pd.read_feather('Obtained Model Data/pure_pursuit_artificial_df.feather')
expert_data = expert_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
expert_data = expert_data.to_dict()

all_x_expert = []
all_y_expert = []

for trajectory in expert_data.values():
    all_x_expert.extend(trajectory['x'])
    all_y_expert.extend(trajectory['y'])

all_x_expert = np.array(all_x_expert)
all_y_expert = np.array(all_y_expert)

# Load track shape data
with open("la_track.yaml", "r") as file:
    track_shape_data = yaml.safe_load(file)

# Define the models, their title mapping, and collision counts
# models = [37, 38, 39, 41, 42, 44, 45]
# title_map = {39: 1, 37: 2, 38: 3, 41: 4, 42: 5, 44: 7, 45: 8}
# collision_counts = {37: 8, 38: 2, 39: 0, 41: 126, 42: 0, 44: 13, 45: 24}

models = [40]
collision_counts = {40:0}

# Loop through the models
for model in models:
    # Load model data
    model_file = f'Obtained Model Data/model{model}_dist_wrapped.feather'
    model_data = pd.read_feather(model_file)
    model_data = model_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    model_data = model_data.to_dict()

    all_x_model = []
    all_y_model = []

    for trajectory in model_data.values():
        all_x_model.extend(trajectory['x'])
        all_y_model.extend(trajectory['y'])

    all_x_model = np.array(all_x_model)
    all_y_model = np.array(all_y_model)

    # Create the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot the track
    left_track_x, left_track_y, right_track_x, right_track_y = plot_track(fig, ax, track_shape_data)

    # Plot the trajectories
    ax.plot(all_x_expert, all_y_expert, label="Expert Trajectory", color='red', linewidth=0.5, alpha=0.1)
    ax.plot(all_x_model, all_y_model, label="Model Trajectory", color='green', linestyle='--', linewidth=0.5)

    # Add collision text
    ax.text(2.5, 2, f'Collisions: {collision_counts[model]}', ha='right', va='top', fontsize=16, color='red', fontweight="bold")

    # Add labels, title, and legend
    ax.set_xlabel("X-coordinate", fontsize=18)
    ax.set_ylabel("Y-coordinate", fontsize=18)
    # ax.set_title(f"Model {title_map[model]}", fontsize=20, fontweight="bold")

    legend = ax.legend(fontsize=17)
    legend.get_frame().set_alpha(1)  # Ensure legend frame is fully opaque

    ax.grid(True)
    ax.axis("equal")

    # Save the plot
    save_path = f"figures/model{model}/X-Y_overlayed_model_{model}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Close the figure to avoid memory issues
    plt.close(fig)

print("All plots saved successfully.")
