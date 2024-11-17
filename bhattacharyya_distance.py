import numpy as np
from scipy.stats import gaussian_kde, entropy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



plot_bandwidths = False
plot_dist = True


def bhattacharyya_distance(P1_samples, P2_samples, num_samples=1000, bandwidth=0.1):

    kde1 = gaussian_kde(P1_samples.T, bw_method=bandwidth) 
    kde2 = gaussian_kde(P2_samples.T, bw_method=bandwidth)
    
    # low_bounds = np.min(P1_samples, axis=0)  
    # high_bounds = np.max(P1_samples, axis=0) 
    
    # integral_1 = kde1.integrate_box(low_bounds, high_bounds)
    # integral_2 = kde2.integrate_box(low_bounds, high_bounds)
    
    # print(f"Integral of KDE1 over the box: {integral_1}")
    # print(f"Integral of KDE2 over the box: {integral_2}")
    
    sampled_points_1 = kde1.resample(num_samples)  
    sampled_points_2 = kde2.resample(num_samples)  
    
    p1_values = kde1(sampled_points_1)  
    p2_values = kde2(sampled_points_2) 
    
    bhattacharyya_coef = np.sum(np.sqrt(p1_values * p2_values)) / num_samples
    
    distance = -np.log(bhattacharyya_coef)
    
    return distance

def normalize_data(data):
    """
    Normalize the data by subtracting the mean and dividing by the standard deviation for each feature.
    """
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def extract_state_action_pairs(data):
    """
    Extract state-action pairs from trajectory data.
    """
    state_action_pairs = []
    for trajectory in data.keys():
        trajectory_data = data[trajectory]
        
        # Combine states into a single array (5D states: s, e, dtheta, vx, vy)
        states = np.vstack([trajectory_data['s'], trajectory_data['e'], trajectory_data['dtheta'], 
                            trajectory_data['vx'], trajectory_data['vy']]).T
        
        # Combine actions (steering, throttle)
        actions = np.vstack([trajectory_data['steering'], trajectory_data['throttle']]).T
        
        # Zip and append state-action pairs
        state_action_pairs.extend(zip(states, actions))
    
    return np.array([pair[0] for pair in state_action_pairs]), np.array([pair[1] for pair in state_action_pairs])

# Load expert data
expert_data = pd.read_feather('all_trajectories.feather').to_dict()
states_expert, actions_expert = extract_state_action_pairs(expert_data)

# Normalize expert data
states_expert = normalize_data(states_expert)
actions_expert = normalize_data(actions_expert)

# Load controller-generated data (Pure Pursuit controller samples)
controller_data = pd.read_feather('model18_dist.feather').to_dict()
states_controller, actions_controller = extract_state_action_pairs(controller_data)

# Normalize controller data
states_controller = normalize_data(states_controller)
actions_controller = normalize_data(actions_controller)

# Load controller-generated data (Pure Pursuit controller samples)
controller_data_pure_pursuit = pd.read_feather('model18_dist_pure_pursuit.feather').to_dict()
states_controller_pure_pursuit, actions_controller_pure_pursuit = extract_state_action_pairs(controller_data_pure_pursuit)

# Normalize Pure Pursuit controller data
states_controller_pure_pursuit = normalize_data(states_controller_pure_pursuit)
actions_controller_pure_pursuit = normalize_data(actions_controller_pure_pursuit)


# 1. **Visual Inspection**: Plot histograms and KDEs

def plot_kde(data, ax, label="Data", color='g'):
    # Create the KDE for the data
    kde = gaussian_kde(data.T)
    x = np.linspace(data.min(), data.max(), 1000)
    
    # Plot the KDE curve
    ax.plot(x, kde(x), label="KDE", color='blue')
    
    # Plot the histogram (normalized to density)
    ax.hist(data, bins=30, density=True, alpha=0.6, color=color, label="Histogram")
    
    # Set the labels and title
    ax.legend()
    ax.set_title(label)

if plot_dist == True:
    fig, axes = plt.subplots(3, 7, figsize=(30, 10))  # Create a 3x7 grid of subplots

    # Ensure axes are accessed correctly, using 2D indexing.
    plot_kde(states_expert[:, 0], axes[0, 0], "Expert States (s)", color='blue')
    plot_kde(states_expert[:, 1], axes[0, 1], "Expert States (e)", color='blue')
    plot_kde(states_expert[:, 2], axes[0, 2], "Expert States (dtheta)", color='blue')
    plot_kde(states_expert[:, 3], axes[0, 3], "Expert States (vx)", color='blue')
    plot_kde(states_expert[:, 4], axes[0, 4], "Expert States (vy)", color='blue')
    plot_kde(actions_expert[:, 0], axes[0, 5], "Expert Actions (steering)", color='blue')
    plot_kde(actions_expert[:, 1], axes[0, 6], "Expert Actions (throttle)", color='blue')

    plot_kde(states_controller[:, 0], axes[1, 0], "Controller States (s)", color='green')
    plot_kde(states_controller[:, 1], axes[1, 1], "Controller States (e)", color='green')
    plot_kde(states_controller[:, 2], axes[1, 2], "Controller States (dtheta)", color='green')
    plot_kde(states_controller[:, 3], axes[1, 3], "Controller States (vx)", color='green')
    plot_kde(states_controller[:, 4], axes[1, 4], "Controller States (vy)", color='green')
    plot_kde(actions_controller[:, 0], axes[1, 5], "Controller Actions (steering)", color='green')
    plot_kde(actions_controller[:, 1], axes[1, 6], "Controller Actions (throttle)", color='green')

    plot_kde(states_controller_pure_pursuit[:, 0], axes[2, 0], "Controller Pure Pursuit States (s)", color='red')
    plot_kde(states_controller_pure_pursuit[:, 1], axes[2, 1], "Controller Pure Pursuit States (e)", color='red')
    plot_kde(states_controller_pure_pursuit[:, 2], axes[2, 2], "Controller Pure Pursuit States (dtheta)", color='red')
    plot_kde(states_controller_pure_pursuit[:, 3], axes[2, 3], "Controller Pure Pursuit States (vx)", color='red')
    plot_kde(states_controller_pure_pursuit[:, 4], axes[2, 4], "Controller Pure Pursuit States (vy)", color='red')
    plot_kde(actions_controller_pure_pursuit[:, 0], axes[2, 5], "Controller Pure Pursuit Actions (steering)", color='red')
    plot_kde(actions_controller_pure_pursuit[:, 1], axes[2, 6], "Controller Pure Pursuit Actions (throttle)", color='red')

    plt.tight_layout()  # Automatically adjust layout to prevent overlap
    plt.show()

# 2. **Bandwidth Sensitivity**: Plot KDE with different bandwidths

def bandwidth_sensitivity(data, ax, label="Data", bandwidths=[0.1, 0.5, 1.0]):
    """
    Plot KDEs with different bandwidths on a single axis.
    """
    # Loop over each bandwidth value
    for bw in bandwidths:
        kde = gaussian_kde(data.T, bw_method=bw)
        x = np.linspace(data.min(), data.max(), 1000)
        ax.plot(x, kde(x), label=f"Bandwidth {bw}")
    
    # Plot the histogram (normalized to density)
    ax.hist(data, bins=30, density=True, alpha=0.6, color='g', label="Histogram")
    
    # Set the labels and title
    ax.legend()
    ax.set_title(label)

if plot_bandwidths == True:

    fig, axes = plt.subplots(3, 7, figsize=(30, 10))

    bandwidths = [0.1, 0.5, 1.0]

    # Plot the bandwidth sensitivity for expert states and actions
    bandwidth_sensitivity(states_expert[:, 0], axes[0, 0], "Expert States (s)", bandwidths)
    bandwidth_sensitivity(states_expert[:, 1], axes[0, 1], "Expert States (e)", bandwidths)
    bandwidth_sensitivity(states_expert[:, 2], axes[0, 2], "Expert States (dtheta)", bandwidths)
    bandwidth_sensitivity(states_expert[:, 3], axes[0, 3], "Expert States (vx)", bandwidths)
    bandwidth_sensitivity(states_expert[:, 4], axes[0, 4], "Expert States (vy)", bandwidths)
    bandwidth_sensitivity(actions_expert[:, 0], axes[0, 5], "Expert Actions (steering)", bandwidths)
    bandwidth_sensitivity(actions_expert[:, 1], axes[0, 6], "Expert Actions (throttle)", bandwidths)

    # Plot the bandwidth sensitivity for controller states and actions
    bandwidth_sensitivity(states_controller[:, 0], axes[1, 0], "Controller States (s)", bandwidths)
    bandwidth_sensitivity(states_controller[:, 1], axes[1, 1], "Controller States (e)", bandwidths)
    bandwidth_sensitivity(states_controller[:, 2], axes[1, 2], "Controller States (dtheta)", bandwidths)
    bandwidth_sensitivity(states_controller[:, 3], axes[1, 3], "Controller States (vx)", bandwidths)
    bandwidth_sensitivity(states_controller[:, 4], axes[1, 4], "Controller States (vy)", bandwidths)
    bandwidth_sensitivity(actions_controller[:, 0], axes[1, 5], "Controller Actions (steering)", bandwidths)
    bandwidth_sensitivity(actions_controller[:, 1], axes[1, 6], "Controller Actions (throttle)", bandwidths)

    # Plot the bandwidth sensitivity for controller pure pursuit states and actions
    bandwidth_sensitivity(states_controller_pure_pursuit[:, 0], axes[2, 0], "Controller Pure Pursuit States (s)", bandwidths)
    bandwidth_sensitivity(states_controller_pure_pursuit[:, 1], axes[2, 1], "Controller Pure Pursuit States (e)", bandwidths)
    bandwidth_sensitivity(states_controller_pure_pursuit[:, 2], axes[2, 2], "Controller Pure Pursuit States (dtheta)", bandwidths)
    bandwidth_sensitivity(states_controller_pure_pursuit[:, 3], axes[2, 3], "Controller Pure Pursuit States (vx)", bandwidths)
    bandwidth_sensitivity(states_controller_pure_pursuit[:, 4], axes[2, 4], "Controller Pure Pursuit States (vy)", bandwidths)
    bandwidth_sensitivity(actions_controller_pure_pursuit[:, 0], axes[2, 5], "Controller Pure Pursuit Actions (steering)", bandwidths)
    bandwidth_sensitivity(actions_controller_pure_pursuit[:, 1], axes[2, 6], "Controller Pure Pursuit Actions (throttle)", bandwidths)
    # Apply tight layout for proper spacing
    plt.tight_layout()

    # Show the bandwidth sensitivity plot
    plt.show()


# 5. **Bhattacharyya Distance**:
distance_states_expert_controller = bhattacharyya_distance(states_expert, states_controller)
print(f"Bhattacharyya distance for Expert vs Controller States: {distance_states_expert_controller}")

# Actions
distance_actions_expert_controller = bhattacharyya_distance(actions_expert, actions_controller)
print(f"Bhattacharyya distance for Expert vs Controller Actions: {distance_actions_expert_controller}")

# Compute Bhattacharyya distance for steering actions only
distance_steering_expert_controller = bhattacharyya_distance(actions_expert[:, 0], actions_controller[:, 0])
print(f"Bhattacharyya distance for Expert vs Controller Steering: {distance_steering_expert_controller}")

distance_steering_expert_pure_pursuit = bhattacharyya_distance(actions_expert[:, 0], actions_controller_pure_pursuit[:, 0])
print(f"Bhattacharyya distance for Expert vs Controller Pure Pursuit Steering: {distance_steering_expert_pure_pursuit}")

