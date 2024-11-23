import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt


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

# Estimate the joint KDE for state-action pairs
def estimate_joint_kde(states, actions, bandwidth=0.1):
    # Combine states and actions into a single array for joint distribution estimation
    state_action_data = np.hstack([states, actions])
    
    # Estimate KDE for the joint distribution
    kde_joint = gaussian_kde(state_action_data.T, bw_method=bandwidth)
    
    return kde_joint

# Marginalize over actions to get P(S)
def marginalize_kde(kde_joint, states, kde_states, num_samples=10):
    # Precompute P(S) for all states in the dataset (not sampled every time)
    p_s_all = kde_states(states.T)  # Directly use the KDE fitted on states
    
    return p_s_all

# Compute conditional probability P(A|S) = P(S, A) / P(S)
def conditional_probability(kde_joint, states, actions, kde_states, num_samples=1000, subset_size=500):
    # Combine states and actions into one array, along axis 1 (horizontal stack)
    joint_data = np.hstack([states, actions])  # Shape should be (N, 7) where N is the number of samples
    
    # Select a random subset of the data for faster evaluation
    subset_indices = np.random.choice(joint_data.shape[0], subset_size, replace=False)
    joint_data_subset = joint_data[subset_indices]
    
    # Compute the joint probability on the subset
    joint_prob = kde_joint(joint_data_subset.T)  # Transpose to (7, subset_size)
    
    # Precompute P(S) for all states in the subset (using kde_states)
    sampled_states = joint_data_subset[:, :-actions.shape[1]]  # Extract only the state part
    p_s_subset = kde_states(sampled_states.T)  # Compute P(S) for the sampled states
    
    # Conditional probability P(A|S) = P(S, A) / P(S)
    p_a_given_s = joint_prob / p_s_subset  # Element-wise division for the conditional probability
    
    return p_a_given_s, sampled_states

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

# Estimate joint KDEs for each dataset
kde_joint_expert = estimate_joint_kde(states_expert, actions_expert, bandwidth=0.1)
kde_joint_controller = estimate_joint_kde(states_controller, actions_controller, bandwidth=0.1)
kde_joint_pure_pursuit = estimate_joint_kde(states_controller_pure_pursuit, actions_controller_pure_pursuit, bandwidth=0.1)

# Estimate KDE for states (marginal distributions)
kde_states_expert = gaussian_kde(states_expert.T, bw_method=0.1)
kde_states_controller = gaussian_kde(states_controller.T, bw_method=0.1)
kde_states_pure_pursuit = gaussian_kde(states_controller_pure_pursuit.T, bw_method=0.1)

# Compute conditional probabilities and corresponding states
p_a_given_s_expert, sampled_states_expert = conditional_probability(kde_joint_expert, states_expert, actions_expert, kde_states_expert)
p_a_given_s_controller, sampled_states_controller = conditional_probability(kde_joint_controller, states_controller, actions_controller, kde_states_controller)
p_a_given_s_pure_pursuit, sampled_states_pure_pursuit = conditional_probability(kde_joint_pure_pursuit, states_controller_pure_pursuit, actions_controller_pure_pursuit, kde_states_pure_pursuit)

# Plot conditional probabilities for comparison (using the same sampled states)
plt.figure(figsize=(10, 6))

# Plot for Expert
plt.plot(sampled_states_expert[:, 0], p_a_given_s_expert, label="Expert", color='blue')

# Plot for Controller
plt.plot(sampled_states_controller[:, 0], p_a_given_s_controller, label="Controller", color='green')

# Plot for Pure Pursuit
plt.plot(sampled_states_pure_pursuit[:, 0], p_a_given_s_pure_pursuit, label="Pure Pursuit", color='red')

plt.legend()
plt.title('Conditional Probability of Actions Given States')
plt.xlabel('State (s)')
plt.ylabel('P(A|S)')
plt.show()

# Compute Bhattacharyya Distance (as an example)
def bhattacharyya_distance(p, q):
    # Compute the Bhattacharyya distance between two distributions
    return -np.log(np.sum(np.sqrt(p * q)))

# Example of computing Bhattacharyya distance for actions between expert and controller
distance_actions_expert_controller = bhattacharyya_distance(actions_expert, actions_controller)
print(f"Bhattacharyya distance for Expert vs Controller Actions: {distance_actions_expert_controller}")