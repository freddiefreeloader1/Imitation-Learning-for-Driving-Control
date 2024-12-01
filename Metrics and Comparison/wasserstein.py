import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn_lr

model_data = pd.read_feather('Obtained Model Data/model29_dist_wrapped.feather')
model_data = model_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
model_data = model_data.to_dict()

expert_data = pd.read_feather('Obtained Model Data/all_trajectories_filtered.feather')
expert_data = expert_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
expert_data = expert_data.to_dict()

def extract_trajectory_data(data_dict, key):
    trajectory = data_dict[key]
    s = np.array(trajectory['s'])
    e = np.array(trajectory['e'])
    dtheta = np.array(trajectory['dtheta'])
    vx = np.array(trajectory['vx'])
    vy = np.array(trajectory['vy'])
    omega = np.array(trajectory['omega'])
    # Combine the components into a 6D vector (each row is a 6D vector)
    return np.vstack([s, e, dtheta, vx, vy, omega]).T

model_trajectories = [extract_trajectory_data(model_data, key) for key in model_data.keys()]
expert_trajectories = [extract_trajectory_data(expert_data, key) for key in expert_data.keys()]

all_model_data = np.vstack(model_trajectories)
all_expert_data = np.vstack(expert_trajectories)

def downsample_data(data, max_samples=150000):
    if len(data) > max_samples:
        indices = np.random.choice(len(data), max_samples, replace=False)
        return data[indices]
    else:
        return data

all_model_data_downsampled = downsample_data(all_model_data)
all_expert_data_downsampled = downsample_data(all_expert_data, max_samples = len(all_model_data))

# Convert to JAX arrays
model_data_jax = jnp.array(all_model_data_downsampled)
expert_data_jax = jnp.array(all_expert_data_downsampled)

n_samples = model_data_jax.shape[0]  # Number of samples after downsampling
a = jnp.ones(n_samples) / n_samples  # Uniform weights for model data
b = jnp.ones(n_samples) / n_samples  # Uniform weights for expert data

key = jax.random.PRNGKey(42)

def calculate_ot_cost(x, y):
    geom = pointcloud.PointCloud(x, y)
    ot_prob = linear_problem.LinearProblem(geom, a, b)
    rank = 10 
    solver = jax.jit(sinkhorn_lr.LRSinkhorn(rank=rank, initializer="k-means", epsilon=1e-1))
    ot_lr = solver(ot_prob)
    return ot_lr.reg_ot_cost

ot_cost = calculate_ot_cost(model_data_jax, expert_data_jax)

print("OT cost between model and downsampled expert data:", ot_cost)
