import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Select one model to plot (example: the model 'model42_dist_wrapped.feather')
model = 'model42_dist_wrapped.feather'

# Read the model data
model_data = pd.read_feather(f'Obtained Model Data/{model}')

# Convert arrays to lists where necessary
model_data = model_data.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

# Convert the DataFrame to a dictionary
model_data = model_data.to_dict()

# Initialize empty lists for the steering data
all_steering = []

# Extract the steering data from the model's trajectories
for trajectory in model_data.values():
    all_steering.extend(trajectory["steering"])

# Convert the steering data to a numpy array
all_steering = np.array(all_steering)

# Select the first 250 samples of the steering data
first_250_steering = all_steering[:10000]

# Plot the first 250 samples of the steering data for the selected model
plt.figure(figsize=(10, 6))
plt.plot(first_250_steering)
plt.title(f'Steering Data for Model {model.split("_")[0]} (First 250 Samples)')
plt.xlabel('Time (samples)')
plt.ylabel('Steering Angle')
plt.grid(True)
plt.tight_layout()
plt.show()
