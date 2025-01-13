import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import welch

# List of models to process
models = ['model37_dist_wrapped.feather', 'model38_dist_wrapped.feather', 
          "model39_dist_wrapped.feather",  "model41_dist_wrapped.feather",  "model42_dist_wrapped.feather", "model44_dist_wrapped.feather", 'model45_dist_wrapped.feather', 
          "all_trajectories_filtered.feather"]

# models = ["all_trajectories_filtered.feather"]

num_models = len(models)

# Check if there is only one model
if num_models == 1:
    # If only one model, create a single plot (not a subplot grid)
    fig, ax = plt.subplots(figsize=(8, 6))  # Single plot, adjust size as necessary
else:
    # Calculate the number of rows and columns for the subplot grid
    n_cols = 3  # Fixed number of columns (you can adjust this if you prefer)
    n_rows = math.ceil(num_models / n_cols)  # Calculate the number of rows needed

    # Create the grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))

    # Flatten axes to make indexing easier
    axes = axes.flatten()

# Define frequency threshold for signal vs noise separation
low_frequency_threshold = 5  # Frequencies below this are considered signal

# Loop through each model and plot its PSD using Welch's method on a different subplot
for i, model in enumerate(models):
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
    nfft = 1000
    # Compute the Power Spectral Density using Welch's method
    fs = 250  # Sampling rate (Hz), adjust as necessary
    f, Pxx = welch(all_steering, fs=fs, window='hamming', nperseg=1000, noverlap=0, scaling="density", nfft=nfft)

    # If there's only one model, use the single axis for plotting
    if num_models == 1:
        ax.semilogy(f, Pxx)  # Plot in log scale to reduce large magnitude range
        ax.set_title('Expert DFT')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (V^2/Hz)')
        ax.grid(True)
        ax.set_ylim(1e-8, 1e-0)
    else:
        # Plot the PSD on the corresponding subplot (using a logarithmic scale for better visualization)
        axes[i].semilogy(f, Pxx)  # Plot in log scale to reduce large magnitude range
        
        # Check if the model is 'all_trajectories_filtered.feather'
        if model == "all_trajectories_filtered.feather":
            axes[i].set_title('Expert DFT')
        else:
            # Extract model number from the filename (e.g., 'model30' -> 30)
            model_number = model.split('_')[0].replace('model', '')
            axes[i].set_title(f'Model {model_number}')
        
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('Power Spectral Density (V^2/Hz)')
        axes[i].grid(True)

        axes[i].set_ylim(1e-8, 1e-0)
    
    # Display SNR on the plot
    # For SNR, calculate signal energy (below threshold) and noise energy (above threshold)
    signal_indices = f < low_frequency_threshold
    noise_indices = f >= low_frequency_threshold
    
    # Compute signal and noise energy
    signal_energy = np.sum(Pxx[signal_indices])
    noise_energy = np.sum(Pxx[noise_indices])
    
    # Calculate SNR
    snr = signal_energy / noise_energy
    # axes[i].text(0.95, 0.95, f'SNR: {snr:.2f}', transform=axes[i].transAxes, ha='right', va='top', fontsize=12, 
    #              bbox=dict(facecolor='white', alpha=0.7))

# Hide unused subplots (if there are any)
if num_models > 1:
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

# Adjust the layout for better spacing
plt.tight_layout()
plt.savefig('figures/PSD_welch.svg', format='svg')  # Save the plot
plt.show()
