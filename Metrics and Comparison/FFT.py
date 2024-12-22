import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# List of models to process
models = ['model18_dist_wrapped.feather', 'model30_dist_wrapped.feather', 'model37_dist_wrapped.feather', "model39_dist_wrapped.feather", "all_trajectories_filtered.feather"]

# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 10))  # 2 rows, 2 columns

# Flatten axes to make indexing easier
axes = axes.flatten()

# Define frequency threshold for signal vs noise separation
low_frequency_threshold = 0.05  # Frequencies below this are considered signal

# Loop through each model and plot its FFT on a different subplot
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
    
    # Perform FFT on the steering data
    fft_values = np.fft.fft(all_steering)
    
    # Get corresponding frequencies
    frequencies = np.fft.fftfreq(len(all_steering))
    
    # Compute the magnitude of the FFT (which represents the strength of each frequency)
    fft_magnitude = np.abs(fft_values)
    
    # Take only the positive half of the frequency spectrum (because it's symmetric)
    half_spectrum = len(all_steering) // 2
    frequencies = frequencies[:half_spectrum]
    fft_magnitude = fft_magnitude[:half_spectrum]
    
    # Compute signal and noise energy for SNR calculation
    signal_indices = frequencies < low_frequency_threshold
    noise_indices = frequencies >= low_frequency_threshold
    
    # Compute signal and noise energy
    signal_energy = np.sum(fft_magnitude[signal_indices])
    noise_energy = np.sum(fft_magnitude[noise_indices])
    
    # Calculate SNR
    snr = signal_energy / noise_energy
    
    # Extract model number from the model filename (e.g., 'model30_dist_wrapped.feather' -> '30')
    if 'model' in model:
        model_number = model.split('model')[1].split('_')[0]
    
    # Plot the FFT on the corresponding subplot
    axes[i].plot(frequencies, np.log(fft_magnitude))  # Plot in log scale
    if "model" in model:
        axes[i].set_title(f'FFT of Steering Data for Model {model_number}')
    else:
        axes[i].set_title(f'FFT of Steering Data for Expert Trajectories')
    axes[i].set_xlabel('Frequency (Hz)')
    axes[i].set_ylabel('Log(Magnitude)')
    axes[i].grid(True)
    
    # Display SNR on the plot
    axes[i].text(0.95, 0.95, f'SNR: {snr:.2f}', transform=axes[i].transAxes, ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))


for j in range(i + 1, len(axes)):
    axes[j].axis('off')
    
# Adjust the layout for better spacing
plt.tight_layout()
plt.show()
