import matplotlib.pyplot as plt
import numpy as np

# Data from the table
models = ['39 (Base)', '37', '38', '41', '42', '44', '45']
noise_levels = ['Very Low', 'Low', 'Very Low', 'High', 'Low', 'Low', 'Low']
collisions = [0, 8, 2, 126, 0, 13, 24]
outliers = [0.16, 0.0, 0.0, 7.47, 0.0, 4.30, 5.90]
ott = [31.10, 27.73, 33.75, 27.61, 25.79, 29.03, 30.66]

# Define the noise_map for converting noise levels to numeric values
noise_map = {'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4}

# Normalize the data
def normalize(values):
    min_val = min(values)
    max_val = max(values)
    return [(v - min_val) / (max_val - min_val) for v in values]

# Normalize the data for plotting
normalized_noise_levels = normalize([noise_map[level] for level in noise_levels])
normalized_collisions = normalize(collisions)
normalized_outliers = normalize(outliers)
normalized_ott = normalize(ott)

# Set up the figure
fig, ax = plt.subplots(figsize=(20, 10))

# Plot Noise Level (normalized)
ax.plot(models, normalized_noise_levels, marker='o', color='c', linestyle='-', label=r'$\mathcal{N}_{noise}$', linewidth=2)

# Plot Number of Collisions (normalized)
ax.plot(models, normalized_collisions, marker='o', color='g', linestyle='-', label=r'$N_{collisions}$', linewidth=2)

# Plot Percentage of Outliers (normalized)
ax.plot(models, normalized_outliers, marker='o', color='b', linestyle='-', label=r'$M_{out}$', linewidth=2)

# Plot Optimal Transport Cost (OTT) (normalized)
ax.plot(models, normalized_ott, marker='o', color='r', linestyle='-', label='Optimal Transport Cost', linewidth=2)

# Manually position text annotations for each data point with original values
ax.text(0, normalized_noise_levels[0] + 0.05, 'Very Low', color='c', ha='center', va='bottom', fontsize=10)
ax.text(1, normalized_noise_levels[1] + 0.01, 'Low', color='c', ha='center', va='bottom', fontsize=10)
ax.text(2 + 0.2, normalized_noise_levels[2] + 0.05, 'Very Low', color='c', ha='center', va='bottom', fontsize=10)
ax.text(3 , normalized_noise_levels[3] + 0.03, 'High', color='c', ha='center', va='bottom', fontsize=10)
ax.text(4 + 0.05, normalized_noise_levels[4] + 0.02, 'Low', color='c', ha='center', va='bottom', fontsize=10)
ax.text(5, normalized_noise_levels[5] - 0.025, 'Low', color='c', ha='center', va='bottom', fontsize=10)
ax.text(6, normalized_noise_levels[6] + 0.01, 'Low', color='c', ha='center', va='bottom', fontsize=10)

ax.text(0, normalized_collisions[0] + 0.09, '0', color='g', ha='center', va='top', fontsize=10)
ax.text(1, normalized_collisions[1] + 0.03, '8', color='g', ha='center', va='top', fontsize=10)
ax.text(2+ 0.2, normalized_collisions[2] + 0.08, '2', color='g', ha='center', va='top', fontsize=10)
ax.text(3, normalized_collisions[3] + 0.07, '126', color='g', ha='center', va='top', fontsize=10)
ax.text(4, normalized_collisions[4] + 0.13, '0', color='g', ha='center', va='top', fontsize=10)
ax.text(5, normalized_collisions[5] + 0.05, '13', color='g', ha='center', va='top', fontsize=10)
ax.text(6, normalized_collisions[6] - 0.05, '24', color='g', ha='center', va='top', fontsize=10)

ax.text(0, normalized_outliers[0] + 0.01, '0.16', color='b', ha='center', va='bottom', fontsize=10)
ax.text(1, normalized_outliers[1] + 0.01, '0.00', color='b', ha='center', va='bottom', fontsize=10)
ax.text(2+ 0.2, normalized_outliers[2] + 0.03, '0.00', color='b', ha='center', va='bottom', fontsize=10)
ax.text(3, normalized_outliers[3] + 0.01, '7.47', color='b', ha='center', va='bottom', fontsize=10)
ax.text(4, normalized_outliers[4] + 0.09, '0.00', color='b', ha='center', va='bottom', fontsize=10)
ax.text(5, normalized_outliers[5] + 0.01, '4.30', color='b', ha='center', va='bottom', fontsize=10)
ax.text(6, normalized_outliers[6] + 0.01, '5.90', color='b', ha='center', va='bottom', fontsize=10)

ax.text(0, normalized_ott[0] - 0.05, '31.10', color='r', ha='center', va='bottom', fontsize=10)
ax.text(1, normalized_ott[1] - 0.05, '27.73', color='r', ha='center', va='bottom', fontsize=10)
ax.text(2, normalized_ott[2] + 0.03, '33.75', color='r', ha='center', va='bottom', fontsize=10)
ax.text(3, normalized_ott[3] - 0.05, '27.61', color='r', ha='center', va='bottom', fontsize=10)
ax.text(4, normalized_ott[4] + 0.07, '25.79', color='r', ha='center', va='bottom', fontsize=10)
ax.text(5, normalized_ott[5] - 0.05, '29.03', color='r', ha='center', va='bottom', fontsize=10)
ax.text(6, normalized_ott[6] - 0.05, '30.66', color='r', ha='center', va='bottom', fontsize=10)

# Set labels and title
ax.set_ylabel('Generic Metric Axis (Normalized)')
ax.set_xlabel('Model')
ax.set_title('Performance Comparison Across Models')
ax.set_ylim(0, 1.2)

# Remove y-axis ticks
ax.set_yticks([])

# Add legend
ax.legend()

# Add explanation text box for noise levels
text_box = '\n'.join([
    'Noise Levels:',
    '4: High Noise',
    '3: Moderate Noise',
    '2: Low Noise',
    '1: Very Low Noise'
])

# plt.text(0.78, 0.84, text_box, fontsize=12, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'), transform=fig.transFigure)

# Adjust layout
plt.tight_layout()
plt.show()


fig.savefig("figures/result_comparison.svg")