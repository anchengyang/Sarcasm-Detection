import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Data for our models (Normalize values - higher is better for all metrics)
# For speed (training time), we invert the values since lower is better
models = ['RoBERTa', 'BERT', 'DistilBERT', 'ALBERT', 'SBERT']

# Accuracy values directly from data (higher is better)
accuracy = [93.68, 92.42, 92.35, 91.86, 91.45]

# Speed (training time) - invert and normalize so higher values = better speed
train_times = [456, 445, 295, 613, 144]
max_time = max(train_times)
speed = [(max_time - time + 100) / max_time * 100 for time in train_times]  # Normalize and invert

# Model size in millions of parameters - invert and normalize
model_sizes = [125, 110, 66, 12, 22]
max_size = max(model_sizes)
compactness = [(max_size - size + 10) / max_size * 100 for size in model_sizes]  # Normalize and invert

# Normalize all values to 0-100 range for consistent radar chart
# Normalize accuracy to 0-100 range (already in percentage)
min_acc = min(accuracy)
max_acc = max(accuracy)
norm_accuracy = [(acc - min_acc) / (max_acc - min_acc) * 100 for acc in accuracy]

# Data array for radar chart
data = np.array([norm_accuracy, speed, compactness])

# Function to create radar chart
def radar_chart(fig, titles, data, colors):
    # Number of variables
    N = len(titles)
    
    # Angle of each axis
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Data for each model (plus closing the loop)
    model_data = []
    for i in range(len(models)):
        model_values = data[:, i].tolist()
        model_values += [model_values[0]]  # Close the loop
        model_data.append(model_values)
    
    # Set up radar chart
    ax = fig.add_subplot(111, polar=True)
    
    # Draw axis lines
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], titles, size=12, fontweight='bold')
    
    # Draw ylabels (we'll use 0-100%)
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75], ["25", "50", "75"], color="grey", size=10)
    plt.ylim(0, 100)
    
    # Plot data for each model
    for i, model in enumerate(models):
        ax.plot(angles, model_data[i], color=colors[i], linewidth=2, label=model)
        ax.fill(angles, model_data[i], color=colors[i], alpha=0.25)
    
    # Add legend - Move to top-right outside the main plot area
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    return ax

# Set up figure - Increase width to accommodate legend on the right
fig = plt.figure(figsize=(12, 8))

# Adjust subplots to leave room for the legend and caption
fig.subplots_adjust(top=0.85, bottom=0.15, right=0.75)

# Set titles for the axes
titles = ['Accuracy', 'Speed', 'Compactness']

# Define colors for each model
colors = ['#FF4B4B', '#4B4BFF', '#FFB000', '#00C9A7', '#C47AFF']

# Create radar chart
radar_chart(fig, titles, data, colors)

# Add title
plt.title('BERT Model Variants: Performance Trade-offs', size=20, y=1.1)

# Add subtitle explaining the metrics - Moved higher to avoid overlap with the bottom margin
plt.figtext(0.5, 0.05, 'Higher values are better for all metrics.\nSpeed reflects inverse of training time, Compactness reflects inverse of model size.', 
            ha='center', fontsize=12, style='italic')

# Save figure
plt.savefig('bert_models_radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

print("Radar chart saved as 'bert_models_radar_chart.png'") 