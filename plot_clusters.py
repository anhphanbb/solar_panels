# -*- coding: utf-8 -*-
"""
Calculate and plot heat maps and 3D visualization for solar panel metrics with running average.

@author: anhph
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from scipy.ndimage import label
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
import re
import os

# Define the path to the parent directory where the dataset is located
parent_directory = 'nc_files_with_mlsp'

# Define the orbit number
orbit_number = 110  # Orbit number

# Pad the orbit number with zeros until it has 5 digits
orbit_str = str(orbit_number).zfill(5)

# Search for the correct file name in all subdirectories
pattern = re.compile(r'awe_l(.*)_' + orbit_str + r'_(.*)\.nc')
dataset_filename = None
dataset_path = None

for root, dirs, files in os.walk(parent_directory):
    for file in files:
        if pattern.match(file):
            dataset_filename = file
            dataset_path = os.path.join(root, file)
            break
    if dataset_filename:
        break

if dataset_filename is None:
    raise FileNotFoundError(f"No file found for orbit number {orbit_str}")

# Load the dataset
dataset = nc.Dataset(dataset_path, 'r')
mlsp = dataset.variables['MLSP'][:]  # Load MLSP variable

# Define thresholds and parameters
threshold = 0.6  # MLSP value considered as solar panel presence
running_average_window = 5  # Default running average window size


# Function to calculate the running average
def calculate_running_average(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=data)


# Function to plot 3D visualization of clusters
def highlight_largest_clusters(averaged_mlsp, threshold, orbit_number, num_clusters=5):
    # Threshold the averaged MLSP data
    thresholded = averaged_mlsp > threshold

    # Perform 3D connected component labeling
    # structure = np.ones((3, 3, 3))  # Define connectivity (26-connected in 3D)
    
    # # Use Edge-Connected (18-Connected) in 3D
    # structure = np.ones((3, 3, 3), dtype=int)  # Start with full connectivity
    # structure[0, 0, 0] = 0  # Remove corners
    # structure[0, 0, 2] = 0
    # structure[0, 2, 0] = 0
    # structure[0, 2, 2] = 0
    # structure[2, 0, 0] = 0
    # structure[2, 0, 2] = 0
    # structure[2, 2, 0] = 0
    # structure[2, 2, 2] = 0
    
    # Face-Connected (6-Connected) in 3D
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, :] = 1  # Middle slice, x-axis neighbors
    structure[1, :, 1] = 1  # Middle slice, y-axis neighbors
    structure[:, 1, 1] = 1  # Middle slice, z-axis neighbors

    
    labeled_array, num_features = label(thresholded, structure=structure)

    # Count the size of each cluster
    cluster_sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude background (0)
    largest_clusters = np.argsort(cluster_sizes)[-num_clusters:][::-1]  # Get largest clusters

    # Prepare for plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"3D Clusters of MLSP > {threshold:.2f} (Orbit {orbit_number})", fontsize=16)
    ax.set_xlabel('Frame (Time Step)', fontsize=12)
    ax.set_ylabel('Y Box Index', fontsize=12)
    ax.set_zlabel('X Box Index', fontsize=12)

    # Define distinct colors for each cluster
    colors = list(mcolors.TABLEAU_COLORS.values())[:num_clusters]

    # Plot the largest clusters
    for i, cluster_id in enumerate(largest_clusters):
        cluster_mask = labeled_array == (cluster_id + 1)  # Cluster IDs start from 1
        z, y, x = np.where(cluster_mask)  # Get the coordinates of the cluster

        # Scatter plot the cluster points with unique color
        ax.scatter(z, y, x, color=colors[i], label=f'Cluster {i+1} ({cluster_sizes[cluster_id]} points)', alpha=0.7)

    ax.legend()
    ax.view_init(30, 120)  # Adjust view angle
    plt.tight_layout()
    plt.show()


# Compute the running average for MLSP
averaged_mlsp = calculate_running_average(mlsp, running_average_window)

# Highlight the largest clusters
highlight_largest_clusters(averaged_mlsp, threshold, orbit_number, num_clusters=5)
