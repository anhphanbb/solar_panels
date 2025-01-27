# -*- coding: utf-8 -*-
"""
Calculate and plot heat maps and 3D visualization for solar panel metrics with running average.

@author: anhph
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from scipy.ndimage import label, center_of_mass
# from sklearn.linear_model import LinearRegression
import matplotlib.colors as mcolors
import re
import os
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

# Define the path to the parent directory where the dataset is located
parent_directory = 'nc_files_with_mlsp'

# Define the orbit number
orbit_number = 90 # Orbit number

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
# Set values to 0 for x < 3
mlsp[:, :, :3] = 0  # Set all values in x < 3 to 0

# Define thresholds and parameters
threshold = 0.4  # MLSP value considered as solar panel presence
running_average_window = 5  # Default running average window size

# Function to calculate the running average
def calculate_running_average(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=data)



def fit_line_and_envelope_with_expansion(averaged_mlsp, threshold, orbit_number, num_clusters=5, expansion_factor=1.2):
    # Threshold the averaged MLSP data
    thresholded = averaged_mlsp > threshold

    # Use Face-Connected (6-Connected) in 3D
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, :] = 1  # Middle slice, x-axis neighbors
    structure[1, :, 1] = 1  # Middle slice, y-axis neighbors
    structure[:, 1, 1] = 1  # Middle slice, z-axis neighbors

    # Perform 3D connected component labeling
    labeled_array, num_features = label(thresholded, structure=structure)

    # Count the size of each cluster
    cluster_sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude background (0)
    largest_clusters = np.argsort(cluster_sizes)[-num_clusters:][::-1]  # Get largest clusters

    # Prepare for plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"3D Clusters of MLSP > {threshold:.2f} with Fitted Lines and Expanded Envelopes (Orbit {orbit_number})", fontsize=16)
    ax.set_xlabel('Frame (Time Step)', fontsize=12)
    ax.set_ylabel('Y Box Index', fontsize=12)
    ax.set_zlabel('X Box Index', fontsize=12)

    # Define distinct colors for each cluster
    colors = list(mcolors.TABLEAU_COLORS.values())[:num_clusters]

    for i, cluster_id in enumerate(largest_clusters):
        cluster_mask = labeled_array == (cluster_id + 1)  # Cluster IDs start from 1
        z, y, x = np.where(cluster_mask)  # Get the coordinates of the cluster

        # Scatter plot the cluster points
        ax.scatter(z, y, x, color=colors[i], label=f'Cluster {i+1} ({cluster_sizes[cluster_id]} points)', alpha=0.7)

        # Combine coordinates into a single array
        points = np.column_stack((z, y, x))

        # Create convex hull if there are enough points
        if len(points) > 3:  # ConvexHull requires at least 4 points
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]

            # Calculate the centroid of the cluster
            centroid = np.mean(hull_points, axis=0)

            # Expand the convex hull points outward
            expanded_points = centroid + expansion_factor * (hull_points - centroid)

            # Plot the standard convex hull
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], color=colors[i], alpha=0.5)

            # Create a new convex hull for the expanded points
            expanded_hull = ConvexHull(expanded_points)
            for simplex in expanded_hull.simplices:
                ax.plot(expanded_points[simplex, 0], expanded_points[simplex, 1], expanded_points[simplex, 2],
                        color=colors[i], linestyle='--', alpha=0.5)

        # Calculate centroids for each frame
        centroids = []
        for frame in range(averaged_mlsp.shape[0]):
            frame_mask = cluster_mask[frame]
            if np.any(frame_mask):  # Check if the cluster exists in this frame
                centroid = center_of_mass(frame_mask)
                centroids.append((frame, centroid[0], centroid[1]))

        # Fit a line through the centroids
        if len(centroids) > 1:
            centroids = np.array(centroids)  # Convert to numpy array
            pca = PCA(n_components=1)       # PCA for 3D line fitting
            pca.fit(centroids)

            # Extract the principal direction and the mean
            direction = pca.components_[0]
            mean_point = pca.mean_

            # Generate points along the line
            t = np.linspace(-len(centroids)/2, len(centroids)/2, len(centroids))  # Match number of centroids
            line_points = mean_point + t[:, None] * direction

            # Plot the fitted line
            ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], color=colors[i], linewidth=2)

    ax.legend()
    ax.view_init(30, 120)  # Adjust view angle
    plt.tight_layout()
    plt.show()

# Compute the running average for MLSP
averaged_mlsp = calculate_running_average(mlsp, running_average_window)

# Fit lines and create envelopes (expanded by 20%) around the largest clusters
fit_line_and_envelope_with_expansion(averaged_mlsp, threshold, orbit_number, num_clusters=2, expansion_factor=1.5)
