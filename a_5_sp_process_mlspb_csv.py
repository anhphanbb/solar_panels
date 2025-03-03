import netCDF4 as nc
import numpy as np
from scipy.ndimage import label
import os
import re
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay

# Paths and parameters
parent_directory = 'sp_orbit_predictions/mlsp'
output_directory = 'sp_orbit_predictions/mlspb'
csv_output_path = 'sp_orbit_predictions/mlspb/cluster_expansion.csv'
os.makedirs(output_directory, exist_ok=True)

threshold = 0.3
running_average_window = 5
expansion_factor = 1.5

# Initialize list to store cluster expansions
cluster_data = []

# Function to calculate running average
def calculate_running_average(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=data)

# Function to expand cluster points
def expand_cluster_points(cluster_points, mlsp, expansion_factor):
    if len(cluster_points) < 4:
        return cluster_points
    try:
        min_coords = cluster_points.min(axis=0)
        max_coords = cluster_points.max(axis=0)
        dimensions = (max_coords - min_coords) > 0

        if dimensions.sum() < 3:
            print("Cluster is lower-dimensional. Skipping ConvexHull expansion.")
            return cluster_points

        hull = ConvexHull(cluster_points)
        hull_points = cluster_points[hull.vertices]
        centroid = np.mean(hull_points, axis=0)
        expanded_hull_points = centroid + expansion_factor * (hull_points - centroid)
        all_points = np.column_stack(np.where(mlsp > 0.1))
        inside_hull = Delaunay(expanded_hull_points).find_simplex(all_points) >= 0
        return np.unique(np.vstack((cluster_points, all_points[inside_hull])), axis=0)
    except Exception as e:
        print(f"Error expanding cluster: {e}")
        return cluster_points

# Function to select and expand clusters
def select_and_expand_clusters(thresholded, mlsp, expansion_factor, orbit_number):
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, :] = structure[1, :, 1] = structure[:, 1, 1] = 1
    labeled_array, num_features = label(thresholded, structure=structure)

    cluster_sizes = np.bincount(labeled_array.ravel())[1:]
    clusters = [(cluster_id + 1, size) for cluster_id, size in enumerate(cluster_sizes)]

    clusters_start_0 = [c for c in clusters if np.any(labeled_array[0] == c[0])]
    largest_start_0 = max(clusters_start_0, key=lambda c: c[1], default=None)
    clusters_end_last = [c for c in clusters if np.any(labeled_array[-1] == c[0])]
    largest_end_last = max(clusters_end_last, key=lambda c: c[1], default=None)

    selected_clusters = []
    if largest_start_0 and largest_end_last:
        if largest_start_0[0] == largest_end_last[0]:
            selected_clusters = [largest_start_0[0]]
        else:
            selected_clusters = [largest_start_0[0], largest_end_last[0]]
    elif largest_start_0:
        selected_clusters = [largest_start_0[0]]
    elif largest_end_last:
        selected_clusters = [largest_end_last[0]]

    total_frames = mlsp.shape[0]
    cluster_record = [orbit_number, total_frames]

    for i, cluster_id in enumerate(selected_clusters):
        cluster_mask = labeled_array == cluster_id
        z, y, x = np.where(cluster_mask)
        cluster_points = np.column_stack((z, y, x))
        t_range = (z.min(), z.max())
        print(f"Cluster {i+1} Initial Dimensions (t: {t_range})")
        combined_points = expand_cluster_points(cluster_points, mlsp, expansion_factor)
        expanded_t_range = (combined_points[:, 0].min(), combined_points[:, 0].max())
        print(f"Cluster {i+1} Expanded Dimensions (t: {expanded_t_range})")
        cluster_record.extend([expanded_t_range[0], expanded_t_range[1]])
    
    while len(cluster_record) < 6:
        cluster_record.append(None)
    cluster_data.append(cluster_record)

# Process all files
for root, dirs, files in os.walk(parent_directory):
    for file in files:
        if file.endswith('_mlsp.nc'):
            dataset_path = os.path.join(root, file)
            orbit_number = re.search(r'_(\d+)_mlsp.nc', file).group(1)
            dataset = nc.Dataset(dataset_path, 'r')
            mlsp = dataset.variables['MLSP'][:]
            mlsp[:, :, :3] = 0
            averaged_mlsp = calculate_running_average(mlsp, running_average_window)
            thresholded = averaged_mlsp > threshold
            select_and_expand_clusters(thresholded, averaged_mlsp, expansion_factor, orbit_number)

# Save cluster expansions to CSV
columns = ['Orbit', 'TotalFrames', 'Cluster1_Expanded_t0', 'Cluster1_Expanded_t1', 'Cluster2_Expanded_t0', 'Cluster2_Expanded_t1']
pd.DataFrame(cluster_data, columns=columns).to_csv(csv_output_path, index=False)

print("Processing completed.")
