# -*- coding: utf-8 -*-
"""
Save MLSP > threshold binary values of the two largest clusters into new NetCDF files with expansion for all orbits.

@author: anhph
"""

import netCDF4 as nc
import numpy as np
from scipy.ndimage import label
import os
import re
from scipy.spatial import ConvexHull, Delaunay

# Paths and parameters
parent_directory = 'nc_files_with_mlsp'
output_directory = 'nc_files_with_mlspb'
os.makedirs(output_directory, exist_ok=True)

threshold = 0.3
running_average_window = 5
expansion_factor = 1.5

# Filename pattern
pattern = re.compile(r'awe_l(.*)_(\d{5})_(.*)\.nc')

# Function to calculate running average
def calculate_running_average(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=data)

# Function to expand cluster points, handling straight-line or flat clusters
def expand_cluster_points(cluster_points, mlsp, expansion_factor):
    if len(cluster_points) < 4:  # Not enough points for a convex hull
        return cluster_points

    try:
        # Check dimensionality of the cluster
        min_coords = cluster_points.min(axis=0)
        max_coords = cluster_points.max(axis=0)
        dimensions = (max_coords - min_coords) > 0  # Check variability along each dimension

        if dimensions.sum() < 3:  # Less than 3 dimensions (e.g., line or flat cluster)
            print("Cluster is lower-dimensional. Skipping ConvexHull expansion.")
            return cluster_points

        # Regular ConvexHull expansion for full-dimensional clusters
        hull = ConvexHull(cluster_points)
        hull_points = cluster_points[hull.vertices]
        centroid = np.mean(hull_points, axis=0)
        expanded_hull_points = centroid + expansion_factor * (hull_points - centroid)

        # Select points inside the expanded hull
        all_points = np.column_stack(np.where(mlsp > 0.1))
        inside_hull = Delaunay(expanded_hull_points).find_simplex(all_points) >= 0
        return np.unique(np.vstack((cluster_points, all_points[inside_hull])), axis=0)

    except Exception as e:
        print(f"Error expanding cluster: {e}")
        return cluster_points

# Function to select and expand clusters
def select_and_expand_clusters(thresholded, mlsp, expansion_factor):
    structure = np.zeros((3, 3, 3), dtype=int)
    structure[1, 1, :] = structure[1, :, 1] = structure[:, 1, 1] = 1
    labeled_array, num_features = label(thresholded, structure=structure)

    # Find all clusters
    cluster_sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude background (0)
    clusters = [(cluster_id + 1, size) for cluster_id, size in enumerate(cluster_sizes)]

    # Get clusters starting at frame 0
    clusters_start_0 = [c for c in clusters if np.any(labeled_array[0] == c[0])]
    largest_start_0 = max(clusters_start_0, key=lambda c: c[1], default=None)

    # Get clusters ending at the last frame
    clusters_end_last = [c for c in clusters if np.any(labeled_array[-1] == c[0])]
    largest_end_last = max(clusters_end_last, key=lambda c: c[1], default=None)

    # If they're the same, only use one
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

    # Create the binary MLSPB array
    mlspb = np.zeros_like(labeled_array, dtype=np.uint8)
    for i, cluster_id in enumerate(selected_clusters):
        cluster_mask = labeled_array == cluster_id
        z, y, x = np.where(cluster_mask)
        cluster_points = np.column_stack((z, y, x))

        # Initial size and dimensions
        t_range = (z.min(), z.max())
        y_range = (y.min(), y.max())
        x_range = (x.min(), x.max())
        print(f"Cluster {i+1}: Initial size = {len(cluster_points)} points")
        print(f"Cluster {i+1} Initial Dimensions (t: {t_range}, y: {y_range}, x: {x_range})")

        # Expand the cluster
        combined_points = expand_cluster_points(cluster_points, mlsp, expansion_factor)

        # Expanded size and dimensions
        expanded_t_range = (combined_points[:, 0].min(), combined_points[:, 0].max())
        expanded_y_range = (combined_points[:, 1].min(), combined_points[:, 1].max())
        expanded_x_range = (combined_points[:, 2].min(), combined_points[:, 2].max())
        print(f"Cluster {i+1}: Final size after expansion = {len(combined_points)} points")
        print(f"Cluster {i+1} Expanded Dimensions (t: {expanded_t_range}, y: {expanded_y_range}, x: {expanded_x_range})")

        mlspb[tuple(combined_points.T)] = 1

    return mlspb

# Process all files
for root, dirs, files in os.walk(parent_directory):
    for file in files:
        if pattern.match(file):
            dataset_path = os.path.join(root, file)
            dataset = nc.Dataset(dataset_path, 'r')
            mlsp = dataset.variables['MLSP'][:]
            mlsp[:, :, :3] = 0  # Set values to 0 for x < 3

            averaged_mlsp = calculate_running_average(mlsp, running_average_window)
            thresholded = averaged_mlsp > threshold

            mlspb = select_and_expand_clusters(thresholded, averaged_mlsp, expansion_factor)

            # Save to NetCDF
            new_filename = re.sub(r'l0b', r'l0s', file)
            new_filepath = os.path.join(output_directory, new_filename)

            with nc.Dataset(new_filepath, 'w', format='NETCDF4') as new_dataset:
                for name, dimension in dataset.dimensions.items():
                    new_dataset.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

                for name, variable in dataset.variables.items():
                    if name != 'MLSP':
                        new_var = new_dataset.createVariable(name, variable.datatype, variable.dimensions)
                        new_var.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})
                        new_var[:] = variable[:]

                mlspb_var = new_dataset.createVariable('MLSPB', 'u1', dataset.variables['MLSP'].dimensions)
                mlspb_var.setncatts({'description': f'Binary representation of MLSP > {threshold}, expanded clusters starting at frame 0 or ending at the last frame'})
                mlspb_var[:] = mlspb

            print(f"New NetCDF file saved: {new_filepath}")
