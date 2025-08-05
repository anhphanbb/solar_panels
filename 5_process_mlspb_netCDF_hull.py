# -*- coding: utf-8 -*-
"""
Save MLSP > threshold binary values of the two largest clusters into new NetCDF files with expansion.
Reads from 2 NetCDF files: one with original variables, one with MLSP only.
Applies compression and retains original chunk sizes.
Processes files in parallel.

@author: anhph
"""

import os
import re
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import label
from scipy.spatial import ConvexHull, Delaunay
from concurrent.futures import ProcessPoolExecutor, as_completed

# Paths and settings
original_nc_folder = r'E:\soc\l0c\2025\06'
mlsp_nc_folder = r'E:\soc\l0c\2025\06\nc_files_with_mlsp'
output_directory = r'E:\soc\l0d\2025\06'
os.makedirs(output_directory, exist_ok=True)

threshold = 0.6  # Main classification threshold
expansion_candidate_threshold = 0.4  # Used to select points for cluster expansion

running_average_window = 5
expansion_factor = 1.5
pattern = re.compile(r'awe_l(.*)_(\d{5})_(.*)\.nc')

def calculate_running_average(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=data)

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
        all_points = np.column_stack(np.where(mlsp > expansion_candidate_threshold))
        inside_hull = Delaunay(expanded_hull_points).find_simplex(all_points) >= 0
        return np.unique(np.vstack((cluster_points, all_points[inside_hull])), axis=0)
    except Exception as e:
        print(f"Error expanding cluster: {e}")
        return cluster_points
    

def select_and_expand_clusters(thresholded, mlsp, expansion_factor):
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

    mlspb = np.zeros_like(labeled_array, dtype=np.uint8)
    for i, cluster_id in enumerate(selected_clusters):
        cluster_mask = labeled_array == cluster_id
        z, y, x = np.where(cluster_mask)
        cluster_points = np.column_stack((z, y, x))

        print(f"Cluster {i+1}: Initial size = {len(cluster_points)} points")
        print(f"Cluster {i+1} Initial Dimensions (t: {(z.min(), z.max())}, y: {(y.min(), y.max())}, x: {(x.min(), x.max())})")

        combined_points = expand_cluster_points(cluster_points, mlsp, expansion_factor)

        print(f"Cluster {i+1}: Final size after expansion = {len(combined_points)} points")
        print(f"Cluster {i+1} Expanded Dimensions (t: {combined_points[:,0].min(), combined_points[:,0].max()}, y: {combined_points[:,1].min(), combined_points[:,1].max()}, x: {combined_points[:,2].min(), combined_points[:,2].max()})")

        mlspb[tuple(combined_points.T)] = 1
    
    return mlspb

def process_file(file):
    if not pattern.match(file):
        return

    mlsp_path = os.path.join(mlsp_nc_folder, file)
    original_path = os.path.join(original_nc_folder, file.replace('l0c', 'l0c'))
    output_path = os.path.join(output_directory, file.replace('l0c', 'l0c'))

    if not os.path.exists(original_path):
        print(f"Original file not found: {original_path}")
        return

    try:
        with Dataset(original_path, 'r') as orig_ds, Dataset(mlsp_path, 'r') as mlsp_ds:
            mlsp = mlsp_ds.variables['MLSP'][:]
            mlsp[:, :, :3] = 0  # Remove solar panel glare

            averaged_mlsp = calculate_running_average(mlsp, running_average_window)
            thresholded = averaged_mlsp > threshold
            mlspb = select_and_expand_clusters(thresholded, averaged_mlsp, expansion_factor)

            with Dataset(output_path, 'w', format='NETCDF4') as new_ds:
                # Copy dimensions
                for name, dim in orig_ds.dimensions.items():
                    new_ds.createDimension(name, len(dim) if not dim.isunlimited() else None)

                # Copy global attributes
                new_ds.setncatts({attr: orig_ds.getncattr(attr) for attr in orig_ds.ncattrs()})

                # Copy variables with original chunks and compression
                for name, var in orig_ds.variables.items():
                    if name != 'MLSP':
                        chunksizes = var.chunking() if var.chunking() else None
                        new_var = new_ds.createVariable(name, var.datatype, var.dimensions, chunksizes=chunksizes, zlib=True, complevel=4)
                        new_var[:] = var[:]
                        new_var.setncatts({attr: var.getncattr(attr) for attr in var.ncattrs()})

                # Ensure MLSP dimensions are in new file
                for dim in mlsp_ds.variables['MLSP'].dimensions:
                    if dim not in new_ds.dimensions:
                        new_ds.createDimension(dim, mlsp_ds.dimensions[dim].size)

                mlspb_var = new_ds.createVariable('MLSPB', 'u1', mlsp_ds.variables['MLSP'].dimensions, zlib=True, complevel=4)
                mlspb_var.setncatts({'description': f'Binary MLSP > {threshold}, expanded clusters touching frame 0 or last'})
                mlspb_var[:] = mlspb

        print(f"Finished: {output_path}")
    except Exception as e:
        print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    files_to_process = [f for f in os.listdir(mlsp_nc_folder) if pattern.match(f)]
    max_workers = min(8, os.cpu_count())  # Adjust as needed

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in files_to_process}
        for future in as_completed(futures):
            future.result()  # Raise exception if occurred

    print("All processing completed.")
