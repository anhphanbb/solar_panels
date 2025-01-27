# -*- coding: utf-8 -*-
"""
Save MLSP > threshold binary values of the two largest clusters into a new NetCDF file.

@author: anhph
"""

import netCDF4 as nc
import numpy as np
from scipy.ndimage import label
import os
import re

# Define the path to the parent directory where the dataset is located
parent_directory = 'nc_files_with_mlsp'

# Define the folder for the new NetCDF files
output_directory = 'nc_files_with_mlspb'
os.makedirs(output_directory, exist_ok=True)  # Ensure the output folder exists

# Define the orbit number
orbit_number = 1  # Orbit number

# Pad the orbit number with zeros until it has 5 digits
orbit_str = str(orbit_number).zfill(5)

# User-specified parameters
threshold = 0.4  # MLSP threshold
running_average_window = 3  # Averaging window size
num_clusters = 2  # Number of largest clusters to save

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

# Function to calculate the running average
def calculate_running_average(data, window_size):
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=data)

# Threshold the averaged MLSP data and remove points with x < 3
def threshold_and_filter(data, threshold, x_min):
    thresholded = data > threshold  # Apply threshold
    mask = np.ones_like(data, dtype=bool)
    mask[:, :, :x_min] = False  # Set all x < x_min to False
    return thresholded & mask

# Apply running average
averaged_mlsp = calculate_running_average(mlsp, running_average_window)

# Apply threshold and filter for x < 3
thresholded = threshold_and_filter(averaged_mlsp, threshold, x_min=3)

# Perform 3D connected component labeling
structure = np.zeros((3, 3, 3), dtype=int)
structure[1, 1, :] = 1  # Middle slice, x-axis neighbors
structure[1, :, 1] = 1  # Middle slice, y-axis neighbors
structure[:, 1, 1] = 1  # Middle slice, z-axis neighbors

labeled_array, num_features = label(thresholded, structure=structure)

# Count the size of each cluster
cluster_sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude background (0)
largest_clusters = np.argsort(cluster_sizes)[-num_clusters:][::-1]  # Get largest clusters

# Create the binary MLSPB array
mlspb = np.zeros_like(labeled_array, dtype=np.uint8)
for cluster_id in largest_clusters:
    mlspb[labeled_array == (cluster_id + 1)] = 1  # Mark as 1 if part of the cluster

# Prepare new file name by replacing 'l0b' with 'l0s'
new_filename = re.sub(r'l0b', r'l0s', dataset_filename)
new_filepath = os.path.join(output_directory, new_filename)

# Create a new NetCDF file and save MLSPB
with nc.Dataset(new_filepath, 'w', format='NETCDF4') as new_dataset:
    # Copy dimensions from the original file
    for name, dimension in dataset.dimensions.items():
        new_dataset.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

    # Copy variables from the original file except MLSP
    for name, variable in dataset.variables.items():
        if name != 'MLSP':
            new_var = new_dataset.createVariable(name, variable.datatype, variable.dimensions)
            new_var.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})
            new_var[:] = variable[:]

    # Add the MLSPB variable
    mlspb_var = new_dataset.createVariable('MLSPB', 'u1', dataset.variables['MLSP'].dimensions)
    mlspb_var.setncatts({'description': f'Binary representation of MLSP > {threshold}, largest {num_clusters} clusters (x >= 3)'})
    mlspb_var[:] = mlspb

print(f"New NetCDF file saved: {new_filepath}")
