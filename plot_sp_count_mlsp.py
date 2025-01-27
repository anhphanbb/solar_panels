# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:18:49 2024

@author: anhph
"""

# -*- coding: utf-8 -*-
"""
Plot number of solar panel boxes across frames.

@author: anhph
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# Define the path to the parent directory where the dataset is located
parent_directory = 'nc_files_with_mlsp'

# Define the orbit number
orbit_number = 90  # orbit number

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

# Count solar panel boxes (MLSP > 0.5) for each frame
solar_panel_counts = np.sum(mlsp > 0.5, axis=(1, 2))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(solar_panel_counts, label='Solar Panel Box Count', color='blue', linewidth=2)
plt.title(f'Number of Solar Panel Boxes Across Frames (Orbit {orbit_number})', fontsize=16)
plt.xlabel('Frame Number', fontsize=14)
plt.ylabel('Number of Solar Panel Boxes', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

# Save and display the plot
output_plot_path = f"solar_panel_counts_orbit_{orbit_number}.png"
# plt.savefig(output_plot_path, dpi=300)
plt.show()

# print(f"Plot saved as {output_plot_path}")
