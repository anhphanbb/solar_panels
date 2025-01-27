# -*- coding: utf-8 -*-
"""
Plot number of solar panel boxes across frames, combining NetCDF and CSV data.

@author: anhph
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

# Define the paths
# parent_directory = 'nc_files_with_mlsp'  # Path to NetCDF files
parent_directory = 'nc_files_with_mlspb'  # Path to NetCDF files
csv_directory = 'solar_panel_csv'       # Path to CSV files
orbit_number = 1                       # Define the orbit number

# Pad the orbit number with zeros until it has 5 digits
orbit_str = str(orbit_number).zfill(5)

# Search for the NetCDF file
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

# Load the NetCDF file and calculate solar panel counts
dataset = nc.Dataset(dataset_path, 'r')
# mlsp = dataset.variables['MLSP'][:]  # Load MLSP variable
mlspb = dataset.variables['MLSPB'][:]  # Load MLSP variable
solar_panel_counts_nc = np.sum(mlspb == 1, axis=(1, 2))

# Search for the corresponding CSV file
csv_filename = f"orbit_{orbit_number}_solar_panel.csv"
csv_path = os.path.join(csv_directory, csv_filename)

# Load CSV file and calculate solar panel counts if available
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
    solar_panel_counts_csv = data.groupby('Frame')['Solar Panel'].sum().values
else:
    solar_panel_counts_csv = None

# Plot the results
plt.figure(figsize=(12, 8))

# Plot NetCDF data
plt.plot(
    solar_panel_counts_nc,
    label=f'Solar Panel Box Count (NetCDF, Orbit {orbit_number})',
    color='blue', linewidth=2
)

# Plot CSV data if available
if solar_panel_counts_csv is not None:
    plt.plot(
        solar_panel_counts_csv,
        label=f'Solar Panel Box Count (CSV, Orbit {orbit_number})',
        color='orange', linestyle='--', linewidth=2
    )

plt.title(f'Number of Solar Panel Boxes Across Frames (Orbit {orbit_number})', fontsize=16)
plt.xlabel('Frame Number', fontsize=14)
plt.ylabel('Number of Solar Panel Boxes', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.tight_layout()

# Save and display the plot
# output_plot_path = f"solar_panel_counts_combined_orbit_{orbit_number}.png"
# plt.savefig(output_plot_path, dpi=300)
plt.show()

# print(f"Plot saved as {output_plot_path}")
