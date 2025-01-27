# -*- coding: utf-8 -*-
"""
Calculate and plot heat maps for solar panel metrics.

@author: anhph
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
import re

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

# Define thresholds
threshold = 0.5  # MLSP value considered as solar panel presence

# Initialize arrays to store results
total_count = np.zeros((6, 6), dtype=int)  # Total count of solar panel frames per box
longest_streak = np.zeros((6, 6), dtype=int)  # Longest streak of solar panel frames per box

# Function to calculate the longest streak
def calculate_longest_streak(data):
    """Calculate the longest streak of `True` in a Boolean array."""
    padded_data = np.concatenate(([0], data, [0]))  # Pad with 0 at both ends
    streak_boundaries = np.diff(padded_data.astype(int))  # Identify transitions
    starts = np.where(streak_boundaries == 1)[0]  # Start of streaks
    ends = np.where(streak_boundaries == -1)[0]  # End of streaks
    streak_lengths = ends - starts  # Calculate lengths of streaks
    return np.max(streak_lengths) if len(streak_lengths) > 0 else 0

# Process MLSP data to calculate metrics
for j in range(6):  # y_box_across_track
    for i in range(6):  # x_box_along_track
        box_data = mlsp[:, j, i] > threshold  # Boolean array for solar panel presence
        total_count[j, i] = np.sum(box_data)  # Total count
        longest_streak[j, i] = calculate_longest_streak(box_data)  # Longest streak

# Function to plot heat maps
def plot_heat_map(data, title, output_filename, cmap="viridis"):
    plt.figure(figsize=(8, 6))
    plt.imshow(data, cmap=cmap, origin='lower')
    plt.colorbar(label='Count / Streak Length')
    plt.title(title, fontsize=16)
    plt.xlabel('X Box Index', fontsize=14)
    plt.ylabel('Y Box Index', fontsize=14)
    plt.xticks(ticks=np.arange(6), labels=np.arange(6))
    plt.yticks(ticks=np.arange(6), labels=np.arange(6))
    plt.tight_layout()
    # plt.savefig(output_filename, dpi=300)
    plt.show()
    # print(f"Plot saved as {output_filename}")

# Plot total solar panel frame count heat map
plot_heat_map(
    total_count,
    f'Total Solar Panel Frame Count (Orbit {orbit_number})',
    f"total_count_heatmap_orbit_{orbit_number}.png"
)

# Plot longest streak heat map
plot_heat_map(
    longest_streak,
    f'Longest Streak of Solar Panel Frames (Orbit {orbit_number})',
    f"longest_streak_heatmap_orbit_{orbit_number}.png"
)
