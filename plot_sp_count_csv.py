# -*- coding: utf-8 -*-
"""
Plot total number of solar panel boxes for each frame in an orbit.

@author: anhph
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the path to the CSV file
orbit_csv_file = 'solar_panel_csv/orbit_90_solar_panel.csv' 

# Check if the file exists
if not os.path.exists(orbit_csv_file):
    raise FileNotFoundError(f"CSV file not found: {orbit_csv_file}")

# Load the CSV file
data = pd.read_csv(orbit_csv_file)

# Check the structure of the CSV
if 'Frame' not in data.columns or 'Solar Panel' not in data.columns:
    raise ValueError("CSV file must contain 'Frame' and 'Solar Panel' columns.")

# Group by frames and sum the solar panel counts
solar_panel_counts = data.groupby('Frame')['Solar Panel'].sum()

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(solar_panel_counts.index, solar_panel_counts.values, label='Solar Panel Boxes', color='blue', linewidth=2)
plt.title('Total Solar Panel Boxes per Frame', fontsize=16)
plt.xlabel('Frame', fontsize=14)
plt.ylabel('Number of Solar Panel Boxes', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Save and show the plot
output_plot_path = f"solar_panel_counts_orbit_{os.path.basename(orbit_csv_file).split('_')[1]}.png"
plt.savefig(output_plot_path, dpi=300)
plt.legend(fontsize=12)
plt.show()

print(f"Plot saved as {output_plot_path}")
