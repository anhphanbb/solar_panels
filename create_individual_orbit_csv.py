# -*- coding: utf-8 -*-
"""
Generate solar panel CSV information for each orbit.

@author: anhph
"""

import pandas as pd
import os
import re
from netCDF4 import Dataset

# Path to the CSV file with filenames and intervals
csv_file_path = 'csv/solar_panels_6x6.csv'
parent_directory = 'l0b'

# Output folder for generated CSV files
output_csv_folder = 'solar_panel_csv'
os.makedirs(output_csv_folder, exist_ok=True)


# Function to extract intervals per orbit and box from CSV data
def extract_intervals_per_orbit(data):
    orbit_intervals = {}
    for _, row in data.iterrows():
        orbit = row['Orbit #']
        if pd.notna(orbit):
            orbit = int(orbit)
            orbit_intervals[orbit] = {}
            for col in data.columns:
                if "start" in col:
                    box = col.split("start")[0].strip()
                    end_col = f"{box}end"
                    if end_col in data.columns:
                        start = row[col]
                        end = row[end_col]
                        if pd.notna(start) and pd.notna(end):
                            if box not in orbit_intervals[orbit]:
                                orbit_intervals[orbit][box] = []
                            orbit_intervals[orbit][box].append((start, end))
    return orbit_intervals

# Function to search for .nc file
def find_nc_file(parent_directory, orbit_number):
    orbit_str = str(int(orbit_number)).zfill(5)
    pattern = re.compile(r'awe_l0b_(.*)_' + orbit_str + r'_(.*)\.nc')
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No file found for orbit number {orbit_number}")

# Main function to process intervals and generate CSV files
def process_intervals_and_generate_csv(data):
    orbit_intervals = extract_intervals_per_orbit(data)
    for orbit_number, boxes in orbit_intervals.items():
        try:
            nc_file_path = find_nc_file(parent_directory, orbit_number)
        except FileNotFoundError as e:
            print(e)
            continue

        with Dataset(nc_file_path, 'r') as nc:
            num_frames = len(nc.variables['Epoch'])
            
            # Initialize a list to store CSV rows
            csv_rows = []
            
            # Process each box and its intervals
            for box, intervals in boxes.items():
                for frame in range(num_frames):
                    in_interval = any(interval[0] <= frame <= interval[1] for interval in intervals)
                    solar_panel = 1 if in_interval else 0
                    csv_rows.append({"Frame": frame, "Box": box, "Solar Panel": solar_panel})
            
            # Save CSV file for the orbit
            output_csv_path = os.path.join(output_csv_folder, f"orbit_{orbit_number}_solar_panel.csv")
            pd.DataFrame(csv_rows).to_csv(output_csv_path, index=False)
            print(f"Generated CSV for Orbit {orbit_number}: {output_csv_path}")

# Load data
data = pd.read_csv(csv_file_path)

# Generate CSV files for all orbits
process_intervals_and_generate_csv(data)
