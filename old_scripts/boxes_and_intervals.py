# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:51:37 2024

@author: anhph
"""

import pandas as pd

# Load the CSV file
csv_file_path = 'csv/solar_panels_6x6.csv' 
parent_directory = 'l1r_11_updated_10072024'

data = pd.read_csv(csv_file_path)

# Clean and organize the data
def extract_intervals_per_orbit(data):
    # Initialize a dictionary to store intervals by orbit
    orbit_intervals = {}

    # Iterate over each row (each orbit)
    for _, row in data.iterrows():
        orbit = row['Orbit #']  # Extract the orbit number
        if pd.notna(orbit):  # Ensure orbit is valid
            orbit = int(orbit)
            orbit_intervals[orbit] = {}

            for col in data.columns:
                # Check if column represents a grid box (e.g., "(1,0) start")
                if "start" in col:
                    box = col.split("start")[0].strip()  # Extract the box label (e.g., "(1,0)")
                    end_col = f"{box}end"  # Corresponding end column

                    # Ensure both start and end columns exist
                    if end_col in data.columns:
                        # Extract start and end times for this box
                        start = row[col]
                        end = row[end_col]

                        # Only add valid intervals
                        if pd.notna(start) and pd.notna(end):
                            if box not in orbit_intervals[orbit]:
                                orbit_intervals[orbit][box] = []
                            orbit_intervals[orbit][box].append((start, end))

    return orbit_intervals
            
def define_boxes():
    # Define ranges for x and y
    ranges = [
        (0, 42), (43, 85), (86, 128), (129, 171), (172, 214), (215, 256)
    ]
    boxes = {}
    
    # Assign ranges to each box
    for i, y_range in enumerate(ranges):
        for j, x_range in enumerate(ranges):
            box_id = f"({i},{j})"
            boxes[box_id] = {'x': x_range, 'y': y_range}
    
    return boxes

# Define the boxes
grid_boxes = define_boxes()

# Print the boxes and their ranges
for box, coords in grid_boxes.items():
    print(f"Box {box}: x={coords['x']}, y={coords['y']}")

# Get intervals for each orbit and box
orbit_box_intervals = extract_intervals_per_orbit(data)

# Print the intervals grouped by orbit and box
for orbit, boxes in orbit_box_intervals.items():
    print(f"Orbit {orbit}:")
    for box, intervals in boxes.items():
        print(f"  Box {box}:")
        for start, end in intervals:
            print(f"    Start: {start}, End: {end}")
            
