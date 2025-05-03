# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:51:37 2024

@author: Anh

Update Apr 29 2025: Generate MLSPB variable from the csv file and save updated NetCDF files.
"""

import pandas as pd
import os
from netCDF4 import Dataset
import numpy as np
import re

# Path to the CSV file with filenames and intervals
csv_file_path = 'csv/solar_panels_6x6_april_29_2025.csv'
parent_directory = r'E:\soc\l0c\2025\so01'
output_directory = r'E:\soc\l0c\2025\so01\mlspb_from_labels'

# Ensure output folder exists
os.makedirs(output_directory, exist_ok=True)

# Define 6x6 boxes
def define_boxes():
    ranges = [
        (0, 42), (43, 85), (86, 128), (129, 171), (172, 214), (215, 256)
    ]
    boxes = {}
    for j, y_range in enumerate(ranges):
        for i, x_range in enumerate(ranges):
            box_id = f"({i},{j})"
            boxes[box_id] = {'x': x_range, 'y': y_range}
    return boxes

grid_boxes = define_boxes()

# Extract intervals from CSV
def extract_intervals(data):
    orbit_intervals = {}
    glare_intervals = {}
    for _, row in data.iterrows():
        orbit = row['Orbit #']
        if pd.notna(orbit):
            orbit = int(orbit)
            orbit_intervals[orbit] = {}
            if 'glare_initial' in row and 'glare_final' in row:
                glare_intervals[orbit] = (row['glare_initial'], row['glare_final']) if pd.notna(row['glare_initial']) and pd.notna(row['glare_final']) else None
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
                            orbit_intervals[orbit][box].append((int(start), int(end)))
    return orbit_intervals, glare_intervals

# Find the corresponding NetCDF file
def find_nc_file(parent_directory, orbit_number):
    orbit_str = str(int(orbit_number)).zfill(5)
    pattern = re.compile(r'awe_l0c_q20_(.*)_' + orbit_str + r'_(.*)\.nc')
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No file found for orbit {orbit_str}")

# Add MLSP to NetCDF
def add_mlsp_to_nc(input_path, output_path, mlsp_data):
    with Dataset(input_path, 'r') as src, Dataset(output_path, 'w', format=src.file_format) as dst:
        dst.setncatts({attr: src.getncattr(attr) for attr in src.ncattrs()})
        for name, dimension in src.dimensions.items():
            dst.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
        dst.createDimension('y_box_across_track', 6)
        dst.createDimension('x_box_along_track', 6)
        for name, variable in src.variables.items():
            var = dst.createVariable(name, variable.datatype, variable.dimensions, zlib=True, complevel=4)
            var[:] = variable[:]
            var.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})
        mlsp_var = dst.createVariable('MLSPB', 'f4', ('time', 'y_box_across_track', 'x_box_along_track'), zlib=True, complevel=4)
        mlsp_var[:] = mlsp_data
        print(f"Saved MLSPB to {output_path}")

# Main
def main():
    data = pd.read_csv(csv_file_path)
    orbit_intervals, glare_intervals = extract_intervals(data)

    extra_intervals = {
        # 113: {"(3,4)": [(1258, 1817)]},
        2450: {"(5,3)": [(1873, 1911)]},
        2480: {"(5,3)": [(1840, 1863)]},
        4700: {"(5,3)": [(1827, 1892)]},
        4730: {"(5,3)": [(1873, 1931)]}, 
        6292: {"(5,3)": [(1867, 1909)]},
        6320: {"(5,3)": [(1862, 1902)]},
        6350: {"(5,3)": [(1835, 1870)]},
        6380: {"(5,3)": [(1838, 1859)]},
        6410: {"(5,3)": [(1820, 1832)]},
        6440: {"(5,3)": [(1838, 1862)]},
        6470: {"(5,3)": [(1852, 1878)]},
        6500: {"(5,3)": [(1838, 1892)]},
        6560: {"(5,3)": [(1900, 1930)]}
    }
    
    for orbit, boxes in extra_intervals.items():
        if orbit in orbit_intervals:
            for box, intervals in boxes.items():
                if box in orbit_intervals[orbit]:
                    orbit_intervals[orbit][box].extend(intervals)
                else:
                    orbit_intervals[orbit][box] = intervals
        else:
            orbit_intervals[orbit] = boxes

    for orbit, boxes in orbit_intervals.items():
        print(f"Processing orbit {orbit}")
        print("Intervals:")
        for box, intervals in boxes.items():
            print(f"  Box {box}: {intervals}")
            
        try:
            nc_file = find_nc_file(parent_directory, orbit)
        except FileNotFoundError as e:
            print(e)
            continue
        
        with Dataset(nc_file, 'r') as nc:
            num_frames = nc.variables['Radiance'].shape[0]
        
        mlsp = np.zeros((num_frames, 6, 6), dtype=np.float32)
        glare = glare_intervals.get(orbit)

        for box, intervals in boxes.items():
            i, j = map(int, box.strip("()").split(","))
            for start, end in intervals:
                if start is None or end is None:
                    continue
                for t in range(start, end+1):
                    if 0 <= t < num_frames:
                        if glare and glare[0] <= t <= glare[1]:
                            continue  # skip glare
                        mlsp[t, j, i] = 1  # note: y_box_across_track is j, x_box_along_track is i

        output_path = os.path.join(output_directory, os.path.basename(nc_file))
        add_mlsp_to_nc(nc_file, output_path, mlsp)

if __name__ == "__main__":
    main()
