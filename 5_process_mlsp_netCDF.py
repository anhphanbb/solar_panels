# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:44:44 2024

@author: anhph
"""

import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import re

# Define input and output folders
# nc_input_folder = r'E:\soc\l0c\2024\09'
nc_input_folder = r'E:\soc\l0c\2024\09\no_glare'

csv_predictions_folder = 'sp_orbit_predictions/csv'

nc_output_folder = r'E:\soc\l0c\2024\09\no_glare\nc_files_with_mlsp'
# nc_output_folder = r'E:\soc\l0c\2024\09\nc_files_with_mlsp'

# Ensure the output folder exists
os.makedirs(nc_output_folder, exist_ok=True)

# Define the box-to-index mapping
box_mapping = {
    "(0,0)": (0, 0), "(0,1)": (0, 1), "(0,2)": (0, 2), "(0,3)": (0, 3), "(0,4)": (0, 4), "(0,5)": (0, 5),
    "(1,0)": (1, 0), "(1,1)": (1, 1), "(1,2)": (1, 2), "(1,3)": (1, 3), "(1,4)": (1, 4), "(1,5)": (1, 5),
    "(2,0)": (2, 0), "(2,1)": (2, 1), "(2,2)": (2, 2), "(2,3)": (2, 3), "(2,4)": (2, 4), "(2,5)": (2, 5),
    "(3,0)": (3, 0), "(3,1)": (3, 1), "(3,2)": (3, 2), "(3,3)": (3, 3), "(3,4)": (3, 4), "(3,5)": (3, 5),
    "(4,0)": (4, 0), "(4,1)": (4, 1), "(4,2)": (4, 2), "(4,3)": (4, 3), "(4,4)": (4, 4), "(4,5)": (4, 5),
    "(5,0)": (5, 0), "(5,1)": (5, 1), "(5,2)": (5, 2), "(5,3)": (5, 3), "(5,4)": (5, 4), "(5,5)": (5, 5)
}

# box_mapping = {
#     f"({x},{y})": (x, y) for y in range(5) for x in range(5)
# }

# Function to extract orbit number from filename
def extract_orbit_number(filename):
    return filename.split('_')[4]

# Function to create a new NetCDF file with the MLSP variable
def add_mlsp_to_nc_file(input_file_path, output_file_path, mlsp_data):
    with Dataset(input_file_path, 'r') as src_nc, Dataset(output_file_path, 'w', format=src_nc.file_format) as dst_nc:
        # Copy global attributes
        dst_nc.setncatts({attr: src_nc.getncattr(attr) for attr in src_nc.ncattrs()})
        
        # Copy dimensions
        for name, dimension in src_nc.dimensions.items():
            dst_nc.createDimension(name, (len(dimension) if not dimension.isunlimited() else None))
        
        # Add new dimensions for MLSP
        dst_nc.createDimension('y_box_across_track', 6)
        dst_nc.createDimension('x_box_along_track', 6)
        
        # Copy variables
        for name, variable in src_nc.variables.items():
            new_var = dst_nc.createVariable(name, variable.datatype, variable.dimensions)
            new_var[:] = variable[:]
            new_var.setncatts({attr: variable.getncattr(attr) for attr in variable.ncattrs()})
        
        # Add the MLSP variable
        mlsp_var = dst_nc.createVariable('MLSP', 'f4', ('time', 'y_box_across_track', 'x_box_along_track'), zlib=True)
        
        # Write MLSP data
        mlsp_var[:] = mlsp_data
        
        print(f"Created file with MLSP variable: {output_file_path}")

# Main script to process all files
for file_name in os.listdir(nc_input_folder):
    if file_name.endswith('.nc') and 'q20' in file_name:
        print(file_name)
        orbit_number = extract_orbit_number(file_name)
        print(orbit_number)
        if orbit_number:
            nc_file_path = os.path.join(nc_input_folder, file_name)
            csv_file_path = os.path.join(csv_predictions_folder, f"orbit_{orbit_number}_predictions.csv")
            if os.path.exists(csv_file_path):
                output_file_path = os.path.join(nc_output_folder, file_name)
                
                # Read CSV
                predictions_df = pd.read_csv(csv_file_path)
                predictions_df = predictions_df.sort_values(by=['Frame', 'Box'])
                
                # Open the NetCDF file to get the time dimension
                with Dataset(nc_file_path, 'r') as nc_file:
                    time_dim = len(nc_file.dimensions['time'])  # Extract the number of frames from NetCDF
                
                # Initialize MLSP array
                mlsp_data = np.zeros((time_dim, 6, 6))  # Dimensions: (time, y_box_across_track, x_box_along_track)
                
                # Populate MLSP
                for _, row in predictions_df.iterrows():
                    frame = int(row['Frame'])
                    box = row['Box']
                    # probability = row['RunningAverageProbability']
                    probability = row['Probability']
                    
                    if box in box_mapping:
                        x_idx, y_idx = box_mapping[box]
                        mlsp_data[frame, y_idx, x_idx] = probability
                    else:
                        print(f"Box {box} not found in mapping.")
                        
                # Handle first and last four frames (Because of combining frames)
                if time_dim > 5:
                    mlsp_data[:5, :, :] = mlsp_data[5, :, :]  # Fill first 4 frames with the 5th frame
                    mlsp_data[-5:, :, :] = mlsp_data[-6, :, :]  # Fill last 4 frames with the 5th-to-last frame
                
                
                # Write to NetCDF
                add_mlsp_to_nc_file(nc_file_path, output_file_path, mlsp_data)
            else:
                print(f"CSV file for orbit {orbit_number} not found in {csv_predictions_folder}")

print("Processing completed.")
