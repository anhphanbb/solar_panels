# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:43:39 2025

@author: domin
"""

import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# Define input and output folders
csv_predictions_folder = 'sp_orbit_predictions/csv'
nc_output_folder = 'sp_orbit_predictions/mlsp'

# Ensure the output folder exists
os.makedirs(nc_output_folder, exist_ok=True)

# Define the box-to-index mapping
box_mapping = {
    f"({x},{y})": (x, y) for x in range(6) for y in range(6)
}

# Main script to process all CSV files
for file_name in os.listdir(csv_predictions_folder):
    if file_name.endswith('_predictions.csv'):
        orbit_number = file_name.split('_')[1]  # Extract orbit number from filename
        csv_file_path = os.path.join(csv_predictions_folder, file_name)
        output_file_path = os.path.join(nc_output_folder, f'orbit_{orbit_number}_mlsp.nc')
        
        # Read CSV
        predictions_df = pd.read_csv(csv_file_path)
        predictions_df = predictions_df.sort_values(by=['Frame', 'Box'])
        
        # Determine number of frames
        # Determine number of frames correctly
        time_dim = predictions_df['Frame'].max() + 1  # Ensure the array size matches the highest frame number

        
        # Initialize MLSP array
        mlsp_data = np.zeros((time_dim, 6, 6))  # Dimensions: (time, y_box_across_track, x_box_along_track)
        
        # Populate MLSP
        for _, row in predictions_df.iterrows():
            frame = int(row['Frame'])
            box = row['Box']
            probability = row['Probability']
            
            if box in box_mapping:
                x_idx, y_idx = box_mapping[box]
                mlsp_data[frame, y_idx, x_idx] = probability
            else:
                print(f"Box {box} not found in mapping.")
        
        # Handle first and last four frames
        if time_dim > 5:
            mlsp_data[:5, :, :] = mlsp_data[5, :, :]
            mlsp_data[-5:, :, :] = mlsp_data[-6, :, :]
        
        # Create new NetCDF file with only the MLSP variable
        with Dataset(output_file_path, 'w', format='NETCDF4') as dst_nc:
            dst_nc.createDimension('time', time_dim)
            dst_nc.createDimension('y_box_across_track', 6)
            dst_nc.createDimension('x_box_along_track', 6)
            
            mlsp_var = dst_nc.createVariable('MLSP', 'f4', ('time', 'y_box_across_track', 'x_box_along_track'), zlib=True)
            mlsp_var[:] = mlsp_data
            
            print(f"Created new NetCDF file: {output_file_path}")

print("Processing completed.")
