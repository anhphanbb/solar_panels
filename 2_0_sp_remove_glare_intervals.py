import os
import re
import pandas as pd
import numpy as np
from netCDF4 import Dataset

# Path to the CSV file with glare intervals
csv_file_path = 'csv/glare_orbit_intervals.csv'
# parent_directory = 'sp_selected_orbits'
parent_directory = r'E:\soc\l0c\2024\09'

output_directory = r'E:\soc\l0c\2024\09\no_glare'

# Ensure output folder exists
os.makedirs(output_directory, exist_ok=True)

# Function to extract glare intervals from CSV data
def extract_glare_intervals(data):
    glare_intervals = {}
    for _, row in data.iterrows():
        orbit = row['Orbit #']
        if pd.notna(orbit):
            orbit = int(orbit)
            if 'glare_initial' in row and 'glare_final' in row:
                if pd.notna(row['glare_initial']) and pd.notna(row['glare_final']):
                    glare_intervals[orbit] = (int(row['glare_initial']), int(row['glare_final']))
    return glare_intervals

# Function to remove glare frames and save a new NetCDF file
def remove_glare_and_save(nc_file_path, glare_intervals, output_directory):
    with Dataset(nc_file_path, 'r') as nc:
        # Read dimensions and variables
        dimensions = {dim: nc.dimensions[dim].size for dim in nc.dimensions}
        variables = {var: nc.variables[var] for var in nc.variables}
        
        # Extract orbit number from filename
        match = re.search(r'_(\d{5})_', os.path.basename(nc_file_path))
        if not match:
            print(f"Skipping file (orbit not found in name): {nc_file_path}")
            return
        orbit_number = int(match.group(1))
        
        # Get glare range for this orbit
        glare_range = glare_intervals.get(orbit_number, None)
        if glare_range is None:
            print(f"No glare data for orbit {orbit_number}, skipping.")
            return
        
        # Determine frames to keep
        total_frames = dimensions['time']
        keep_indices = [i for i in range(total_frames) if i < glare_range[0] or i > glare_range[1]]
        
        # Create new NetCDF file
        output_file_path = os.path.join(output_directory, os.path.basename(nc_file_path))
        with Dataset(output_file_path, 'w', format='NETCDF4') as new_nc:
            
            # Copy dimensions
            for dim, size in dimensions.items():
                if dim == 'time':
                    new_nc.createDimension(dim, len(keep_indices))  # Adjust frame count
                else:
                    new_nc.createDimension(dim, size)
            
            # Copy variables
            for var_name, var in variables.items():
                new_var = new_nc.createVariable(var_name, var.datatype, var.dimensions,
                                                zlib=var.filters().get('zlib', False),
                                                chunksizes=var.chunking() if var.chunking() != 'contiguous' else None)
                new_var.setncatts({attr: var.getncattr(attr) for attr in var.ncattrs()})
                
                # Copy data, excluding glare frames for time-dependent variables
                if 'time' in var.dimensions:
                    new_var[:] = var[keep_indices]
                else:
                    new_var[:] = var[:]
    
            print(f"Processed: {output_file_path}")

# Load data
data = pd.read_csv(csv_file_path)
glare_intervals = extract_glare_intervals(data)

# Process all NetCDF files in the folder
for file in os.listdir(parent_directory):
    if file.endswith(".nc") and 'q20' in file:
        nc_file_path = os.path.join(parent_directory, file)
        remove_glare_and_save(nc_file_path, glare_intervals, output_directory)