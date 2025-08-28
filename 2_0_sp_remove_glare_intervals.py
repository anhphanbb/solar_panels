import os
import re
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from concurrent.futures import ProcessPoolExecutor, as_completed

# Path to the CSV file with glare intervals
csv_file_path = 'csv/glare_orbit_intervals_july_2025_bkg.csv'
# Parent directory with NetCDF files
parent_directory = r'E:\soc\l0c\2025\07'
# Output directory to save files with glare removed
output_directory = r'E:\soc\l0c\2025\07\no_glare'

# Ensure output folder exists
os.makedirs(output_directory, exist_ok=True)

# Function to extract multiple glare intervals per orbit from CSV
def extract_glare_intervals(data):
    glare_intervals = {}
    for _, row in data.iterrows():
        orbit = row['Orbit #']
        if pd.notna(orbit):
            orbit = int(orbit)
            if pd.notna(row['glare_initial']) and pd.notna(row['glare_final']):
                interval = (int(row['glare_initial']), int(row['glare_final']))
                if orbit not in glare_intervals:
                    glare_intervals[orbit] = []
                glare_intervals[orbit].append(interval)
    return glare_intervals

# Function to remove glare frames and save a new NetCDF file
def remove_glare_and_save(nc_file_path, glare_intervals):
    with Dataset(nc_file_path, 'r') as nc:
        dimensions = {dim: nc.dimensions[dim].size for dim in nc.dimensions}
        variables = {var: nc.variables[var] for var in nc.variables}
        
        match = re.search(r'_(\d{5})_', os.path.basename(nc_file_path))
        if not match:
            print(f"Skipping file (orbit not found in name): {nc_file_path}")
            return
        orbit_number = int(match.group(1))
        
        glare_ranges = glare_intervals.get(orbit_number, None)
        if glare_ranges is None:
            print(f"No glare data for orbit {orbit_number}, skipping.")
            return
        
        total_frames = dimensions['time']
        keep_indices = set(range(total_frames))
        for start, end in glare_ranges:
            keep_indices -= set(range(start, end + 1))
        keep_indices = sorted(keep_indices)
        
        output_file_path = os.path.join(output_directory, os.path.basename(nc_file_path))
        with Dataset(output_file_path, 'w', format='NETCDF4') as new_nc:
            new_nc.setncatts({attr: nc.getncattr(attr) for attr in nc.ncattrs()})
            
            for dim, size in dimensions.items():
                if dim == 'time':
                    new_nc.createDimension(dim, len(keep_indices))
                else:
                    new_nc.createDimension(dim, size)
            
            for var_name, var in variables.items():
                # Adjust chunking to avoid exceeding dimensions
                original_chunks = var.chunking() if var.chunking() != 'contiguous' else None
                safe_chunks = None
                if original_chunks:
                    safe_chunks = tuple(
                        min(dim_size if dim != 'time' else len(keep_indices), chunk)
                        for dim, chunk, dim_size in zip(var.dimensions, original_chunks,
                                                        [len(keep_indices) if d == 'time' else dimensions[d] for d in var.dimensions])
                    )
                
                new_var = new_nc.createVariable(
                    var_name,
                    var.datatype,
                    var.dimensions,
                    zlib=var.filters().get('zlib', False),
                    chunksizes=safe_chunks
                )
                new_var.setncatts({attr: var.getncattr(attr) for attr in var.ncattrs()})
                
                if 'time' in var.dimensions:
                    new_var[:] = var[keep_indices]
                else:
                    new_var[:] = var[:]
    
        print(f"Processed: {output_file_path}")

# Wrapper function for multiprocessing
def process_file(args):
    nc_file_path, glare_intervals = args
    try:
        remove_glare_and_save(nc_file_path, glare_intervals)
    except Exception as e:
        print(f"Error processing {nc_file_path}: {e}")

if __name__ == "__main__":
    data = pd.read_csv(csv_file_path)
    glare_intervals = extract_glare_intervals(data)

    nc_files = [
        os.path.join(parent_directory, file)
        for file in os.listdir(parent_directory)
        if file.endswith(".nc")
    ]

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_file, (nc_file, glare_intervals)): nc_file
            for nc_file in nc_files
        }
        for future in as_completed(futures):
            file = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Unhandled error in {file}: {e}")
