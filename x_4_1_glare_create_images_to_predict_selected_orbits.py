# -*- coding: utf-8 -*-
"""
Modified on Wed Dec  4 2024

@author: Anh

Script to create prediction images using consecutive frames from all .nc files in a folder.
"""

import os
from netCDF4 import Dataset
import cv2
import numpy as np
import re

# Input folder containing .nc files
# nc_folder = r'Z:\moc\l0b'
nc_folder = r'E:\soc\l0c\2024\09'

# Output folder to save prediction images
output_folder = 'glare_images_to_predict'

# Number of frames before and after for consecutive image combination
space = 3

# List of specific orbits to process
september_orbits_1 = list(range(4402, 4561, 1)) #4402 to 4560
orbit_list = september_orbits_1

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def find_nc_file(parent_directory, orbit_number):
    orbit_str = str(int(orbit_number)).zfill(5)
    pattern = re.compile(r'awe_l0c_q20_(.*)_' + orbit_str + r'_(.*)\.nc')
    
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No file found for orbit number {orbit_str}")

# Define single large box covering the full image
full_image_box = {'x': (0, 255), 'y': (0, 255)}

# Function to normalize radiance values to 8-bit range
def normalize_radiance(frame, min_radiance=0, max_radiance=24):
    return np.clip((frame - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255).astype(np.uint8)

# Function to create images for prediction with consecutive frame combination
def create_images_from_nc_file(nc_file_path, full_image_box, output_folder, space):
    orbit_number = re.search(r'_(\d{5})_', nc_file_path).group(1)
    orbit_output_folder = os.path.join(output_folder, f"orbit_{orbit_number}")
    os.makedirs(orbit_output_folder, exist_ok=True)

    # Read the .nc file
    with Dataset(nc_file_path, 'r') as nc:
        radiance = nc.variables['Radiance'][:]
        num_frames = radiance.shape[0]

        for i in range(space, num_frames - space):
            # Normalize the radiance frame and consecutive frames
            norm_radiance = normalize_radiance(radiance[i])
            prev_frame_norm = normalize_radiance(radiance[i - space]) if i >= space else None
            next_frame_norm = normalize_radiance(radiance[i + space]) if i < num_frames - space else None

            # Create a three-layer image
            three_layer_image = np.zeros((radiance.shape[1], radiance.shape[2], 3), dtype=np.uint8)
            if prev_frame_norm is not None:
                three_layer_image[..., 0] = prev_frame_norm
            three_layer_image[..., 1] = norm_radiance
            if next_frame_norm is not None:
                three_layer_image[..., 2] = next_frame_norm

            # Use the full image for glare detection
            x_start, x_end = full_image_box['x']
            y_start, y_end = full_image_box['y']

            cropped_image = three_layer_image[y_start:y_end+1, x_start:x_end+1]
            file_path = os.path.join(orbit_output_folder, f"frame_{i}.png")
            cv2.imwrite(file_path, cropped_image)

# Process selected orbits
def process_selected_orbits(parent_directory, output_folder, full_image_box, space, orbit_list):
    for orbit_number in orbit_list:
        try:
            nc_file_path = find_nc_file(parent_directory, orbit_number)
            print(f"Processing orbit {orbit_number} from file: {nc_file_path}")
            create_images_from_nc_file(nc_file_path, full_image_box, output_folder, space)
        except FileNotFoundError as e:
            print(e)

# Run the script
process_selected_orbits(nc_folder, output_folder, full_image_box, space, orbit_list)
