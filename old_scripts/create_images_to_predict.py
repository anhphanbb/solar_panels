# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:45:37 2024

@author: Anh

Script to create prediction images from all .nc files in a folder.
"""

import os
from netCDF4 import Dataset
import cv2
import numpy as np
import re

# Input folder containing .nc files
nc_folder = 'nc_files_to_predict'

# Output folder to save prediction images
output_folder = 'images_to_predict'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define grid boxes
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

ignored_boxes = ["(0,0)", "(0,5)", "(5,0)", "(5,5)"]

grid_boxes = define_boxes()

# Function to normalize radiance values to 8-bit range
def normalize_radiance(frame, min_radiance=0, max_radiance=24):
    return np.clip((frame - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255).astype(np.uint8)

# Function to create images for prediction
def create_images_from_nc_file(nc_file_path, grid_boxes, output_folder):
    orbit_match = re.search(r'_(\d{5})_', nc_file_path)
    if orbit_match:
        orbit_number = orbit_match.group(1)
    else:
        print(f"Could not determine orbit number from file: {nc_file_path}")
        return

    # Create a folder for the orbit
    orbit_output_folder = os.path.join(output_folder, f"orbit_{orbit_number}")
    os.makedirs(orbit_output_folder, exist_ok=True)

    # Read the .nc file
    with Dataset(nc_file_path, 'r') as nc:
        radiance = nc.variables['Radiance'][:]
        num_frames = radiance.shape[0]

        for i in range(num_frames):
            # Normalize the radiance frame
            norm_radiance = normalize_radiance(radiance[i])

            # Save an image for each grid box
            for box, coords in grid_boxes.items():
                if box not in ignored_boxes:
                    # print(box)
                    x_start, x_end = coords['x']
                    y_start, y_end = coords['y']
    
                    cropped_image = norm_radiance[y_start:y_end, x_start:x_end]
                    file_path = os.path.join(orbit_output_folder, f"frame_{i}_box_{box}.png")
                    cv2.imwrite(file_path, cropped_image)

# Process all .nc files in the folder
def process_all_nc_files(nc_folder, output_folder, grid_boxes):
    for root, _, files in os.walk(nc_folder):
        for file in files:
            if file.endswith('.nc'):
                nc_file_path = os.path.join(root, file)
                print(f"Processing file: {nc_file_path}")
                create_images_from_nc_file(nc_file_path, grid_boxes, output_folder)

# Run the script
process_all_nc_files(nc_folder, output_folder, grid_boxes)
