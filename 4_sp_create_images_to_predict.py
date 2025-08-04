# -*- coding: utf-8 -*-
"""
Modified on Wed Dec  4 2024

@author: Anh

Script to create prediction images using consecutive frames from all .nc files in a folder using multiprocessing.
"""

import os
from netCDF4 import Dataset
import cv2
import numpy as np
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# Input folder containing .nc files
nc_folder = r'E:\soc\l0c\2025\06'

# Output folder to save prediction images
output_folder = r'E:\soc\l0c\2025\06\sp_images_to_predict'

# Number of frames before and after for consecutive image combination
space = 5

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

# Function to create images for prediction with consecutive frame combination
def create_images_from_nc_file(nc_file_path):
    orbit_match = re.search(r'_(\d{5})_', nc_file_path)
    if orbit_match:
        orbit_number = orbit_match.group(1)
    else:
        print(f"Could not determine orbit number from file: {nc_file_path}")
        return

    orbit_output_folder = os.path.join(output_folder, f"orbit_{orbit_number}")
    os.makedirs(orbit_output_folder, exist_ok=True)

    with Dataset(nc_file_path, 'r') as nc:
        radiance = nc.variables['Radiance'][:]
        num_frames = radiance.shape[0]

        for i in range(space, num_frames - space):
            norm_radiance = normalize_radiance(radiance[i])
            prev_frame_norm = normalize_radiance(radiance[i - space])
            next_frame_norm = normalize_radiance(radiance[i + space])

            three_layer_image = np.zeros((radiance.shape[1], radiance.shape[2], 3), dtype=np.uint8)
            three_layer_image[..., 0] = prev_frame_norm
            three_layer_image[..., 1] = norm_radiance
            three_layer_image[..., 2] = next_frame_norm

            for box, coords in grid_boxes.items():
                if box not in ignored_boxes:
                    x_start, x_end = coords['x']
                    y_start, y_end = coords['y']
                    cropped_image = three_layer_image[y_start:y_end+1, x_start:x_end+1]
                    file_path = os.path.join(orbit_output_folder, f"frame_{i}_box_{box}.png")
                    cv2.imwrite(file_path, cropped_image)

# Get all .nc file paths to process
def get_nc_files(nc_folder):
    nc_files = []
    for root, _, files in os.walk(nc_folder):
        for file in files:
            if file.endswith('.nc') and 'q20' in file:
                nc_files.append(os.path.join(root, file))
    return nc_files

# Process all files in parallel
def process_all_nc_files_parallel(nc_files):
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(create_images_from_nc_file, nc_file): nc_file for nc_file in nc_files}
        for future in as_completed(futures):
            file = futures[future]
            try:
                future.result()
                print(f"Finished processing: {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")

# Run the script
if __name__ == "__main__":
    nc_files = get_nc_files(nc_folder)
    process_all_nc_files_parallel(nc_files)
