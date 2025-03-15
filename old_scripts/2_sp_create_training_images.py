# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:51:37 2024

@author: Anh

Update Dec 2 2024: Combines interval extraction with radiance image generation.
"""

import pandas as pd
import os
from netCDF4 import Dataset
import cv2
import numpy as np
import random
import re
import shutil

# Number of frames before and after for consecutive image combination
space = 5

# Path to the CSV file with filenames and intervals
csv_file_path = 'csv/solar_panels_6x6_mar_10_2025.csv'
# parent_directory = 'l0c'
parent_directory = r'Z:\soc\l0c'


# Define output folders
sp_folder = 'training_images/sp'
no_sp_folder = 'training_images/no_sp'

# Ensure output folders exist
os.makedirs(no_sp_folder, exist_ok=True)
os.makedirs(sp_folder, exist_ok=True)

# Function to clear all files in a given folder
def clear_images(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory and its contents
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Clear both folders before saving new images
clear_images(sp_folder)
clear_images(no_sp_folder)

# Function to define grid boxes
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
#print(grid_boxes)

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

# Function to search for .nc file in all subdirectories
def find_nc_file(parent_directory, orbit_number):
    orbit_str = str(int(orbit_number)).zfill(5)
    pattern = re.compile(r'awe_l0c_q20_(.*)_' + orbit_str + r'_(.*)\.nc')
    
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No file found for orbit number {orbit_str} in {parent_directory}")


# Function to save three-layer images
def save_image(data, folder, orbit_number, frame_index, box_idx, boxes):
    min_radiance, max_radiance = 0, 24
    norm_radiance = np.clip((data[frame_index] - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255).astype(np.uint8)
    prev_frame_norm = None
    next_frame_norm = None
    
    if frame_index >= space:
        prev_frame = data[frame_index - space]
        prev_frame_norm = np.clip((prev_frame - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255).astype(np.uint8)
    if frame_index < data.shape[0] - space:
        next_frame = data[frame_index + space]
        next_frame_norm = np.clip((next_frame - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255).astype(np.uint8)
    
    x_start, x_end = boxes[box_idx]['x']  
    y_start, y_end = boxes[box_idx]['y'] 
    
    three_layer_image = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
    
    if prev_frame_norm is not None:
        three_layer_image[..., 0] = prev_frame_norm
    three_layer_image[..., 1] = norm_radiance
    if next_frame_norm is not None:
        three_layer_image[..., 2] = next_frame_norm
    
    cropped_image = three_layer_image[y_start:y_end+1, x_start:x_end+1]
    file_path = os.path.join(folder, f"orbit{orbit_number}_box{box_idx}_{frame_index}.png")
    cv2.imwrite(file_path, cropped_image)

# Main function to process intervals and save images
def process_intervals_and_save_images(data, grid_boxes):
    sp_threshold = 4  # Frames inside the intervals, 4 frames away from the boundary is considered sp
    no_sp_threshold = 8  # Frames outside the intervals, 8 frames away from the boundary is considered no_sp  # Number of images away from the boundary between sp and no sp
    sp_chance = 0.5
    no_sp_chance = 0.10  # Only save 1/10 of no sp images
    orbit_intervals = extract_intervals_per_orbit(data)

    # Add extra intervals manually
    extra_intervals = {
        # 113: {"(3,4)": [(1258, 1817)]},
        2450: {"(5,3)": [(1873, 1911)]},
        2480: {"(5,3)": [(1840, 1863)]}, 
        4700: {"(5,3)": [(1827, 1892)]}, 
        4730: {"(5,3)": [(1873, 1931)]}
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
    
    for orbit_number, boxes in orbit_intervals.items():
        print(f"Processing orbit: {orbit_number}")
        print(f"Intervals: {boxes}")
        
        try:
            nc_file_path = find_nc_file(parent_directory, orbit_number)
        except FileNotFoundError as e:
            print(e)
            continue
        
        with Dataset(nc_file_path, 'r') as nc:
            radiance = nc.variables['Radiance'][:]
            num_frames = radiance.shape[0]
            
            for box, intervals in boxes.items():
                for i in range(space, num_frames - space):
                    for interval in intervals:
                        if interval[0] + sp_threshold <= i <= interval[1] - sp_threshold:
                            if random.random() < sp_chance:
                                save_image(radiance, sp_folder, orbit_number, i, box, grid_boxes)
                        elif all(i < interval[0] - no_sp_threshold or i > interval[1] + no_sp_threshold for interval in intervals):
                            if random.random() < no_sp_chance:
                                # print(f"Saving No-SP image: Orbit {orbit_number}, Box {box}, Frame {i}")
                                save_image(radiance, no_sp_folder, orbit_number, i, box, grid_boxes)

# Load data
data = pd.read_csv(csv_file_path)
process_intervals_and_save_images(data, grid_boxes)
