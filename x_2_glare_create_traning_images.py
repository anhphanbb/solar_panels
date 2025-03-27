# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:51:37 2024

@author: Anh

Update Dec 2 2024: Combines interval extraction with radiance image generation for glare detection.
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
space = 3

# Path to the CSV file with filenames and intervals
csv_file_path = 'csv/AweGlareSeptemberLabeled.csv'
# Parent directory containing NetCDF files
parent_directory = r'E:\soc\l0c\2024\09'

# Define output folders
glare_folder = 'glare_training_images/class_1_glare'
no_glare_folder = 'glare_training_images/class_0_no_glare'

# Ensure output folders exist
os.makedirs(glare_folder, exist_ok=True)
os.makedirs(no_glare_folder, exist_ok=True)

# Function to clear all files in a given folder
def clear_images(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Clear both folders before saving new images
clear_images(glare_folder)
clear_images(no_glare_folder)

# Define single large box covering the full image
full_image_box = {'x': (0, 255), 'y': (0, 255)}

def extract_intervals_per_orbit(data):
    orbit_intervals = {}
    for _, row in data.iterrows():
        orbit = row['Orbit']
        if pd.notna(orbit):
            orbit = int(orbit)
            if orbit not in orbit_intervals:
                orbit_intervals[orbit] = []
            start, end = row['start'], row['end']
            if pd.notna(start) and pd.notna(end):
                orbit_intervals[orbit].append((start, end))
    return orbit_intervals

def find_nc_file(parent_directory, orbit_number):
    orbit_str = str(int(orbit_number)).zfill(5)
    pattern = re.compile(r'awe_l0c_q20_(.*)_' + orbit_str + r'_(.*)\.nc')
    
    for root, dirs, files in os.walk(parent_directory):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    raise FileNotFoundError(f"No file found for orbit number {orbit_str} in {parent_directory}")

def save_image(data, folder, orbit_number, frame_index, box):
    min_radiance, max_radiance = 0, 24
    norm_radiance = np.clip((data[frame_index] - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255).astype(np.uint8)
    prev_frame_norm, next_frame_norm = None, None
    
    if frame_index >= space:
        prev_frame = data[frame_index - space]
        prev_frame_norm = np.clip((prev_frame - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255).astype(np.uint8)
    if frame_index < data.shape[0] - space:
        next_frame = data[frame_index + space]
        next_frame_norm = np.clip((next_frame - min_radiance) / (max_radiance - min_radiance) * 255, 0, 255).astype(np.uint8)
    
    x_start, x_end = box['x']  
    y_start, y_end = box['y'] 
    
    three_layer_image = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
    if prev_frame_norm is not None:
        three_layer_image[..., 0] = prev_frame_norm
    three_layer_image[..., 1] = norm_radiance
    if next_frame_norm is not None:
        three_layer_image[..., 2] = next_frame_norm
    
    cropped_image = three_layer_image[y_start:y_end+1, x_start:x_end+1]

    # Resize to 128x128
    resized_image = cv2.resize(cropped_image, (256, 256), interpolation=cv2.INTER_AREA)

    file_path = os.path.join(folder, f"orbit{orbit_number}_frame{frame_index}.png")
    cv2.imwrite(file_path, resized_image)

# def process_intervals_and_save_images(data, full_image_box):
#     glare_threshold = 2
#     no_glare_threshold = 6
#     glare_chance = 0.75
#     no_glare_chance = 0.1
#     orbit_intervals = extract_intervals_per_orbit(data)
    
#     for orbit_number, intervals in orbit_intervals.items():
#         print(f"Processing orbit: {orbit_number}")
#         print(f"Intervals: {intervals}")
        
#         try:
#             nc_file_path = find_nc_file(parent_directory, orbit_number)
#         except FileNotFoundError as e:
#             print(e)
#             continue
        
#         with Dataset(nc_file_path, 'r') as nc:
#             radiance = nc.variables['Radiance'][:]
#             num_frames = radiance.shape[0]
            
#             for i in range(space, num_frames - space):
#                 for interval in intervals:
#                     if interval[0] + glare_threshold <= i <= interval[1] - glare_threshold:
#                         if random.random() < glare_chance:
#                             save_image(radiance, glare_folder, orbit_number, i, full_image_box)
#                     elif all(i < interval[0] - no_glare_threshold or i > interval[1] + no_glare_threshold for interval in intervals):
#                         if random.random() < no_glare_chance:
#                             save_image(radiance, no_glare_folder, orbit_number, i, full_image_box)

# data = pd.read_csv(csv_file_path)
# process_intervals_and_save_images(data, full_image_box)

def process_intervals_and_save_images(data, full_image_box):
    glare_threshold = 3
    no_glare_threshold = 7
    glare_chance = 1
    no_glare_chance = 0.075

    orbit_intervals = extract_intervals_per_orbit(data)  # Get labeled intervals
    all_orbits = sorted(set(data['Orbit'].dropna().astype(int)))  # Get all unique orbits from CSV

    for orbit_number in all_orbits:
        print(f"Processing orbit: {orbit_number}")
        
        try:
            nc_file_path = find_nc_file(parent_directory, orbit_number)
        except FileNotFoundError as e:
            print(e)
            continue
        
        with Dataset(nc_file_path, 'r') as nc:
            radiance = nc.variables['Radiance'][:]
            num_frames = radiance.shape[0]

            intervals = orbit_intervals.get(orbit_number, [])  # Get intervals, empty list if none
            print(intervals)
            
            for i in range(space, num_frames - space):
                if any(interval[0] + glare_threshold <= i <= interval[1] - glare_threshold for interval in intervals):
                    if random.random() < glare_chance:
                        save_image(radiance, glare_folder, orbit_number, i, full_image_box)
                elif all(i < interval[0] - no_glare_threshold or i > interval[1] + no_glare_threshold for interval in intervals):
                    if random.random() < no_glare_chance:
                        save_image(radiance, no_glare_folder, orbit_number, i, full_image_box)

data = pd.read_csv(csv_file_path)
process_intervals_and_save_images(data, full_image_box)
