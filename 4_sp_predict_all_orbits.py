# -*- coding: utf-8 -*-
"""
Updated on Tue Dec  3 20:10:39 2024

@author: anhph
"""

import os
import cv2
import numpy as np
import pandas as pd
import re
import time
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import backend as K
import gc
from concurrent.futures import ThreadPoolExecutor

# Paths
input_folder = r'E:\soc\l0c\2025\06\sp_images_to_predict'
csv_output_folder = 'sp_orbit_predictions/csv'
model_path = 'models/tf_model_py310_sp_acc_and_recall_july_27_soc2_2025.h5'

# Ensure the output folder for CSV files exists
os.makedirs(csv_output_folder, exist_ok=True)

# These three lines are required to be here in order for GPU to properly set up. 
print("CUDA version:", tf.sysconfig.get_build_info().get("cuda_version", "Not Found")) 
print("cuDNN version:", tf.sysconfig.get_build_info().get("cudnn_version", "Not Found")) 
print("GPU detected:", tf.config.list_physical_devices('GPU')) 

# Initial model load (will be replaced after each orbit)
model = load_model(model_path)

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Parallel image loader
def load_image(path):
    return cv2.resize(cv2.imread(path), (43, 43))

def preprocess_images_batch(image_paths):
    with ThreadPoolExecutor(max_workers=8) as executor:
        images = list(executor.map(load_image, image_paths))
    return preprocess_input(np.array(images))

# Running average
def compute_running_average(predictions, window_size):
    return np.convolve(predictions, np.ones(window_size) / window_size, mode='valid')

# Optional: remove orbit folder after processing
def remove_orbit_images(orbit_folder):
    try:
        shutil.rmtree(orbit_folder)
        print(f'Successfully deleted {orbit_folder}')
    except Exception as e:
        print(f'Failed to delete {orbit_folder}. Reason: {e}')

# Main function per orbit
def process_orbit_folder(orbit_folder, orbit_number, batch_size=32, avg_window_size=9):
    global model  # so we can reassign

    start_time = time.time()
    results = []
    images = os.listdir(orbit_folder)

    # Parse image info
    image_data = []
    for image_name in images:
        match = re.match(r"frame_(\d+)_box_\((\d+),(\d+)\)\.png", image_name)
        if match:
            frame_number = int(match.group(1))
            box = f"({match.group(2)},{match.group(3)})"
            image_path = os.path.join(orbit_folder, image_name)
            image_data.append((frame_number, box, image_path))

    image_data.sort(key=lambda x: x[0])
    total_images = len(image_data)

    for start_idx in range(0, total_images, batch_size):
        batch_data = image_data[start_idx:start_idx + batch_size]
        batch_frame_numbers = [item[0] for item in batch_data]
        batch_boxes = [item[1] for item in batch_data]
        batch_image_paths = [item[2] for item in batch_data]

        batch_images = preprocess_images_batch(batch_image_paths)
        probabilities = model(batch_images, training=False).numpy().flatten()

        for frame_number, box, probability in zip(batch_frame_numbers, batch_boxes, probabilities):
            results.append({"Frame": frame_number, "Box": box, "Probability": probability})

        print(f"Processed {start_idx + len(batch_data)}/{total_images} images in Orbit {orbit_number}...")

    # Save CSV
    df_results = pd.DataFrame(results)
    running_averages = []
    for box, group in df_results.groupby("Box"):
        probabilities = group["Probability"].to_list()
        frames = group["Frame"].to_list()
        averaged_probs = compute_running_average(probabilities, avg_window_size)
        for i, avg_prob in enumerate(averaged_probs):
            running_averages.append({
                "Frame": frames[i + avg_window_size // 2],
                "Box": box,
                "Probability": probabilities[i + avg_window_size // 2],
                "RunningAverageProbability": avg_prob
            })

    output_csv_path = os.path.join(csv_output_folder, f"orbit_{orbit_number}_predictions.csv")
    pd.DataFrame(running_averages).to_csv(output_csv_path, index=False)

    end_time = time.time()
    print(f"Completed processing for Orbit {orbit_number}. Time taken: {end_time - start_time:.2f} seconds")
    print(f"Results saved to {output_csv_path}")

    # Optional: remove folder
    # remove_orbit_images(orbit_folder)

    # Cleanup memory
    del df_results, running_averages, results, image_data, batch_images
    gc.collect()

    # Reset TF session and reload model
    K.clear_session()
    model = load_model(model_path)

# Loop over orbits
def process_all_orbits(input_folder, batch_size=32, avg_window_size=9):
    for orbit_folder_name in os.listdir(input_folder):
        orbit_folder_path = os.path.join(input_folder, orbit_folder_name)
        if os.path.isdir(orbit_folder_path):
            orbit_match = re.search(r'orbit_(\d+)', orbit_folder_name)
            if orbit_match:
                orbit_number = orbit_match.group(1)
                print(f"Processing orbit folder: {orbit_folder_name}")
                process_orbit_folder(orbit_folder_path, orbit_number, batch_size=batch_size, avg_window_size=avg_window_size)

# Run
process_all_orbits(input_folder, batch_size=1024, avg_window_size=1)
