# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:23:38 2024

@author: anhph
"""

import os
import cv2
import numpy as np
import pandas as pd
import re
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Path to the folder containing orbit subfolders with images
input_folder = 'images_to_predict'

# Output folder for CSV results
csv_output_folder = 'orbit_predictions'

# Path to the pre-trained model
model_path = 'models/DeepLearning_resnet_model_sp.h5'

# Ensure the output folder for CSV files exists
os.makedirs(csv_output_folder, exist_ok=True)

# Load the pre-trained model
model = load_model(model_path)

# Function to preprocess images for the model
def preprocess_image(image):
    """
    Preprocess the input image to match the model's expected input.
    Resize to (43, 43) and apply ResNet-50 preprocessing.
    """
    image_resized = cv2.resize(image, (43, 43))  # Resize to match training size
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)  # Convert to RGB
    image_preprocessed = preprocess_input(image_rgb)  # Apply ResNet-50 preprocessing
    return np.expand_dims(image_preprocessed, axis=0)  # Add batch dimension

# Function to predict probabilities for an image using the model
def predict_image(image):
    """
    Predict the probability for the input image using the loaded model.
    """
    preprocessed = preprocess_image(image)
    probability = model.predict(preprocessed, verbose=0)[0][0]  # Binary classification output
    return probability

# Function to compute running averages
def compute_running_average(predictions, window_size=9):
    """
    Compute the running average over a given window size.
    """
    return np.convolve(predictions, np.ones(window_size) / window_size, mode='valid')

# Function to process a single orbit folder
def process_orbit_folder(orbit_folder, orbit_number):
    start_time = time.time()
    results = []  # To store results for the current orbit
    images = os.listdir(orbit_folder)

    # Extract frame numbers and boxes from filenames
    image_data = []
    for image_name in images:
        match = re.match(r"frame_(\d+)_box_\((\d+),(\d+)\)\.png", image_name)
        if match:
            frame_number = int(match.group(1))
            box = f"({match.group(2)},{match.group(3)})"
            image_data.append((frame_number, box, image_name))

    # Sort images by frame number for proper averaging
    image_data.sort(key=lambda x: x[0])

    # Store predictions and metadata
    total_frames = len(image_data)
    for idx, (frame_number, box, image_name) in enumerate(image_data, 1):
        image_path = os.path.join(orbit_folder, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Make prediction for the image
        probability = predict_image(image)
        results.append({"Frame": frame_number, "Box": box, "Probability": probability})

        # Show progress
        print(f"Processing Frame {frame_number} ({idx}/{total_frames}) in Orbit {orbit_number}...")

    # Compute running averages for each box
    df_results = pd.DataFrame(results)
    running_averages = []
    for box, group in df_results.groupby("Box"):
        probabilities = group["Probability"].to_list()
        frames = group["Frame"].to_list()
        averaged_probs = compute_running_average(probabilities)
        
        # Add running averages to the results
        for i, avg_prob in enumerate(averaged_probs):
            running_averages.append({
                "Frame": frames[i + 4],  # Center frame of the 9-frame window
                "Box": box,
                "RunningAverageProbability": avg_prob
            })

    # Save the results to a CSV file for the orbit
    output_csv_path = os.path.join(csv_output_folder, f"orbit_{orbit_number}_predictions.csv")
    pd.DataFrame(running_averages).to_csv(output_csv_path, index=False)
    
    # Print total time taken
    end_time = time.time()
    print(f"Completed processing for Orbit {orbit_number}. Time taken: {end_time - start_time:.2f} seconds")
    print(f"Results saved to {output_csv_path}")

# Main script to process all orbit subfolders
def process_all_orbits(input_folder):
    for orbit_folder_name in os.listdir(input_folder):
        orbit_folder_path = os.path.join(input_folder, orbit_folder_name)
        if os.path.isdir(orbit_folder_path):
            # Extract orbit number from folder name
            orbit_match = re.search(r'orbit_(\d+)', orbit_folder_name)
            if orbit_match:
                orbit_number = orbit_match.group(1)
                print(f"Processing orbit folder: {orbit_folder_name}")
                process_orbit_folder(orbit_folder_path, orbit_number)

# Run the script
process_all_orbits(input_folder)
