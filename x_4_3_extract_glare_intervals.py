# -*- coding: utf-8 -*-
"""
Created on Mar 5 2024

@author: Anh

Script to extract glare intervals from CSV files for each orbit.
An interval is defined as a sequence of at least 4 continuous frames where probability > 0.8,
with 3 extra frames added before and after.
Since there's no prediction for frames 0-2 as well as the last three frames,
frame 3 and the last frame in the CSV are considered as 4 frames.
"""

import os
import pandas as pd

# Path to the folder containing prediction CSV files
csv_folder = 'glare_orbit_predictions/csv'
# Path to save the intervals CSV file
output_csv_file = 'csv/glare_orbit_intervals_mar_16.csv'

# Threshold for glare probability
glare_threshold = 0.01
# Minimum consecutive frames required to form an interval
min_consecutive_frames = 4
# Extra frames to include before and after each interval
extra_frames = 4

# Function to find glare intervals in an orbit
def find_glare_intervals(df):
    glare_frames = df[df['RunningAverageProbability'] > glare_threshold]['Frame'].to_list()
    if not glare_frames:
        return []  # No glare detected
    
    # Identify continuous intervals with at least `min_consecutive_frames`
    intervals = []
    start = glare_frames[0]
    count = 1
    
    for i in range(1, len(glare_frames)):
        if glare_frames[i] == glare_frames[i - 1] + 1:
            count += 1
        else:
            if count >= min_consecutive_frames or start == 3 or glare_frames[i - 1] == df['Frame'].max():
                intervals.append((start, glare_frames[i - 1]))
            start = glare_frames[i]
            count = 1
    
    # Add the last interval if it meets the condition
    if count >= min_consecutive_frames or start == 3 or glare_frames[-1] == df['Frame'].max():
        intervals.append((start, glare_frames[-1]))
    
    # Expand intervals by adding extra frames
    expanded_intervals = []
    for start, end in intervals:
        expanded_intervals.append((max(0, start - extra_frames), end + extra_frames))
    
    return expanded_intervals

# List to store all interval results
interval_results = []

# Process each CSV file in the folder
for csv_file in sorted(os.listdir(csv_folder)):
    if csv_file.endswith('.csv'):
        orbit_number = csv_file.split('_')[1]
        csv_path = os.path.join(csv_folder, csv_file)
        
        # Load CSV file
        df = pd.read_csv(csv_path)
        
        # Find glare intervals
        intervals = find_glare_intervals(df)
        
        # Store results
        for start, end in intervals:
            interval_results.append({"Orbit": orbit_number, "Start_Frame": start, "End_Frame": end})
        
        # Print results
        print(f"Orbit {orbit_number}:")
        if intervals:
            for start, end in intervals:
                print(f"  Interval: {start} - {end}")
        else:
            print("  No glare detected.")
        print("-")

# Save intervals to a CSV file
interval_df = pd.DataFrame(interval_results)
interval_df.to_csv(output_csv_file, index=False)

print(f"Glare intervals saved to {output_csv_file}")
