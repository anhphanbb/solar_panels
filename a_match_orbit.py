# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 23:52:32 2025

@author: domin
"""

import pandas as pd
import os

# Define file paths
big_csv_path = 'sp_orbit_predictions/mlspb/updated_cluster_expansion.csv'
folder_path = 'csv/AweSolarPanels/'
output_folder = 'csv/AweSolarPanels/updated/'  # Directory to save new files
os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

# Read the main CSV file
big_df = pd.read_csv(big_csv_path)

# List of monthly files
monthly_files = [
    'AweSolarPanelNov23.csv', 'AweSolarPanelDec23.csv', 'AweSolarPanelJan24.csv', 'AweSolarPanelFeb24.csv',
    'AweSolarPanelMar24.csv', 'AweSolarPanelApr24.csv', 'AweSolarPanelMay24.csv', 'AweSolarPanelJun24.csv',
    'AweSolarPanelJul24.csv', 'AweSolarPanelAug24.csv', 'AweSolarPanelSep24.csv'
]

# Process each monthly file
for file_name in monthly_files:
    file_path = os.path.join(folder_path, file_name)
    output_file_path = os.path.join(output_folder, file_name)
    df = pd.read_csv(file_path)
    
    # Merge on Orbit column
    merged_df = df.merge(big_df[['Orbit', 'TotalFrames', 'Start_Outside', 'End_Outside']], on='Orbit', how='left')
    
    # Calculate differences
    merged_df['First No Panel Difference'] = merged_df['Start_Outside'] - merged_df['First no panel']
    merged_df['Last No Panel Difference'] = merged_df['End_Outside'] - merged_df['Last no panel']
    
    # Reorder columns to insert new data at the correct positions
    columns_order = df.columns.tolist() + ['TotalFrames', 'Start_Outside', 'End_Outside', 'First No Panel Difference', 'Last No Panel Difference']
    merged_df = merged_df[columns_order]
    
    # Save the updated file in a new location
    merged_df.to_csv(output_file_path, index=False)
    print(f"Saved updated file: {output_file_path}")
