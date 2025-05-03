# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 00:23:05 2025

@author: domin
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

# Define folder path
updated_folder = 'csv/AweSolarPanels/updated/'
plot_folder = 'csv/AweSolarPanels/plots/'
os.makedirs(plot_folder, exist_ok=True)  # Ensure plot folder exists

# List of updated monthly files
monthly_files = [
    'AweSolarPanelNov23.csv', 'AweSolarPanelDec23.csv', 'AweSolarPanelJan24.csv', 'AweSolarPanelFeb24.csv',
    'AweSolarPanelMar24.csv', 'AweSolarPanelApr24.csv', 'AweSolarPanelMay24.csv', 'AweSolarPanelJun24.csv',
    'AweSolarPanelJul24.csv', 'AweSolarPanelAug24.csv', 'AweSolarPanelSep24.csv'
]

# Combined data storage
combined_df = pd.DataFrame()

# Process each file for plotting
for file_name in monthly_files:
    file_path = os.path.join(updated_folder, file_name)
    df = pd.read_csv(file_path)
    
    if 'Orbit' in df.columns and 'First No Panel Difference' in df.columns and 'Last No Panel Difference' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df['Orbit'], df['First No Panel Difference'], label='First No Panel Difference', marker='o')
        plt.plot(df['Orbit'], df['Last No Panel Difference'], label='Last No Panel Difference', marker='s')
        
        plt.xlabel('Orbit')
        plt.ylabel('Difference')
        plt.title(f'No Panel Differences for {file_name}')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_file_path = os.path.join(plot_folder, file_name.replace('.csv', '.png'))
        plt.savefig(plot_file_path)
        plt.close()
        print(f"Saved plot: {plot_file_path}")
        
        # Append data for combined plot
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# Create combined plot
if not combined_df.empty:
    plt.figure(figsize=(12, 6))
    plt.plot(combined_df['Orbit'], combined_df['First No Panel Difference'], label='First No Panel Difference', marker='o', linestyle='None')
    plt.plot(combined_df['Orbit'], combined_df['Last No Panel Difference'], label='Last No Panel Difference', marker='s', linestyle='None')
    
    plt.xlabel('Orbit')
    plt.ylabel('Difference')
    plt.title('No Panel Differences Across All Months')
    plt.legend()
    plt.grid(True)
    
    # Save combined plot
    combined_plot_path = os.path.join(plot_folder, 'combined_no_panel_differences.png')
    plt.savefig(combined_plot_path)
    plt.close()
    print(f"Saved combined plot: {combined_plot_path}")
