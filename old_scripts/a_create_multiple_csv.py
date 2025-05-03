# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 23:31:46 2025

@author: domin
"""

import pandas as pd

# Define file paths
excel_file = 'csv/AweSolarPanels.xlsx'
output_dir = 'csv/AweSolarPanels/'  # Directory to save CSV files

# Load Excel file
xls = pd.ExcelFile(excel_file)

# Save each sheet as a separate CSV file
for sheet_name in xls.sheet_names:
    df = xls.parse(sheet_name)
    csv_file = f"{output_dir}{sheet_name}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Saved {csv_file}")
