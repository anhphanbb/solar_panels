import os
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import re

folder_labels = r'E:\soc\l0c\2025\so01\mlspb_from_labels'
folder_predictions = r'E:\soc\l0d\2025\so01'

results = []

def extract_orbit_number(filename):
    match = re.search(r'_(\d{5})_', filename)
    if match:
        return int(match.group(1))
    else:
        return None

def compare_mlsp_arrays(label, pred):
    label = label.astype(bool)
    pred = pred.astype(bool)

    TP = np.sum((label == 1) & (pred == 1))
    TN = np.sum((label == 0) & (pred == 0))
    FP = np.sum((label == 0) & (pred == 1))
    FN = np.sum((label == 1) & (pred == 0))
    
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    
    return TP, TN, FP, FN, accuracy, recall

# Match files by name
for filename in os.listdir(folder_labels):
    if not filename.endswith('.nc'):
        continue

    orbit = extract_orbit_number(filename)
    if orbit is None:
        print(f"Could not extract orbit number from {filename}")
        continue
    
    path_label = os.path.join(folder_labels, filename)
    path_pred = os.path.join(folder_predictions, filename)
    
    if not os.path.exists(path_pred):
        print(f"Prediction file not found for {filename}")
        continue
    
    with Dataset(path_label, 'r') as nc_label, Dataset(path_pred, 'r') as nc_pred:
        label_data = nc_label.variables['MLSPB'][:]
        pred_data = nc_pred.variables['MLSPB'][:]
        
        if label_data.shape != pred_data.shape:
            print(f"Shape mismatch for orbit {orbit}")
            continue
        
        TP, TN, FP, FN, accuracy, recall = compare_mlsp_arrays(label_data, pred_data)
        total = TP + TN + FP + FN
        results.append({
            'Orbit': orbit,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'TP (%)': round(100 * TP / total, 2),
            'TN (%)': round(100 * TN / total, 2),
            'FP (%)': round(100 * FP / total, 2),
            'FN (%)': round(100 * FN / total, 2),
            'Accuracy': round(accuracy, 4),
            'Recall': round(recall, 4)
        })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv('csv/mlspb_comparison_results_april_29_2025.csv', index=False)
print("Saved results to csv/mlspb_comparison_results.csv")
