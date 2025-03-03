import pandas as pd

def find_outside_clusters(df):
    start_outside = []
    end_outside = []
    
    for _, row in df.iterrows():
        total_frames = int(row['TotalFrames'])
        
        # Handle NaN values in cluster columns
        cluster1_t0 = int(row['Cluster1_Expanded_t0']) if pd.notna(row['Cluster1_Expanded_t0']) else 0
        cluster1_t1 = int(row['Cluster1_Expanded_t1']) if pd.notna(row['Cluster1_Expanded_t1']) else 0
        cluster2_t0 = int(row['Cluster2_Expanded_t0']) if pd.notna(row['Cluster2_Expanded_t0']) else 0
        cluster2_t1 = int(row['Cluster2_Expanded_t1']) if pd.notna(row['Cluster2_Expanded_t1']) else 0
        
        cluster1_range = set(range(cluster1_t0, cluster1_t1 + 1))
        cluster2_range = set(range(cluster2_t0, cluster2_t1 + 1))
        
        all_frames = set(range(total_frames))
        outside_frames = sorted(all_frames - cluster1_range - cluster2_range)
        
        if outside_frames:
            start_outside.append(outside_frames[0])
            end_outside.append(outside_frames[-1])
        else:
            start_outside.append(None)
            end_outside.append(None)
    
    df['Start_Outside'] = start_outside
    df['End_Outside'] = end_outside
    
    return df

# File paths
csv_input_path = 'sp_orbit_predictions/mlspb/cluster_expansion.csv'
csv_output_path = 'sp_orbit_predictions/mlspb/updated_cluster_expansion.csv'

# Process CSV
df = pd.read_csv(csv_input_path)
df.fillna(0, inplace=True)  # Replace NaN values with 0
df = df.astype({'TotalFrames': 'int', 'Cluster1_Expanded_t0': 'int', 'Cluster1_Expanded_t1': 'int', 'Cluster2_Expanded_t0': 'int', 'Cluster2_Expanded_t1': 'int'})
df = find_outside_clusters(df)
df.to_csv(csv_output_path, index=False)
