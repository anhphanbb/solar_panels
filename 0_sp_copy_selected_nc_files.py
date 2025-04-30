# import os
# import shutil
# import re

# # Define source and destination directories
# src_dir = r"Z:\soc\l0c\2025\01"
# dst_dir = r"E:\soc\l0c\2025\01"

# # Create the destination directory if it doesn't exist
# os.makedirs(dst_dir, exist_ok=True)

# # Define orbit range
# min_orbit = 6292
# max_orbit = 6451

# # Regex pattern to extract orbit number before '_v01'
# pattern = re.compile(r'_([0-9]{5})_v01\.nc$')

# # Loop through files and filter
# for filename in os.listdir(src_dir):
#     match = pattern.search(filename)
#     if match:
#         orbit_num = int(match.group(1))
#         if min_orbit <= orbit_num <= max_orbit:
#             src_path = os.path.join(src_dir, filename)
#             dst_path = os.path.join(dst_dir, filename)
#             shutil.copy2(src_path, dst_path)
#             print(f"Copied: {filename}")

# print("Done copying files.")


import os
import shutil
import re

# Define source and destination directories
src_dir = r"Z:\soc\l0c\2025\01"
dst_dir = r"E:\soc\l0c\2025\so01"

# Create the destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# List of selected orbits to copy
selected_orbits = [6292, 6320, 6350, 6380, 6410, 6440, 6470, 6500, 6530, 6560, 6590, 6620, 6650, 6680, 6710, 6740, 6770]

# Regex pattern to extract orbit number before '_v01'
pattern = re.compile(r'_([0-9]{5})_v01\.nc$')

# Loop through files and filter
for filename in os.listdir(src_dir):
    match = pattern.search(filename)
    if match:
        orbit_num = int(match.group(1))
        if orbit_num in selected_orbits:
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {filename}")

print("Done copying selected files.")
