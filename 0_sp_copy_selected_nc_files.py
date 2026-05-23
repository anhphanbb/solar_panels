import os
import shutil
import re

# Define source and destination directories
src_dir = r"Y:\soc\public\TranferFromSDL"
dst_dir = r"E:\soc\l0c\2026\04"

# Create the destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Define orbit range
min_orbit = 0
max_orbit = 99999

# Regex pattern to extract orbit number before '_v01'
pattern = re.compile(r'_([0-9]{5})_v99\.nc$')

# Loop through files and filter
for filename in os.listdir(src_dir):
    match = pattern.search(filename)
    # if match and 'bkg' in filename:
    if match:
        orbit_num = int(match.group(1))
        if min_orbit <= orbit_num <= max_orbit:
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {filename}")

print("Done copying files.")


# import os
# import shutil
# import re

# # Define source and destination directories
# src_dir = r"Z:\socfiles\l0c\2024_03"
# dst_dir = r"E:\soc\l0c\2024\03"

# # Create the destination directory if it doesn't exist
# os.makedirs(dst_dir, exist_ok=True)

# # List of selected orbits to copy
# selected_orbits = [1580, 1598, 1607, 1614, 1800, 1826, 1847, 1874, 1895, 1903, 1904, 1910, 1942, 2002, 2027]

# # Regex pattern to extract orbit number before '_v01'
# pattern = re.compile(r'_([0-9]{5})_v23\.nc$')

# # Loop through files and filter
# for filename in os.listdir(src_dir):
#     match = pattern.search(filename)
#     if match:
#         orbit_num = int(match.group(1))
#         if orbit_num in selected_orbits:
#             src_path = os.path.join(src_dir, filename)
#             dst_path = os.path.join(dst_dir, filename)
#             shutil.copy2(src_path, dst_path)
#             print(f"Copied: {filename}")

# print("Done copying selected files.")
