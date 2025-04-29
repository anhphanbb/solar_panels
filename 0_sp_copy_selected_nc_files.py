import os
import shutil
import re

# Define source and destination directories
src_dir = r"Z:\soc\l0c\2025\01"
dst_dir = r"E:\soc\l0c\2025\01"

# Create the destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Define orbit range
min_orbit = 6292
max_orbit = 6451

# Regex pattern to extract orbit number before '_v01'
pattern = re.compile(r'_([0-9]{5})_v01\.nc$')

# Loop through files and filter
for filename in os.listdir(src_dir):
    match = pattern.search(filename)
    if match:
        orbit_num = int(match.group(1))
        if min_orbit <= orbit_num <= max_orbit:
            src_path = os.path.join(src_dir, filename)
            dst_path = os.path.join(dst_dir, filename)
            shutil.copy2(src_path, dst_path)
            print(f"Copied: {filename}")

print("Done copying files.")
