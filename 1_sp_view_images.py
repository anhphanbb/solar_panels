"""
Updated on Wed Nov 20 2024

@author: anhph
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, RangeSlider, Button
import os
import re

# Define the path to the parent directory where the dataset is located
# parent_directory = r'E:\GitHub\solar_panels\sp_selected_orbits_no_glare'
parent_directory = r'E:\soc\l0c\2024\11'

# Define the orbit number
orbit_number = 5471  # orbit number

# Pad the orbit number with zeros until it has 5 digits
orbit_str = str(orbit_number).zfill(5)

# Search for the correct file name in all subdirectories
pattern = re.compile(r'awe_l0c_q20(.*)_' + orbit_str + r'_(.*)\.nc')
dataset_filename = None
dataset_path = None

for root, dirs, files in os.walk(parent_directory):
    for file in files:
        if pattern.match(file):
            dataset_filename = file
            dataset_path = os.path.join(root, file)
            break
    if dataset_filename:
        break

if dataset_filename is None:
    raise FileNotFoundError(f"No file found for orbit number {orbit_str}")
    
# Load the dataset
dataset = nc.Dataset(dataset_path, 'r')
radiance = dataset.variables['Radiance'][:]
iss_latitude = dataset.variables['ISS_Latitude'][:]  # Load ISS latitude data
iss_longitude = dataset.variables['ISS_Longitude'][:]  # Load ISS longitude data

print("=== Global Attributes ===")
for attr in dataset.ncattrs():
    print(f"{attr}: {dataset.getncattr(attr)}")

print("\n=== Dimensions ===")
for dim in dataset.dimensions.keys():
    print(f"{dim}: {len(dataset.dimensions[dim])}")

print("\n=== Variables ===")
for var in dataset.variables.keys():
    print(f"{var}: {dataset.variables[var]}")
    print("Attributes:")
    for attr in dataset.variables[var].ncattrs():
        print(f"    {attr}: {dataset.variables[var].getncattr(attr)}")
        
print("\n=== Chunk Sizes ===")
for var_name, variable in dataset.variables.items():
    chunk_sizes = variable.chunking()
    print(f"{var_name}: {chunk_sizes}")

# Close the dataset
dataset.close()

# Initial setup
current_time_step = 0
show_numbers = True  # Variable to track the state of grid box number display
show_lines = False  # Variable to track the state of the vertical and horizontal lines display

# Calculate initial vmin and vmax using the 0.4th and 99.7th percentiles
radiance_at_time_0 = radiance[0, :, :]
radiance_flat = radiance_at_time_0.flatten()
radiance_flat = radiance_flat[~np.isnan(radiance_flat)]
vmin_default = np.percentile(radiance_flat, 0.4) * 0.85
vmax_default = np.percentile(radiance_flat, 99.7) * 1.2

# Create figure and axes for the plot
fig, ax = plt.subplots(figsize=(12, 12))

# Adjust the figure to add space for the sliders
plt.subplots_adjust(bottom=0.3)

# Create the range slider for vmin and vmax on the bottom right
ax_range_slider = plt.axes([0.1, 0.1, 0.7, 0.03], facecolor='lightgoldenrodyellow')
range_slider = RangeSlider(ax_range_slider, 'vmin - vmax', 0, 40, valinit=(vmin_default, vmax_default))

# Create the slider for time step on the bottom left
ax_slider = plt.axes([0.1, 0.05, 0.7, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Time Step', 0, radiance.shape[0] - 1, valinit=current_time_step, valfmt='%0.0f')

colorbar = None  # To keep track of the colorbar

# Set initial vmin and vmax
vmin, vmax = vmin_default, vmax_default

def update_plot(time_step):
    global colorbar, current_time_step
    current_time_step = int(time_step)
    
    # Get vmin and vmax from the range slider
    vmin, vmax = range_slider.val
    
    # Ensure vmin <= vmax
    vmin, vmax = min(vmin, vmax), max(vmin, vmax)
    
    # Clear previous content
    ax.clear()
    radiance_at_time = radiance[current_time_step, :, :]
    iss_lat = iss_latitude[current_time_step]
    iss_lon = iss_longitude[current_time_step]
    
    # Plot the radiance data
    img = ax.imshow(radiance_at_time, origin='lower', cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(f'Radiance at Time Step {current_time_step}\nISS Position: Lat {iss_lat:.2f}, Lon {iss_lon:.2f}\nOrbit Number: {orbit_str}')
    ax.set_xlabel('Spatial Dimension X')
    ax.set_ylabel('Spatial Dimension Y')
    
    # Draw vertical and horizontal lines if enabled
    if show_lines:
        for x in range(42, radiance_at_time.shape[1], 43):
            ax.axvline(x=x, color='green', linestyle='-')
        for y in range(42, radiance_at_time.shape[0], 43):
            ax.axhline(y=y, color='green', linestyle='-')
            
    # Display the box numbers with a semi-transparent background if enabled
    if show_numbers:
        for x in range(0, radiance_at_time.shape[1], 43):
            for y in range(0, radiance_at_time.shape[0], 43):
                box_label_x = x + 21
                box_label_y = y + 21
                ax.text(
                    box_label_x, box_label_y, f"({x//43},{y//43})",
                    color='blue', fontsize=16, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')  # Semi-transparent background
                )
    
    # Set the aspect of the plot axis to equal, enforcing a 1:1 aspect ratio
    ax.set_aspect('equal')

    # Add colorbar if it doesn't exist
    if colorbar is None:
        colorbar = plt.colorbar(img, ax=ax, orientation='vertical')
    else:
        colorbar.update_normal(img)

    plt.draw()

def toggle_numbers(event):
    global show_numbers
    show_numbers = not show_numbers
    update_plot(current_time_step)

def update_vmin_vmax(event):
    global vmin, vmax, current_time_step
    radiance_at_time = radiance[current_time_step, :, :]
    radiance_flat = radiance_at_time.flatten()
    radiance_flat = radiance_flat[~np.isnan(radiance_flat)]
    
    if len(radiance_flat) == 0:
        raise ValueError("No valid data to compute percentiles.")
    
    vmin = np.percentile(radiance_flat, 0.4) * 0.85
    vmax = np.percentile(radiance_flat, 99.7) * 1.2
    range_slider.set_val((vmin, vmax))

def increase_range(event):
    global vmin, vmax
    vmin, vmax = range_slider.val
    range_slider.set_val((vmin * 0.96, vmax * 1.05))

def decrease_range(event):
    global vmin, vmax
    vmin, vmax = range_slider.val
    range_slider.set_val((vmin / 0.96, vmax / 1.05))

def toggle_lines(event):
    global show_lines
    show_lines = not show_lines
    update_plot(current_time_step)

# Connect the slider and range slider to the update_plot function
slider.on_changed(lambda val: update_plot(val))
range_slider.on_changed(lambda val: update_plot(slider.val))

# Create buttons and connect their functions
ax_button_vmin_vmax = plt.axes([0.1, 0.15, 0.14, 0.03], facecolor='lightgoldenrodyellow')
button_vmin_vmax = Button(ax_button_vmin_vmax, 'Set vmin-vmax (V)')
button_vmin_vmax.on_clicked(update_vmin_vmax)

ax_button_numbers = plt.axes([0.46, 0.15, 0.1, 0.03], facecolor='lightgoldenrodyellow')
button_numbers = Button(ax_button_numbers, 'Toggle Numbers')
button_numbers.on_clicked(toggle_numbers)

ax_button_lines = plt.axes([0.58, 0.15, 0.1, 0.03], facecolor='lightgoldenrodyellow')
button_lines = Button(ax_button_lines, 'Toggle Lines')
button_lines.on_clicked(toggle_lines)

ax_button_increase_range = plt.axes([0.26, 0.15, 0.08, 0.03], facecolor='lightgoldenrodyellow')
button_increase_range = Button(ax_button_increase_range, '+Range (B)')
button_increase_range.on_clicked(increase_range)

ax_button_decrease_range = plt.axes([0.36, 0.15, 0.08, 0.03], facecolor='lightgoldenrodyellow')
button_decrease_range = Button(ax_button_decrease_range, '-Range (C)')
button_decrease_range.on_clicked(decrease_range)

# Key press event
def on_key(event):
    global current_time_step
    if event.key == 'right':
        current_time_step = min(current_time_step + 1, radiance.shape[0] - 1)
    elif event.key == 'left':
        current_time_step = max(current_time_step - 1, 0)
    elif event.key == 'up':
        current_time_step = max(current_time_step - 20, 0)
    elif event.key == 'down':
        current_time_step = min(current_time_step + 20, radiance.shape[0] - 1)
    elif event.key == 'v':
        update_vmin_vmax(None)
    elif event.key == 'b':
        increase_range(None)
    elif event.key == 'c':
        decrease_range(None)

    slider.set_val(current_time_step)

fig.canvas.mpl_connect('key_press_event', on_key)

update_plot(current_time_step)

plt.show()
