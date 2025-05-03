# -*- coding: utf-8 -*-
"""
Created on Fri May  2 13:19:54 2025

@author: domin
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Highlighted points
highlight_layer0 = [(4, 0), (4, 1), (5, 0), (5, 1), (6, 1), (5, 4), (6, 4), (6, 5)]
highlight_layer1 = [(5, 0), (5, 1), (5, 3), (5, 2), (5, 4)]

# Plot only highlighted cubes
for (x, y) in highlight_layer0:
    ax.bar3d(x, y, 0, 1, 1, 1, shade=True, color='red', alpha=0.8, edgecolor='black')
for (x, y) in highlight_layer1:
    ax.bar3d(x, y, 1, 1, 1, 1, shade=True, color='blue', alpha=0.8, edgecolor='black')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Highlighted Points Only (Red: Layer 0, Blue: Layer 1)')

# Set equal scale
ax.set_box_aspect([7, 7, 2])
ax.set_xlim(0, 7)
ax.set_ylim(0, 7)
ax.set_zlim(0, 2)

plt.show()
