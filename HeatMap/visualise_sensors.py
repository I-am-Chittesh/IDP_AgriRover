import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load the generated data
df = pd.read_csv('rover_sensors.csv')
x = df['x']
y = df['y']
temp = df['temperature']
moist = df['moisture']

# Create the interpolation grid
grid_x, grid_y = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]

# Interpolate both temperature and moisture data simultaneously
grid_temp = griddata((x, y), temp, (grid_x, grid_y), method='cubic')
grid_moist = griddata((x, y), moist, (grid_x, grid_y), method='cubic')

# Set up a figure with 2 side-by-side subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- MAP 1: Temperature Heatmap ---
# Using 'inferno' color map which is visually standard for heat/temperature
im1 = ax1.imshow(grid_temp.T, extent=(x.min(), x.max(), y.min(), y.max()), 
                 origin='lower', cmap='inferno')
ax1.scatter(x, y, c='white', s=10, alpha=0.4, label='Readings') # White dots for visibility
ax1.set_title('DHT11 Temperature Heatmap')
ax1.set_xlabel('X Coordinate (m)')
ax1.set_ylabel('Y Coordinate (m)')
fig.colorbar(im1, ax=ax1, label='Temperature (°C)')

# --- MAP 2: Soil Moisture Map ---
# Using 'YlGnBu' (Yellow-Green-Blue) for water/moisture
im2 = ax2.imshow(grid_moist.T, extent=(x.min(), x.max(), y.min(), y.max()), 
                 origin='lower', cmap='YlGnBu')
ax2.scatter(x, y, c='red', s=10, alpha=0.4, label='Readings')
ax2.set_title('Soil Moisture Surface Map')
ax2.set_xlabel('X Coordinate (m)')
ax2.set_ylabel('Y Coordinate (m)')
fig.colorbar(im2, ax=ax2, label='Moisture Level (%)')

# Render the dashboard
plt.tight_layout()
plt.show()