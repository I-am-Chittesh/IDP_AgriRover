import numpy as np
import pandas as pd

num_points = 200
grid_size = 20.0 

np.random.seed(42)
x_coords = np.random.uniform(0, grid_size, num_points)
y_coords = np.random.uniform(0, grid_size, num_points)

# DHT11 Temperature sensor has hardware limit 0-50°C. 
# Simulating a warm gradient (28°C - 38°C) getting hotter near coordinate (20, 20)
dist_to_heat = np.sqrt((x_coords - 20)**2 + (y_coords - 20)**2)
temperature = 38.0 - (dist_to_heat * 0.4) + np.random.normal(0, 0.5, num_points)
temperature = np.clip(temperature, 0, 50) # Hardware limits

# DHT11 Humidity sensor has hardware limit 20-90%. 
# Inversely correlated with temperature
humidity = 80.0 - (temperature - 28) * 2.5 + np.random.normal(0, 2, num_points)
humidity = np.clip(humidity, 20, 90)

# Soil Moisture sensor is mapped from 0-100%. 
# Simulating a wet patch near coordinate (5, 5)
dist_to_water = np.sqrt((x_coords - 5)**2 + (y_coords - 5)**2)
moisture = 95.0 - (dist_to_water * 4.5) + np.random.normal(0, 4, num_points)
moisture = np.clip(moisture, 0, 100)

df = pd.DataFrame({
    'x': np.round(x_coords, 2),
    'y': np.round(y_coords, 2),
    'temperature': np.round(temperature, 2),
    'humidity': np.round(humidity, 2),
    'moisture': np.round(moisture, 2)
})

df.to_csv('rover_sensors.csv', index=False)