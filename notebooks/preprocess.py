import os
import pandas as pd
from utils.constants import DATASETURL

# Load the data
data = pd.read_csv(DATASETURL)

# Ensure the Date column is treated as a string
data['Date'] = data['Date'].astype(str)  # Convert Date to string type

# Replace '-' with '.' and convert to float
data['Date'] = data['Date'].str.replace('-', '.').astype(float)

# Save the preprocessed data
directory = 'datasets'

if not os.path.exists(directory):
    os.makedirs(directory)

filepath = os.path.join(directory, 'preprocess.csv')

data.to_csv(filepath, index=False)
