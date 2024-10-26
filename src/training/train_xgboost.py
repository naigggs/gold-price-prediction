from utils.constants import PREPROCESSURL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle  # Import pickle for saving the model
from xgboost import XGBRegressor

# Load the dataset
data = pd.read_csv(PREPROCESSURL)

# Check the data types to ensure 'Date' is a float
print(data.dtypes)

# Clean column names
data.rename(columns={'Date    ': 'Date'}, inplace=True)
data.columns = data.columns.str.strip()

# Split the dataset into features and target
X = data[['Date']]  # Use 'Date' as the feature
y = data['Price']   # Use 'Price' as the target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Save the model as a .pkl file
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model saved as 'xgboost_model.pkl'")
