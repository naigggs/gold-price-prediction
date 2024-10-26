from utils.constants import PREPROCESSURL
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv(PREPROCESSURL)
print(data.columns)

# Clean column names
data.rename(columns={'Date    ': 'Date'}, inplace=True)
data.columns = data.columns.str.strip()

# Prepare the data
X = data[['Date']]  # Feature: Date in YYYY.MM format
y = data['Price']   # Target: Average price of gold

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R2): {r2}')

# Save the model
joblib.dump(model, 'gbm_model.joblib')
print("Model saved as 'gbm_model.joblib'")
