import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv("./datasets/preprocess.csv")

# Correctly parse 'Date' format
data['Date'] = pd.to_datetime(data['Date'].astype(str), format='%Y.%m')
data.set_index('Date', inplace=True)

# Prepare the data
data['Date'] = data.index.year + (data.index.month - 1) / 12  # Convert dates to a continuous variable
X = data[['Date']]  # Feature: Date
y = data['Price']  # Target: Price

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

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
joblib.dump(model, 'rf_model.joblib')
print("Model saved as 'rf_model.joblib'")