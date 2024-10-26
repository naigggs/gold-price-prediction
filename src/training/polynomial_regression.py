from utils.constants import PREPROCESSURL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv(PREPROCESSURL)
print(data.info())
print(data.columns)

# Clean column names
data.rename(columns={'Date    ': 'Date'}, inplace=True)
data.columns = data.columns.str.strip()

# Split the dataset into features and target
X = data[['Date']]  # Use 'Date' as the feature
y = data['Price']   # Use 'Price' as the target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a polynomial regression model (try degree=2, 3, etc. for best performance)
degree = 3  # Increase this value if needed
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model.fit(X_train, y_train)

# Save the model to a file
model_filename = "poly.pkl"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

# Evaluate the model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Additional metrics for model evaluation
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)

# Display results
print(f'Polynomial Regression Mean Squared Error (Train): {train_mse:.2f}')
print(f'Polynomial Regression Mean Squared Error (Test): {test_mse:.2f}')
print(f'Polynomial Regression Mean Absolute Error (Train): {train_mae:.2f}')
print(f'Polynomial Regression Mean Absolute Error (Test): {test_mae:.2f}')
print(f'Polynomial Regression Root Mean Squared Error (Train): {train_rmse:.2f}')
print(f'Polynomial Regression Root Mean Squared Error (Test): {test_rmse:.2f}')
print(f'Polynomial Regression R^2 Score (Train): {train_r2:.2f}')
print(f'Polynomial Regression R^2 Score (Test): {test_r2:.2f}')
print(f"Model Accuracy (R² Score on Test Set): {test_r2:.2f} or {test_r2 * 100:.2f}%")

# Generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2')

# Calculate mean and standard deviation of training and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

# Add labels and title
plt.title("Learning Curve for Polynomial Regression Model")
plt.xlabel("Training Examples")
plt.ylabel("R^2 Score")
plt.legend(loc="best")
plt.grid()

# Show plot
plt.show()
