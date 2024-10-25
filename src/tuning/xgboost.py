from utils.constants import DATASETURL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv(DATASETURL)

# Split the dataset into features and target
X = data.drop("class", axis=1)
y = data["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Build the model
model = Pipeline(
    [("scaler", StandardScaler()), ("xgb", XGBClassifier(n_estimators=500))]
)

model.fit(X_train, y_train)

# Evaluate the model
test_score = model.score(X_test, y_test)
print(
    f"Fine Tuned XGBoost Classifier Accuracy (Test): {test_score:.2f} or {test_score*100:.2f}%"
)
train_score = model.score(X_train, y_train)
print(
    f"Fine Tuned XGBoost Classifier Accuracy (Train): {train_score:.2f} or {train_score*100:.2f}%"
)
print("Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# Generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    model,
    X,
    y,
    cv=5,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy",
)

# Calculate mean and standard deviation of training and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.fill_between(
    train_sizes,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.1,
    color="r",
)
plt.fill_between(
    train_sizes,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.1,
    color="g",
)
plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")

# Add labels and title
plt.title("Learning Curve for Fine-Tuned XGBoost Classifier")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy Score")
plt.legend(loc="best")
plt.grid()

# Show plot
plt.show()
