import joblib
import pandas as pd
from utils.constants import POLY
from src.inputs.input_handler import get_user_input
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Load the model
model = joblib.load(POLY)

# Define the feature names (adjust based on your actual input)
feature_names = ["Date"]  # Assuming we're predicting based on 'Date'

while True:
    # Get user input
    user_input = get_user_input()

    if user_input is None:
        break

    try:
        # Convert user input to the expected format
        features = pd.DataFrame([user_input], columns=feature_names)
        
        # Ensure 'Date' is a float
        features['Date'] = features['Date'].astype(float)

        # Make prediction
        prediction = model.predict(features)

        # Display the prediction result
        print(f"{Fore.BLUE}[+] Predicted Price of Gold: {prediction[0]:.2f} USD{Style.RESET_ALL}\n")

    except ValueError as e:
        print(f"{Fore.RED}[-] Invalid input: {e}{Style.RESET_ALL}")
