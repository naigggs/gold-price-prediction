# Gold Average Price Prediction

The project aims to predict average gold prices over the years using various data science techniques, specifically focusing on Polynomial Regression, LSTM, Random Forest Regressor and XGBoost Regression. By analyzing historical gold price data and identifying trends, the project seeks to provide accurate forecasts that can aid investors and analysts in making informed decisions in the commodities market. The use of different regression methods allows for a comprehensive comparison of performance and accuracy in price prediction.

## Dataset Description

The dataset consists of 2,301 instances, featuring one target variable and one feature. The target variable represents the average price of gold per troy ounce, while the feature is the corresponding date. This collection of monthly gold prices in USD, sourced from the World Gold Council, spans from 1833 to the present. The data is derived from historical records compiled by Timothy Green and enriched with additional information from the World Bank. This extensive dataset serves as a valuable resource for analyzing gold pricing trends and applying various regression techniques for precise price prediction.

## Dataset Snapshot
The following table shows a snapshot of the dataset used in this project:
- Date: Represents the date in the format YYYY-MM.
- Price: Indicates the average price of gold per troy ounce in USD.

## Summary of Findings

The analysis reveals that the average price of gold per troy ounce has been steadily increasing over the years. This upward trend highlights the growing value of gold as a commodity, reflecting various economic factors and market dynamics influencing its price.

The data supports a solid, long-term increasing trend for gold prices. This is consistent with gold's position as a safety investment since major price increases occur in reaction to global economic crises and periods of inflation (1970-1980 energy crisis, 2008 financial crisis, 2020 COVID pandemic), and prices decline slightly as the economy stabilizes. 

## Data Preprocessing

In the data preprocessing phase, the primary step involved converting the data type of the "Date" column from an object to a float. This transformation reformatted the date from the YYYY-MM format to a more suitable YYYY.MM format. This adjustment ensures that the dates are in a numerical format, facilitating easier manipulation and analysis during the modeling process.

## Exploratory Data Analysis

The dataset used for this analysis can be found at the following URL: [Gold Prices Dataset](https://datahub.io/core/gold-prices). This source provides comprehensive information on historical gold prices, allowing for a thorough examination of trends, fluctuations, and the overall behavior of gold prices over time.

## Visualization

### Date (YYYY-MM) vs Average Price of Gold per Troy Ounce in USD

The following scatter plot displays the relationship between the date (in YYYY.MM format) and the average price of gold per troy ounce in USD. This plot illustrates how gold prices are distributed over time.

![image](https://github.com/user-attachments/assets/2cb477f9-fba0-4620-96ff-88b4e26424e8)

> The plot shows a none to minimal jump in the price of gold from the year 1833 up to around 1973 when the prices reached more than 100, correlating to the 1970-1980 energy crisis, from the 1980s up to the 2000s the price started to fluctuate, even showing a decline, until the 2008 financial crisis happened where the price of gold shot up to the thousands, another increase in price happened in 2020s due to the COVID Pandemic. 

## Model Development

The model development process involved selecting and implementing various regression techniques to predict the average price of gold. Polynomial Regression and XGBoost Regression were chosen due to their effectiveness in time series forecasting and capturing complex patterns. Each model was trained on the preprocessed dataset, with the "Date" feature (in YYYY.MM format) as the independent variable and the average gold price as the target variable. Hyperparameter tuning and cross-validation were used to optimize model performance, and results were compared to determine the best-performing model for accurate gold price prediction. LSTM and Random Forest models were also implemented due to their proven effectiveness in time series prediction and handling complex patterns. The LSTM model, a type of recurrent neural network, was chosen for its ability to learn dependencies over time, useful for time series data where sequential relationships play a key role in accurate predictions. Meanwhile, the Random Forest model offers high accuracy and is resilient to overfitting, providing a stable ensemble learning approach that handles data variability well.

## Model Evaluation

The following table shows the evaluation metrics for the classification models trained on the Average Gold Prices Dataset:
| Model Name            | Model File                                                        | Accuracy |
| :-------------------- | :---------------------------------------------------------------- | :------- |
| XGBoost               | xgboost.py (/src/training/xgboost.py)                             | 0.99     |
| Polynomial Regression | polynomial_regression.py (/src/training/polynomial_regression.py) | 0.88     |
| Random Forest Model   | rf_model.py (/src/training/rf_model.py)                           | 0.99     |
| LSTM                  | lstm.py (/src/training/lstm.py)                                   | 0.99     |

## Conclusion

This project successfully demonstrates that various regression models can effectively predict the average price of gold, a commodity with a longstanding upward trend. By analyzing historical data, we can better understand how external factors—like economic crises and inflationary periods—have influenced gold prices. Among the models tested, XGBoost and Random Forest provided high predictive accuracy, emphasizing the robustness of ensemble methods in capturing both trends and fluctuations and the Long Short-Term Memory (LSTM) model further highlighted the strength of deep learning approaches for time series forecasting, as it captures sequential dependencies, making it suitable for complex patterns in long-term data. The Polynomial Regression model, while useful in recognizing long-term trends, was less effective in adapting to short-term volatility. These results underscore the importance of selecting models that balance trend identification with the flexibility to address volatile periods, making this analysis valuable for forecasting gold prices.

## Contributors
