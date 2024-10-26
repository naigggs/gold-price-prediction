# Gold Average Price Prediction

The project aims to predict average gold prices over the years using various data science techniques, specifically focusing on Linear Regression, Polynomial Regression, and XGBoost Regression. By analyzing historical gold price data and identifying trends, the project seeks to provide accurate forecasts that can aid investors and analysts in making informed decisions in the commodities market. The use of different regression methods allows for a comprehensive comparison of performance and accuracy in price prediction.

## Dataset Description

The dataset consists of 2,301 instances, featuring one target variable and one feature. The target variable represents the average price of gold per troy ounce, while the feature is the corresponding date. This collection of monthly gold prices in USD, sourced from the World Gold Council, spans from 1833 to the present. The data is derived from historical records compiled by Timothy Green and enriched with additional information from the World Bank. This extensive dataset serves as a valuable resource for analyzing gold pricing trends and applying various regression techniques for precise price prediction.

## Dataset Snapshot
The following table shows a snapshot of the dataset used in this project:
- Date: Represents the date in the format YYYY-MM.
- Price: Indicates the average price of gold per troy ounce in USD.

## Summary of Findings

The analysis reveals that the average price of gold per troy ounce has been steadily increasing over the years. This upward trend highlights the growing value of gold as a commodity, reflecting various economic factors and market dynamics influencing its price.

## Data Preprocessing

In the data preprocessing phase, the primary step involved converting the data type of the "Date" column from an object to a float. This transformation reformatted the date from the YYYY-MM format to a more suitable YYYY.MM format. This adjustment ensures that the dates are in a numerical format, facilitating easier manipulation and analysis during the modeling process.

## Exploratory Data Analysis

The dataset used for this analysis can be found at the following URL: Gold Prices Dataset. This source provides comprehensive information on historical gold prices, allowing for a thorough examination of trends, fluctuations, and the overall behavior of gold prices over time.

### Visualization

### Date (YYYY-MM) vs Average Price of Gold per Troy Ounce in USD

The following scatter plot displays the relationship between the date (in YYYY.MM format) and the average price of gold per troy ounce in USD. This plot illustrates how gold prices are distributed over time.

![image](https://github.com/user-attachments/assets/2cb477f9-fba0-4620-96ff-88b4e26424e8)

> It is clear that the average price of gold per troy ounce in USD has been on the rise over the years. While there are periods when it declines, the overall trend remains upward.

## Model Development

(model_development) - [text form] This will be a brief description of the model development process that you have taken to create the model for the project.

## Model Evaluation

The following table shows the evaluation metrics for the classification models trained on the Average Gold Prices Dataset:
| Model Name            | Model File                                                        | Accuracy | Precision | Recall | F1 Score |
| :-------------------- | :---------------------------------------------------------------- | :------- | :-------- | :----- | :------- |
| XGBoost               | xgboost.py (/src/training/xgboost.py)                             | 0.99     | 0.99      | 0.99   | 0.99     |
| Polynomial Regression | polynomial_regression.py (/src/training/polynomial_regression.py) | 0.88     | 0.88      | 0.88   | 0.88     |

## Conclusion

(conclusion) - [text form] This will be a brief conclusion of the project, summarizing the key findings and insights from the analysis.

## Contributors
