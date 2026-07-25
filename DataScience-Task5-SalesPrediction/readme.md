# Sales Prediction Using Python

## Objective
Predict product sales based on advertising spend across TV, Radio, and Newspaper.

## Dataset
Advertising Sales Dataset (Kaggle) — 200 records, 3 features + Sales target.

## Approach
1. EDA: correlation matrix, scatter plots per channel.
2. Train/test split (80/20).
3. Trained Linear Regression (baseline) and Random Forest.
4. Evaluated with MAE, RMSE, R².
5. Interpreted coefficients to identify each channel's impact.

## Results
Linear Regression: MAE 1.46, RMSE 1.78, R² 0.899.
TV advertising showed the strongest positive impact on sales, Radio moderate, 
Newspaper negligible.

## Tech Stack
Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter Notebook