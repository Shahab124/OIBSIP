# Task 3 — Car Price Prediction with Machine Learning

A regression problem: estimate the resale price of a used car from its attributes —
showroom price, kilometres driven, fuel type, seller type, transmission, previous
owners and age. The interesting part is that car depreciation is not linear; prices
fall steeply in the first few years and then flatten out. That makes this a useful
test of whether a tree-based model can beat a straight-line one, and the results here
show that it does, by a wide margin.

## Dataset

- **File used:** `car data.csv` — CarDekho used-car dataset (Kaggle)
- **Shape:** 301 rows × 9 columns
- **Columns:** Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type,
  Seller_Type, Transmission, Owner
- **Target:** `Selling_Price` (in lakhs ₹)
- **Missing values:** none
- **Duplicate rows:** 2 (detected, not removed)

Target distribution (`Selling_Price`, lakhs ₹):

| count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|
| 301 | 4.661 | 5.083 | 0.10 | 0.90 | 3.60 | 6.00 | 35.00 |

The folder also contains three other CarDekho CSVs — `CAR DETAILS FROM CAR DEKHO.csv`,
`Car details v3.csv` and `car details v4.csv` — which were downloaded but are not
loaded or used by the notebook.

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter Notebook

## Approach

1. Loaded `car data.csv` and inspected shape and column list — 301 rows, 9 columns.
2. Ran `info()`, `isnull().sum()` and `duplicated().sum()` — no nulls, 2 duplicates.
3. Engineered `Car_Age` as `2024 - Year`, then dropped the raw `Year` column.
4. Plotted the distribution of `Selling_Price` (30-bin histogram with KDE).
5. Plotted `Selling_Price` by `Fuel_Type` (box plot) and against `Car_Age`
   (scatter plot).
6. One-hot encoded `Fuel_Type`, `Seller_Type`, `Transmission` and `Car_Name` using
   `pd.get_dummies(drop_first=True)`, expanding the frame to 106 columns.
7. Split into train and test sets 80/20 (`random_state=42`) → 240 training rows ×
   105 features, 61 test rows.
8. Drew a correlation heatmap over the numeric columns (Selling_Price,
   Present_Price, Kms_Driven, Car_Age, Owner).
9. Trained a **Linear Regression** baseline and scored MAE, RMSE and R².
10. Trained a **Random Forest Regressor** (`n_estimators=100`, `random_state=42`)
    and scored the same three metrics.
11. Plotted the top 10 features by Random Forest feature importance as a horizontal
    bar chart.
12. Summarised the comparison in a markdown table.

## Results

Metrics on the 61-row held-out test set. MAE and RMSE are in lakhs ₹.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 1.1197 | 1.5964 | 0.8894 |
| **Random Forest Regressor** | **0.6222** | **0.9243** | **0.9629** |

**Best model: Random Forest Regressor.** It beats the linear baseline on all three
metrics — MAE down 44%, RMSE down 42%, and R² up from 0.889 to 0.963.

**Visualisations produced:** selling-price histogram with KDE, price-by-fuel-type box
plot, price-vs-age scatter plot, numeric correlation heatmap, top-10 feature
importance bar chart.

## Key Findings

- **Non-linearity is worth real money here.** Random Forest cuts mean absolute error
  from ₹1.12 lakh to ₹0.62 lakh — a 44% reduction. Depreciation curves steeply in the
  first years then flattens, and a straight-line model simply cannot represent that
  shape.
- **An MAE of ₹0.62 lakh is about 13% of the mean selling price** (₹4.66 lakh),
  which puts the Random Forest's typical error in a genuinely usable range for
  ballpark valuation.
- **`Present_Price` dominates.** Both the notebook's own analysis and the feature
  importance chart identify current showroom price as the strongest predictor —
  unsurprising, since resale value is largely a discount applied to new price.
- **The encoding scheme is the main weakness.** One-hot encoding `Car_Name` produced
  roughly 100 binary columns from only 301 rows, so the model has more features than
  it has training examples (105 features, 240 training rows). Extracting the brand
  from the car name — as the task brief calls for — would collapse those hundred
  columns into a handful and reduce the risk of the model memorising individual car
  models rather than learning general depreciation patterns.

## How to Run

Dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn notebook
```

`car data.csv` is committed in this folder, so no download is needed. Open the
notebook and run all cells top to bottom:

```bash
jupyter notebook Car_Price_Prediction.ipynb
```

Note: `Car_Age` is computed as `2024 - Year`. Re-running in a later year will not
update that baseline automatically.

## Requirement Checklist

- [x] Null check performed (`isnull().sum()` — zero nulls found)
- [ ] Duplicates removed — `duplicated().sum()` reports 2, but they were never dropped
- [ ] Inconsistent categoricals cleaned — no normalisation of casing or spelling was
      applied to `Car_Name`, `Fuel_Type`, `Seller_Type` or `Transmission`
- [x] Feature engineering — `Car_Age` derived from `Year`
- [ ] Feature engineering — brand extracted from car name; `Car_Name` was one-hot
      encoded whole instead, producing ~100 columns
- [x] EDA — price distribution histogram
- [x] EDA — price vs fuel type (box plot)
- [x] EDA — price vs car age (scatter plot)
- [x] Categorical encoding (`pd.get_dummies`, `drop_first=True`)
- [x] Correlation heatmap
- [x] Train/test split (80/20, `random_state=42`)
- [x] 2+ regression models (Linear Regression, Random Forest)
- [x] MAE + RMSE + R² reported for both models
- [x] Feature importance chart for the best model
- [ ] Clean, commented notebook — the model-comparison markdown cell still contains
      an unfilled template placeholder (`[outperformed / performed similarly to]`),
      and the 0.88 correlation it cites is not present in any text output since the
      heatmap renders as an image only
