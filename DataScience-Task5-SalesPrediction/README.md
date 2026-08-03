# Task 5 — Sales Prediction Using Python

A regression task with a direct business question behind it: given how much was spent
advertising a product on TV, radio and in newspapers, how many units will sell — and
which channel is actually earning its budget? The dataset is the classic advertising
dataset, small and clean, which makes it a good vehicle for the part that matters:
fitting a linear baseline, reading its coefficients to attribute impact per channel,
then checking whether a non-linear model does better.

## Dataset

- **File:** `Advertising Budget and Sales.csv` (Kaggle advertising dataset)
- **Raw shape:** 200 rows × 5 columns (includes an `Unnamed: 0` index column, dropped)
- **Working shape:** 200 rows × 4 columns
- **Features:** TV Ad Budget ($), Radio Ad Budget ($), Newspaper Ad Budget ($)
- **Target:** Sales ($)
- **Missing values:** none — all 200 rows non-null across every column

| Column | Mean | Std | Min | 25% | 50% | 75% | Max |
|---|---|---|---|---|---|---|---|
| TV Ad Budget ($) | 147.04 | 85.85 | 0.7 | 74.38 | 149.75 | 218.83 | 296.4 |
| Radio Ad Budget ($) | 23.26 | 14.85 | 0.0 | 9.98 | 22.90 | 36.53 | 49.6 |
| Newspaper Ad Budget ($) | 30.55 | 21.78 | 0.3 | 12.75 | 25.75 | 45.10 | 114.0 |
| Sales ($) | 14.02 | 5.22 | 1.6 | 10.38 | 12.90 | 17.40 | 27.0 |

## Tech Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter Notebook

## Approach

1. Loaded the CSV and inspected the shape — 200 × 5.
2. Dropped the redundant `Unnamed: 0` index column, then ran `info()` and
   `describe()` — confirmed 200 non-null rows across all four remaining columns.
3. Plotted a seaborn `pairplot` of all variables and an annotated correlation heatmap.
4. Plotted three side-by-side scatter plots: Sales against TV, Radio and Newspaper
   budget respectively.
5. Split the data 80/20 with `train_test_split(random_state=42)`.
6. Trained a **Linear Regression** baseline and scored MAE, RMSE and R².
7. Extracted and printed the model's coefficients and intercept to quantify each
   channel's per-dollar contribution to sales.
8. Plotted the three coefficients as a horizontal bar chart.
9. Plotted residuals (actual − predicted) against predicted sales for the Linear
   Regression model, with a zero reference line.
10. Wrote a coefficient interpretation and a business recommendation on budget
    allocation.
11. Trained a **Random Forest Regressor** (`n_estimators=100`, `random_state=42`) and
    scored the same three metrics.

## Results

Metrics on the 40-row held-out test set.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 1.4608 | 1.7816 | 0.8994 |
| **Random Forest Regressor** | **0.6203** | **0.7687** | **0.9813** |

Linear Regression coefficients (change in Sales per one-dollar increase in budget):

| Channel | Coefficient |
|---|---|
| Radio Ad Budget ($) | 0.189195 |
| TV Ad Budget ($) | 0.044730 |
| Newspaper Ad Budget ($) | 0.002761 |

Intercept: 2.979067

**Best model: Random Forest Regressor** — MAE down 58%, RMSE down 57%, and R² up
from 0.899 to 0.981 against the linear baseline.

**Correlation matrix values: not recorded.** The heatmap renders as an image only, so
the exact pairwise correlations do not appear in the notebook's text output.

**Visualisations produced:** pairplot, correlation heatmap, three Sales-vs-channel
scatter plots, coefficient bar chart, residual plot.

## Key Findings

- **Radio has by far the largest coefficient per advertising dollar.** At 0.1892,
  radio's marginal effect on sales is roughly 4.2× TV's (0.0447) and 69× newspaper's
  (0.0028). One extra dollar in radio moves sales more than four times as much as the
  same dollar in TV.
- **TV nevertheless dominates total sales volume, because of scale.** Mean TV spend is
  $147.04 against radio's $23.26 — TV budgets are about 6.3× larger. Multiplying
  coefficient by mean spend gives roughly 6.58 from TV versus 4.40 from radio, so TV
  contributes more in absolute terms while radio is the more efficient channel at the
  margin. These are two different questions and they have different answers.
- **Newspaper advertising is effectively dead money.** A coefficient of 0.0028 is
  indistinguishable from zero in practical terms. Reallocating newspaper spend toward
  radio is the clearest actionable conclusion in the dataset.
- **The relationship is not purely linear.** Random Forest cutting MAE by 58% over
  Linear Regression indicates real interaction or saturation effects — most likely a
  TV × radio interaction, or diminishing returns at high spend levels, neither of
  which an additive linear model can express.

> Note on the notebook's own interpretation: the markdown cell concludes that "TV
> advertising has the highest impact on sales", but the coefficient bullet points in
> that cell were left blank and the printed coefficients show radio's coefficient to
> be about four times TV's. The statement holds if "impact" means total contribution
> at current budget levels; it does not hold on a per-dollar basis. That cell should
> be filled in and the distinction made explicit.

## How to Run

Dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn notebook
```

`Advertising Budget and Sales.csv` is committed in this folder, so no download is
needed. Open the notebook and run all cells top to bottom:

```bash
jupyter notebook Sales_Prediction.ipynb
```

## Requirement Checklist

- [x] EDA — null check (`info()` confirms 200 non-null per column) and `describe()`
- [x] EDA — pairplot
- [x] Scatter plots for Sales vs TV, Radio and Newspaper
- [x] Correlation heatmap
- [x] Train/test split (80/20, `random_state=42`)
- [x] Linear Regression baseline
- [x] At least one additional model (Random Forest Regressor)
- [x] MAE + RMSE + R² reported for both models
- [ ] Residual plot **for the best model** — the residual plot was produced for
      Linear Regression, but Random Forest is the better model (R² 0.981 vs 0.899);
      no residual plot exists for it
- [ ] Interpretation of which channel drives sales most — an interpretation cell
      exists, but its coefficient bullet points are blank and its "TV has the highest
      impact" conclusion conflicts with the printed coefficients on a per-dollar basis
- [ ] Clean, commented notebook — the coefficient-interpretation markdown cell has
      unfilled blanks and needs correcting per the note above
