# Task 2 — Unemployment Analysis with Python

An exploratory data analysis of unemployment across Indian states, built around one
central question: how much did the COVID-19 lockdown actually move the numbers? This
task is analysis rather than prediction — no model is trained. The work involves
cleaning a real-world government-survey dataset, engineering a proper date column,
comparing states against each other, plotting unemployment over time, and quantifying
the gap between the pre-March-2020 and post-March-2020 periods.

## Dataset

- **Primary file:** `Unemployment in India.csv` — CMIE / Kaggle unemployment survey data
- **Raw shape:** 768 rows × 7 columns
- **Cleaned shape:** 740 rows × 7 columns (28 fully blank rows dropped)
- **Columns:** Region, Date, Frequency, Estimated Unemployment Rate (%),
  Estimated Employed, Estimated Labour Participation Rate (%), Area (Rural/Urban)
- **Coverage:** 28 states/UTs, monthly observations from May 2019 through mid-2020
- **Second file:** `Unemployment_Rate_upto_11_2020.csv` (17.8 KB, includes
  latitude/longitude) — loaded and its columns printed, but not used in any analysis

Note: every column header in the raw CSV carried a leading space, and 28 rows were
entirely null. Both were handled before analysis.

## Tech Stack

Python · pandas · matplotlib · seaborn · Jupyter Notebook

## Approach

1. Loaded both CSV files and printed their column lists to compare their structure.
2. Stripped leading/trailing whitespace from all column names in both DataFrames.
3. Converted the `Date` column to proper `datetime64` using `format='%d-%m-%Y'`.
4. Ran a null check — found 28 nulls in every column (28 completely empty rows).
5. Dropped those rows with `dropna()`, leaving 740 clean records.
6. Engineered `Month` and `Month_Name` columns from the datetime column.
7. Grouped by `Region` and computed the mean unemployment rate per state, sorted
   descending.
8. Plotted a multi-line time series of unemployment rate over time for three states
   (Uttar Pradesh, Bihar, Tripura).
9. Plotted a bar chart of the top 10 states by average unemployment rate.
10. Built a correlation heatmap across the three numeric columns (unemployment rate,
    estimated employed, labour participation rate).
11. Split the data on `2020-03-01` and compared mean unemployment rate before versus
    after, then visualised the two means as a bar chart.
12. Wrote a conclusion with three policy-oriented recommendations.

## Results

This task produces descriptive statistics rather than model metrics.

| Measure | Value |
|---|---|
| Rows before cleaning | 768 |
| Rows after cleaning | 740 |
| Null rows removed | 28 |
| States / regions covered | 28 |
| Pre-COVID mean unemployment rate (before 2020-03-01) | 9.5095% |
| Post-COVID mean unemployment rate (from 2020-03-01) | 17.7744% |
| Absolute increase | 8.2648 percentage points |

Top 10 states by average unemployment rate:

| Rank | State | Avg unemployment rate (%) |
|---|---|---|
| 1 | Tripura | 28.35 |
| 2 | Haryana | 26.28 |
| 3 | Jharkhand | 20.59 |
| 4 | Bihar | 18.92 |
| 5 | Himachal Pradesh | 18.54 |
| 6 | Delhi | 16.50 |
| 7 | Jammu & Kashmir | 16.19 |
| 8 | Chandigarh | 15.99 |
| 9 | Rajasthan | 14.06 |
| 10 | Uttar Pradesh | 12.55 |

Lowest five: Meghalaya (4.80), Odisha (5.66), Assam (6.43), Uttarakhand (6.58),
Gujarat (6.66).

**Correlation values: not recorded as text.** The heatmap renders as an image only,
so the exact correlation matrix does not appear in the notebook's text output. The
notebook's own conclusion cell cites a correlation of **-0.22** between unemployment
rate and estimated employed, and **0.00** between unemployment rate and labour
participation rate.

**Visualisations produced:** multi-state time-series line chart, top-10 bar chart,
3×3 correlation heatmap, pre-vs-post-COVID comparison bar chart.

## Key Findings

- **Unemployment nearly doubled after the lockdown.** The mean rate went from 9.51%
  before March 2020 to 17.77% after — an 8.26 percentage-point jump, an increase of
  roughly 87% in relative terms. This is a step change, not a trend.
- **Regional disparity is enormous and predates COVID.** Tripura's average rate
  (28.35%) is nearly six times Meghalaya's (4.80%). This spread exists across the
  whole period, meaning it is a structural feature of the labour market rather than
  a pandemic artifact.
- **Unemployment is only weakly tied to the other two measures.** Per the notebook's
  conclusion, the correlation with estimated employed is -0.22 and with labour
  participation rate is 0.00. Unemployment here is driven by discrete shocks and
  regional factors rather than smooth linear relationships — which argues for
  event-based monitoring over trend extrapolation.
- **The data needed real cleaning before it could be trusted.** 28 of 768 rows
  (3.6%) were completely empty and every column name had stray whitespace. Skipping
  either step would have broken the groupby and the datetime conversion.

## How to Run

Dependencies:

```bash
pip install pandas matplotlib seaborn notebook
```

Both CSV files are committed in this folder, so the notebook runs without any
download. Launch it and run all cells in order:

```bash
jupyter notebook Unemployment_Analysis.ipynb
```

## Requirement Checklist

- [x] Dataset loaded, shape checked, null check performed
- [x] Type conversion — `Date` converted to `datetime64`, column names stripped
- [x] Region-wise EDA — mean unemployment rate per state, sorted
- [ ] Month-wise EDA — `Month` and `Month_Name` columns were created but never used
      in any grouping, chart or aggregation
- [x] Time-series line chart for 3+ states (Uttar Pradesh, Bihar, Tripura)
- [x] Bar chart of top 10 states by average unemployment rate
- [x] Correlation heatmap (unemployment / employment / labour participation)
- [x] Pre-COVID vs post-COVID comparison (statistics + bar chart)
- [ ] Written observations between charts — observation cells exist, but the
      correlation-analysis cell still contains unfilled template placeholders
      (`[positive/negative]`, `[value]`), and the regional-analysis cell states
      that Uttar Pradesh has the lowest average rate when the computed output shows
      Meghalaya lowest at 4.80% and Uttar Pradesh tenth-highest at 12.55%
- [ ] Clean, commented notebook — same two issues above need resolving; the
      pre/post-COVID cell also rounds 17.77% to "17.5%" and the 8.26 pp gap to "8"
