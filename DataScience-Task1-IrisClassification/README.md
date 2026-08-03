# Task 1 — Iris Flower Classification

The classic starting problem in supervised machine learning: given four physical
measurements of an iris flower — sepal length, sepal width, petal length and petal
width — predict which of three species it belongs to (setosa, versicolor or
virginica). The goal here was to run a full, honest classification workflow end to
end: load the data, explore it visually, split it into training and test sets, train
more than one classifier, and compare them on real evaluation metrics rather than
picking a favourite by intuition.

## Dataset

- **Source:** built-in scikit-learn dataset, loaded with `sklearn.datasets.load_iris()`
- **Shape:** 150 rows × 5 columns (4 numeric features + species label)
- **Missing values:** none — all 150 entries non-null across every column
- **Class balance:** perfectly balanced — 50 setosa, 50 versicolor, 50 virginica

| Feature | Mean | Std | Min | Max |
|---|---|---|---|---|
| sepal length (cm) | 5.843 | 0.828 | 4.3 | 7.9 |
| sepal width (cm) | 3.057 | 0.436 | 2.0 | 4.4 |
| petal length (cm) | 3.758 | 1.765 | 1.0 | 6.9 |
| petal width (cm) | 1.199 | 0.762 | 0.1 | 2.5 |

## Tech Stack

Python · pandas · scikit-learn · matplotlib · seaborn · Jupyter Notebook

## Approach

1. Loaded the Iris dataset via `load_iris()` and converted it into a pandas
   DataFrame, mapping the numeric target (0/1/2) to readable species names.
2. Ran EDA — `shape`, `info()` (dtypes and non-null counts) and `describe()` —
   confirming 150 complete rows with no missing data.
3. Checked class balance with `value_counts()` — 50 samples per species.
4. Plotted a seaborn `pairplot` coloured by species to inspect how the four
   features separate the classes.
5. Split the data 80/20 with `train_test_split(random_state=42)` → 120 training
   rows, 30 test rows.
6. Trained a **Logistic Regression** classifier (`max_iter=200`).
7. Trained a **K-Nearest Neighbors** classifier (`n_neighbors=5`).
8. Evaluated Logistic Regression with accuracy, a confusion matrix (printed and
   drawn as a seaborn heatmap) and a full classification report.
9. Compared the two models in a markdown cell and declared a preferred model.

## Results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| K-Nearest Neighbors (k=5) | 1.00 | not recorded | not recorded | not recorded |

Per-class results for Logistic Regression on the 30-row test set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| setosa | 1.00 | 1.00 | 1.00 | 10 |
| versicolor | 1.00 | 1.00 | 1.00 | 9 |
| virginica | 1.00 | 1.00 | 1.00 | 11 |

Confusion matrix (Logistic Regression), rows = actual, columns = predicted:

```
              setosa  versicolor  virginica
setosa           10        0          0
versicolor        0        9          0
virginica         0        0         11
```

Zero misclassifications on the held-out set.

> A confusion matrix and classification report were produced for Logistic
> Regression only. For KNN, accuracy alone was recorded.

**Best model: Logistic Regression.** Both classifiers scored identically (100%
accuracy), so the tie was broken on practical grounds: Logistic Regression is
computationally simpler and more interpretable, whereas KNN must store the entire
training set and recompute distances for every single prediction.

## Key Findings

- **Setosa is trivially separable.** The pairplot shows it forming a completely
  isolated cluster across nearly every feature combination — no model needs to work
  hard to identify it.
- **Petal measurements carry the discriminative signal.** Versicolor and virginica
  overlap noticeably on sepal dimensions but separate cleanly on petal length and
  petal width. This is also visible in the summary statistics: petal length has a
  standard deviation of 1.77 cm across a 1.0–6.9 cm range, far more spread than
  sepal width's 0.44 cm.
- **100% accuracy here is a property of the dataset, not evidence of a great
  model.** With only 30 test samples and near-linearly-separable classes, a perfect
  score is the expected outcome. It should not be read as a claim that this pipeline
  would generalise to harder classification problems.
- **Model choice was decided by cost, not by score.** When two models are tied on
  every metric, the correct tiebreaker is simplicity and inference cost.

## How to Run

Dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn notebook
```

The dataset ships with scikit-learn, so there is nothing to download. Open the
notebook and run all cells top to bottom:

```bash
jupyter notebook Iris_Classification.ipynb
```

## Requirement Checklist

- [x] Dataset loaded via `sklearn.datasets.load_iris()`
- [x] EDA — shape, dtypes, null check, `describe()`
- [x] Pairplot / scatter matrix
- [ ] Box plots — not produced; the pairplot is the only distribution visual
- [x] Feature selection discussion — markdown cell identifies petal length/width as
      the most discriminative features
- [x] `train_test_split` used (80/20, `random_state=42`)
- [x] At least 2 classifiers trained (Logistic Regression, KNN)
- [ ] Accuracy + confusion matrix + classification report **for each** model —
      full evaluation done for Logistic Regression; KNN has accuracy only
- [x] Best model declared with justification
- [x] Clean, commented notebook with markdown observations
