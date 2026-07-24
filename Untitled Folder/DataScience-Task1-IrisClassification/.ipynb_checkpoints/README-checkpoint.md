# Iris Flower Classification

## Objective
Train a machine learning classification model to identify the species of an iris 
flower (Setosa, Versicolor, or Virginica) from its physical measurements.

## Dataset
Built-in Iris dataset from scikit-learn (150 samples, 4 features, 3 balanced classes).

## Approach
1. Loaded and explored the dataset (EDA) — confirmed no missing values, perfectly 
   balanced classes (50 per species).
2. Visualized feature relationships using a Seaborn pairplot — found petal length 
   and petal width to be the most discriminative features; setosa is linearly 
   separable from the other two species.
3. Split data 80/20 into train/test sets.
4. Trained two classifiers: Logistic Regression and K-Nearest Neighbors (k=5).
5. Evaluated both using accuracy, confusion matrix, and classification report.

## Results
Both models achieved 100% accuracy on the test set. Logistic Regression was 
selected as the final model for its simplicity and interpretability.

## Tech Stack
Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter Notebook