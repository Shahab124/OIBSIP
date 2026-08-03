# OIBSIP — Data Science Internship

Completed tasks for the **Oasis Infobyte Student Internship Programme (OIBSIP)**,
Data Science track. Each folder is a self-contained project: a Jupyter notebook with
the full analysis, the dataset it runs on, and a README documenting the dataset,
method, real measured results and a requirement checklist.

The track required a minimum of **3 tasks**. All **5** were completed.

## Tasks

| # | Project | Key technique | Headline result | Folder |
|---|---|---|---|---|
| 1 | Iris Flower Classification | Multi-class classification — Logistic Regression vs KNN | 100% accuracy for both models on the 30-row test set; Logistic Regression chosen for simplicity | [DataScience-Task1-IrisClassification](DataScience-Task1-IrisClassification) |
| 2 | Unemployment Analysis in India | Time-series EDA and pre/post-event comparison | Mean unemployment rose from 9.51% to 17.77% after March 2020 — an 8.26 pp jump | [DataScience-Task2-UnemploymentAnalysis](DataScience-Task2-UnemploymentAnalysis) |
| 3 | Car Price Prediction | Regression with one-hot encoding — Linear Regression vs Random Forest | Random Forest R² 0.963, MAE ₹0.62 lakh — 44% lower error than the linear baseline | [DataScience-Task3-CarPricePrediction](DataScience-Task3-CarPricePrediction) |
| 4 | Email Spam Detection | NLP — TF-IDF + Multinomial Naive Bayes vs Logistic Regression | Naive Bayes 96.4% accuracy, spam precision 1.00, spam recall 0.73 | [DataScience-Task4-EmailSpamDetection](DataScience-Task4-EmailSpamDetection) |
| 5 | Sales Prediction | Regression + coefficient attribution | Random Forest R² 0.981; radio has the largest per-dollar coefficient (0.189) | [DataScience-Task5-SalesPrediction](DataScience-Task5-SalesPrediction) |

## Tech Stack

**Language:** Python 3

**Data handling:** pandas · NumPy

**Machine learning:** scikit-learn — `LogisticRegression`, `KNeighborsClassifier`,
`MultinomialNB`, `LinearRegression`, `RandomForestRegressor`, `TfidfVectorizer`,
`train_test_split`, metrics module

**NLP:** NLTK (English stopword corpus) · WordCloud

**Visualisation:** matplotlib · seaborn

**Environment:** Jupyter Notebook

Install everything needed across all five tasks:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud notebook
```

## Repository Structure

```
OIBSIP/
├── README.md
├── DataScience-Task1-IrisClassification/
│   ├── README.md
│   └── Iris_Classification.ipynb          (dataset loads from scikit-learn)
├── DataScience-Task2-UnemploymentAnalysis/
│   ├── README.md
│   ├── Unemployment_Analysis.ipynb
│   ├── Unemployment in India.csv
│   └── Unemployment_Rate_upto_11_2020.csv
├── DataScience-Task3-CarPricePrediction/
│   ├── README.md
│   ├── Car_Price_Prediction.ipynb
│   ├── car data.csv                       (the file the notebook uses)
│   ├── CAR DETAILS FROM CAR DEKHO.csv
│   ├── Car details v3.csv
│   └── car details v4.csv
├── DataScience-Task4-EmailSpamDetection/
│   ├── README.md
│   ├── Spam_Detection.ipynb
│   └── spam.csv
└── DataScience-Task5-SalesPrediction/
    ├── README.md
    ├── Sales_Prediction.ipynb
    └── Advertising Budget and Sales.csv
```

## Running the Notebooks

Every dataset except Iris is committed alongside its notebook, so each task runs
without any download. Clone the repo, install the dependencies above, and open any
notebook:

```bash
jupyter notebook
```

Task 4 additionally calls `nltk.download('stopwords')` on first run, which needs an
internet connection once.

## A Note on Outputs

All charts, tables and metrics are saved inside the notebooks themselves as stored
cell outputs, so they are visible on GitHub without re-running anything. No separate
screenshot or image files are committed in any of the five folders.
