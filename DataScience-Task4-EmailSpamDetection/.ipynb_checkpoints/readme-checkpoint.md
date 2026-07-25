# Email Spam Detection with Machine Learning

## Objective
Build an NLP binary classifier that distinguishes spam messages from legitimate (ham) messages.

## Dataset
SMS Spam Collection Dataset (Kaggle, uciml) — 5,572 messages, ~87% ham / ~13% spam.

## Approach
1. Cleaned text: lowercased, removed punctuation/numbers, removed English stopwords.
2. Converted text to numeric features using TF-IDF Vectorization.
3. Split data 80/20 (stratified to preserve class balance).
4. Trained two classifiers: Multinomial Naive Bayes and Logistic Regression.
5. Evaluated using accuracy, precision, recall, F1-score, and confusion matrix — 
   with particular focus on spam recall given the class imbalance.

## Results
| Model | Accuracy | Spam Precision | Spam Recall |
|---|---|---|---|
| Naive Bayes | 96.4% | 1.00 | 0.73 |
| Logistic Regression | 95.7% | 0.99 | 0.68 |

Naive Bayes selected as final model for higher spam recall.

## Tech Stack
Python, pandas, scikit-learn, NLTK, WordCloud, Jupyter Notebook