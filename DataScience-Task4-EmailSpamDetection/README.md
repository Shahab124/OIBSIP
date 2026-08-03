# Task 4 — Email Spam Detection with Machine Learning

A natural-language binary classification problem: given the raw text of an SMS
message, decide whether it is spam or legitimate ("ham"). Text cannot be fed to a
classifier directly, so the core of this task is the pipeline that turns messages
into numbers — cleaning, stopword removal and TF-IDF vectorisation — followed by
training two classifiers and comparing them. Because the classes are heavily
imbalanced (87% ham), accuracy alone is a misleading score, so the evaluation focuses
on precision and recall for the spam class specifically.

## Dataset

- **Source:** SMS Spam Collection Dataset (Kaggle / UCI), file `spam.csv`,
  read with `encoding='latin-1'`
- **Raw shape:** 5,572 rows × 5 columns (three trailing unnamed columns are empty)
- **Used shape:** 5,572 rows × 2 columns after keeping `v1`/`v2` and renaming them
  to `label` and `message`
- **Class distribution:** ham 86.59% · spam 13.41%

| Class | Share | Test-set support |
|---|---|---|
| ham | 86.59% | 966 |
| spam | 13.41% | 149 |

After TF-IDF vectorisation the feature matrix is **5,572 × 8,389** — one column per
distinct token in the cleaned corpus.

## Tech Stack

Python · pandas · scikit-learn · NLTK · WordCloud · matplotlib · Jupyter Notebook

## Approach

1. Loaded `spam.csv` with latin-1 encoding and inspected the shape — 5,572 × 5.
2. Kept only the two meaningful columns (`v1`, `v2`) and renamed them to `label` and
   `message`, discarding the three empty unnamed columns.
3. Checked class distribution with `value_counts(normalize=True)` — 86.59% ham,
   13.41% spam, confirming significant imbalance.
4. Wrote a `clean_text()` function that lowercases the message, strips punctuation
   and digits with a regex (`[^a-z\s]`), splits into tokens, and removes English
   stopwords from the NLTK corpus.
5. Applied the cleaner to every message, storing results in a `clean_message` column.
6. Vectorised the cleaned text with `TfidfVectorizer()` → 8,389 features.
7. Split 80/20 with `stratify=y` to preserve the class ratio in both halves →
   4,457 training rows, 1,115 test rows.
8. Trained a **Multinomial Naive Bayes** classifier and printed accuracy plus a full
   classification report.
9. Wrote a markdown analysis of why recall is the metric that matters most for spam.
10. Trained a **Logistic Regression** classifier (`max_iter=1000`) as the alternative
    and scored it identically.
11. Compared both models in a markdown table and justified the final choice.
12. Generated side-by-side WordClouds of the most frequent words in spam versus ham.

## Results

Metrics on the 1,115-row stratified test set.

| Model | Accuracy | Spam precision | Spam recall | Spam F1 | Ham precision | Ham recall | Ham F1 |
|---|---|---|---|---|---|---|---|
| **Multinomial Naive Bayes** | **0.9641** | **1.00** | **0.73** | **0.84** | 0.96 | 1.00 | 0.98 |
| Logistic Regression | 0.9570 | 0.99 | 0.68 | 0.81 | 0.95 | 1.00 | 0.98 |

Macro averages — Naive Bayes: precision 0.98, recall 0.87, F1 0.91.
Logistic Regression: precision 0.97, recall 0.84, F1 0.89.

**Confusion matrix: not recorded.** `confusion_matrix` is imported in the notebook
but never called or displayed for either model. The per-class figures above come from
`classification_report`.

**Best model: Multinomial Naive Bayes.** Both models achieve near-perfect spam
precision (1.00 vs 0.99), so they are effectively tied on false alarms. Naive Bayes
wins on the metric that matters — it catches 73% of actual spam against Logistic
Regression's 68%.

**Visualisations produced:** side-by-side WordClouds for spam and ham vocabulary.

## Key Findings

- **Recall is the decisive metric, not accuracy.** A missed spam message means a
  phishing or scam text lands in the user's inbox — a genuine security risk. A false
  positive just means one real message goes to a spam folder the user can check.
  Spam filters therefore deliberately trade precision for recall.
- **Both models are conservative to a fault.** Precision on spam is 1.00 and 0.99 —
  essentially zero false alarms — but recall is only 0.73 and 0.68. Roughly a
  quarter to a third of spam gets through. With 13.4% spam prevalence, a classifier
  that labelled everything as ham would still score 86.6% accuracy, which is why the
  96% headline accuracy overstates how well this pipeline actually performs.
- **Naive Bayes suits high-dimensional sparse text.** With 8,389 features and 4,457
  training rows, the independence assumption that would be a liability elsewhere
  becomes an advantage — it needs far fewer examples per feature than a
  discriminative model.
- **Stratified splitting was necessary, not optional.** With only 149 spam messages
  in the test set, an unstratified random split could easily have skewed the spam
  proportion enough to make the recall figures unstable.

## How to Run

Dependencies:

```bash
pip install pandas scikit-learn nltk wordcloud matplotlib notebook
```

The notebook calls `nltk.download('stopwords')` on first run, so an internet
connection is needed once. `spam.csv` is committed in this folder. Open the notebook
and run all cells in order:

```bash
jupyter notebook Spam_Detection.ipynb
```

## Requirement Checklist

- [x] Class distribution check (86.59% ham / 13.41% spam)
- [x] Text preprocessing — lowercasing
- [x] Text preprocessing — punctuation and digit removal
- [x] Text preprocessing — English stopword removal via NLTK
- [ ] Stemming (optional bonus) — not applied
- [x] TF-IDF vectorizer applied (8,389 features)
- [ ] Written explanation of what TF-IDF actually measures — the vectoriser is
      applied in code with no accompanying markdown explaining term frequency /
      inverse document frequency
- [x] Train/test split (80/20, stratified)
- [x] Multinomial Naive Bayes trained
- [x] One alternative model trained (Logistic Regression)
- [x] Accuracy, precision, recall and F1 reported for both models
- [ ] Confusion matrix — imported but never called or displayed
- [x] Discussion of why recall matters for spam detection
- [x] Bonus: WordClouds for spam and ham
- [x] Clean, commented notebook with markdown analysis
