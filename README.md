# Machine Learning Assignments (UML501)

This repository contains solutions for 8 Machine Learning lab assignments covering various ML concepts and techniques.

## Repository Structure

```
ml_assignments/
├── Assignment_1_NumPy/
├── Assignment_2/
├── Assignment_3/
├── Assignment_4_Webscraping/
├── Assignment_5_Regression/
├── Assignment_6_NaiveBayes_KNN/
├── Assignment_7_Adaboost/
└── Assignment_8_SVM/
```

## Assignments Overview

### Assignment 1 - NumPy Operations
**Topics**: Basic NumPy array operations, mathematics, statistics, searching & sorting

| Question | Description |
|----------|-------------|
| Q1 (a-h) | Array operations: reverse, flatten, compare, frequency, matrix sums, eigen values, multiplication, inner/outer products |
| Q2 (a-b) | Math & Stats: absolute values, percentiles, mean, median, std, floor, ceil, truncate, round |
| Q3 (a-b) | Searching & Sorting: sort, argsort, smallest/largest elements, integer/float filtering |
| Q4 (a-b) | Image to array conversion and loading |

---

### Assignment 2 - Data Preprocessing & Feature Engineering
**Dataset**: Microsoft Adventure Works Cycles Customer Data

| Part | Description |
|------|-------------|
| Part I | Feature selection, cleaning, and data type classification (Nominal, Ordinal, Interval, Ratio) |
| Part II | Data transformation: null handling, normalization, discretization, standardization, one-hot encoding |
| Part III | Similarity measures (Simple Matching, Jaccard, Cosine) and correlation analysis |

---

### Assignment 3 - Multiple Linear Regression
**Datasets**: USA Housing, Car Price Prediction (imports-85)

| Question | Description |
|----------|-------------|
| Q1 | K-Fold Cross Validation for Multiple Linear Regression using Least Square Error Fit |
| Q2 | Validation set approach with Gradient Descent Optimization (learning rates: 0.001, 0.01, 0.1, 1) |
| Q3 | Data preprocessing and Linear Regression with PCA dimensionality reduction |

---

### Assignment 4 - Web Scraping
**Tools**: BeautifulSoup, Requests, Selenium

| Question | Website | Data Extracted |
|----------|---------|----------------|
| Q1 | books.toscrape.com | Title, Price, Availability, Star Rating (with pagination) |
| Q2 | IMDB Top 250 | Rank, Movie Title, Year, Rating (using Selenium) |
| Q3 | timeanddate.com | City, Temperature, Weather Condition |

---

### Assignment 5 - Regression Techniques
**Datasets**: Synthetic data, Hitters, California Housing, Iris

| Question | Description |
|----------|-------------|
| Q1 | Ridge Regression using Gradient Descent with hyperparameter tuning |
| Q2 | Linear vs Ridge vs LASSO regression comparison on Hitters dataset |
| Q3 | RidgeCV and LassoCV for automatic alpha selection |
| Q4 | Multiclass Logistic Regression using One-vs-Rest strategy (from scratch) |

---

### Assignment 6 - Naive Bayes & K-NN
**Dataset**: Iris

| Question | Description |
|----------|-------------|
| Q1 | Gaussian Naive Bayes: (i) Step-by-step implementation from scratch, (ii) sklearn implementation |
| Q2 | GridSearchCV for K-NN hyperparameter tuning (n_neighbors, weights, metric) |

---

### Assignment 7 - AdaBoost
**Datasets**: SMS Spam Collection, Heart Disease (UCI), WISDM Activity Recognition

| Question | Dataset | Parts |
|----------|---------|-------|
| Q1 | SMS Spam | A: Preprocessing, B: Decision Stump baseline, C: Manual AdaBoost (T=15), D: sklearn AdaBoost |
| Q2 | Heart Disease | A: Baseline, B: Hyperparameter tuning, C: Misclassification analysis, D: Feature importance |
| Q3 | WISDM Activity | A: Data prep, B: Weak classifier, C: Manual AdaBoost (T=20), D: sklearn comparison |

---

### Assignment 8 - Support Vector Machines (SVM)
**Datasets**: Iris, Breast Cancer

| Question | Description |
|----------|-------------|
| Q1 | SVM with different kernels (Linear, Polynomial, RBF) - Accuracy, Precision, Recall, F1-Score |
| Q2 | Effect of feature scaling on SVM performance (with vs without StandardScaler) |

---

## Technologies Used

- Python 3.x
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- BeautifulSoup, Selenium
- Jupyter Notebook

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Arijit2772-dev/ml_assignments.git
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn beautifulsoup4 selenium
   ```

3. Open any notebook:
   ```bash
   jupyter notebook
   ```

## Author

Arijit Singh
