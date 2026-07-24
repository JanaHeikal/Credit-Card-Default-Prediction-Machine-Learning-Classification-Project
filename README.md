# Credit Card Default Prediction

A binary classification study predicting whether a credit card customer will default on their next monthly payment, using the Taiwan Credit Card Default dataset. Three model families — Logistic Regression, Decision Tree, and k-Nearest Neighbours — are trained, tuned, and rigorously compared on a held-out test set.

**Course:** CSE374 — Machine Learning and Pattern Recognition
**Institution:** Ain Shams University, Faculty of Computer Science Engineering

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Getting Started](#getting-started)
- [Team](#team)
- [AI Usage Disclosure](#ai-usage-disclosure)

---

## Overview

Financial institutions need to assess credit risk before extending or renewing credit lines. This project frames that problem as **binary classification**: given a customer's demographic profile, credit history, and recent payment behavior, predict whether they will default on their next payment.

The study places particular emphasis on:
- Rigorous, leakage-free preprocessing (train/validation/test splitting *before* scaling)
- Addressing class imbalance (~22% default rate) through appropriate model configuration and metric selection
- Systematic hyperparameter tuning via `GridSearchCV`
- Evaluation that goes beyond accuracy to surface the *business cost* of misclassification errors

## Dataset

| Property | Detail |
|---|---|
| Source | [UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| Origin | National Cheng Kung University, Taiwan |
| Instances | 30,000 |
| Features | 23 |
| Target | `default.payment.next.month` (renamed to `default`) |
| Class balance | 22.1% default / 77.9% no default |
| Time period | April – September 2005 |
| Missing values | None |

## Project Structure

```
.
├── CSE374_Final_Notebook.ipynb   # Full analysis: preprocessing, EDA, modeling, evaluation
├── CSE374_Final_Report.pdf       # Written report with detailed discussion and figures
├── default_of_credit_card_clients.xls  # Raw dataset (UCI source)
└── README.md
```

## Methodology

The pipeline follows a strict sequence — **inspection → cleaning → splitting → scaling** — to prevent data leakage:

**1. Data Cleaning**
- Dropped the `ID` column (sequential index with no predictive value; risk of leakage across splits)
- Merged undocumented categorical codes into "other" categories: `EDUCATION` (values 0, 5, 6 → 468 rows) and `MARRIAGE` (value 0 → 54 rows)

**2. Train / Validation / Test Split**

| Split | Rows | Default Rate | Purpose |
|---|---|---|---|
| Train | 21,012 (70%) | 22.1% | Model fitting |
| Validation | 4,488 (15%) | 22.1% | Hyperparameter tuning & model comparison |
| Test | 4,500 (15%) | 22.1% | Final, untouched evaluation |

Stratified splitting preserved the default rate across all three sets.

**3. Feature Scaling**
`StandardScaler` was fit exclusively on the training set and applied to validation/test sets, excluding the three categorical columns (`SEX`, `EDUCATION`, `MARRIAGE`).

**4. Exploratory Data Analysis**
Four visualizations informed modeling decisions:
- **Class distribution** — confirmed the imbalance and motivated `class_weight="balanced"` plus F1/ROC-AUC as primary metrics
- **Default rate by payment status (`PAY_0`)** — identified `PAY_0` as the single strongest predictor (10% default rate for on-time payers vs. >60% for customers with 2+ month delays)
- **Default rate by credit limit** — revealed a monotonic relationship between credit limit and default risk
- **Correlation heatmap** — surfaced multicollinearity among the six `BILL_AMT` columns (0.7–0.9 correlation)

**5. Modeling**

| Model | Key Configuration |
|---|---|
| Logistic Regression | `class_weight="balanced"`, `max_iter=1000`, `lbfgs` solver |
| Decision Tree | Tuned via `GridSearchCV` (5-fold CV, `roc_auc` scoring, 60 combinations / 300 fits) |
| k-Nearest Neighbours | Tuned via `GridSearchCV` (28 combinations / 140 fits); no `class_weight` support, so imbalance was addressed by tuning `k` |

**Decision Tree — best hyperparameters:** `max_depth=5`, `min_samples_split=50`, `min_samples_leaf=5` (CV ROC-AUC: 0.7604)

**kNN — best hyperparameters:** `n_neighbors=31`, `weights=uniform`, `metric=manhattan` (CV ROC-AUC: 0.7546)

## Results

Final performance on the held-out test set (4,500 rows, untouched during development):

| Model | Accuracy | F1-Score | ROC-AUC | False Negatives (missed defaults) |
|---|---|---|---|---|
| Logistic Regression | 0.6851 | 0.4679 | 0.7107 | 372 |
| **Decision Tree (Tuned)** | **0.7118** | **0.5085** | **0.7544** | **324** |
| kNN (Tuned) | 0.8093 | 0.4123 | 0.7481 | 694 |

**Recommended model: Decision Tree (Tuned)** — best F1-Score, best ROC-AUC, and fewest missed defaults among the three, despite not having the highest raw accuracy.

## Key Findings

- **Accuracy is misleading under class imbalance.** kNN achieved the highest accuracy (80.9%) but missed 694 of 995 actual defaulters — a 70% miss rate on the class that matters most to a lender. Its accuracy is driven almost entirely by correctly classifying the majority (non-default) class.
- **`PAY_0` (most recent payment status) is the strongest single predictor**, confirmed both by EDA and the correlation heatmap.
- **Multicollinearity among `BILL_AMT` columns** (0.7–0.9 correlation) does not affect tree-based or instance-based models but limits the interpretability and reliability of Logistic Regression coefficients.
- **Error analysis of the Decision Tree** shows it tends to miss defaulters who *appear* financially healthy in the most recent month (recent on-time payments, higher credit limits, larger recent payments) — a pattern consistent with the dataset's limited six-month observation window.

## Getting Started

**Requirements:**
```
python >= 3.9
pandas
numpy
scikit-learn
matplotlib
seaborn
xlrd          # required to read the .xls dataset file
jupyter
```

**Install dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xlrd jupyter
```

**Run the analysis:**
```bash
jupyter notebook CSE374_Final_Notebook.ipynb
```

## Team

| Name | Role | Student ID |
|---|---|---|
| Jana Sameh Hamed | Data & Preprocessing Lead | 2301019 |
| Marwa Mohamed Hassan | EDA & Visualization Lead | 2301138 |
| Khaled Mohamed Hussein | Modeling & Evaluation Lead | 2301037 |

## AI Usage Disclosure

AI tools were used to clarify unfamiliar concepts and assist in debugging, with all design decisions, model selection, and result interpretation performed independently by the team and verified against actual code output. Specific uses included:

- Explaining how `GridSearchCV` performs cross-validated hyperparameter search, which informed the choice of parameter ranges for Decision Tree tuning
- Debugging a `ValueError` caused by an incorrect `stratify` argument in the second `train_test_split` call
- Clarifying the "accuracy paradox" in imbalanced classification, which the team then verified directly against confusion matrix results

Full details are available in Section 10 of the accompanying report (`CSE374_Final_Report.pdf`).

---

*This project was developed as coursework for CSE374 (Machine Learning and Pattern Recognition) at Ain Shams University.*
