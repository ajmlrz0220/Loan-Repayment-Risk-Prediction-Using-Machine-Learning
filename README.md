# 💳 Loan Default Risk Predictor

A machine learning project that predicts whether a borrower is likely to
**not fully repay** a loan, using historical LendingClub-style loan data.
Includes full EDA, model comparison, hyperparameter tuning, threshold
optimization, and a deployed **Streamlit** app for interactive predictions.

**🔗 Live demo:** _add your Streamlit Cloud link here after deploying_

---

## Overview

| | |
|---|---|
| **Problem type of my project** | Binary classification (imbalanced target) |
| **Target** | `not.fully.paid` |
| **Models compared** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting |
| **Final model** | Random Forest, tuned via `GridSearchCV` (5-fold stratified CV) |
| **Metric optimized** | PR-AUC (better than accuracy/ROC-AUC for imbalanced targets) |
| **Extra step** | Threshold tuning — picked the probability cutoff that maximizes F1 instead of using the default 0.5 |

## Project structure

```
loan-default-risk/
├── loan_default_prediction.ipynb   # Full EDA + modeling notebook
├── train_model.py                  # Script version — reproducible training, saves the model
├── app.py                          # Streamlit UI (single + batch prediction)
├── requirements.txt
├── .gitignore
└── README.md
```

## Key results

_Fill these in from your notebook's final cell output before publishing:_

- Accuracy: `__`
- Precision: `__`
- Recall: `__`
- F1: `__`
- ROC-AUC: `__`
- PR-AUC: `__`

Top predictive features (from feature importances): FICO score, interest
rate, revolving utilization, debt-to-income ratio, and loan purpose.

## Running locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/loan-default-risk.git
cd loan-default-risk

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (needs loan_data.csv — see "Getting the data" below)
python train_model.py --data loan_data.csv --out loan_default_model.pkl

# 5. Launch the app
streamlit run app.py
```

## Getting the data

The dataset (`loan_data.csv`) is the classic LendingClub 2007–2010 subset
used in many ML courses. It's not committed to this repo (see `.gitignore`)
to keep it lightweight — download it from Kaggle ("LendingClub Loan Data"
or "Loan Data" datasets) and place it in the project root before running
`train_model.py`.

## Deploying the app (Streamlit Community Cloud — free)

1. Push this repo to GitHub (steps below).
2. Since `loan_default_model.pkl` is git-ignored, either:
   - remove `*.pkl` from `.gitignore` and commit the trained model (simplest
     for a portfolio project — the file is small), **or**
   - add a small startup step that trains the model on first run.
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, pick this repo/branch, set the main file to
   `app.py`, and deploy.
4. Copy the live URL into this README and your resume/LinkedIn.

## Notes

This is a portfolio/demo project built for learning and interview
discussion — not for real lending decisions.
