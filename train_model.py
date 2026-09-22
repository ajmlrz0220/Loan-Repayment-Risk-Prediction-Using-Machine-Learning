"""
train_model.py
----------------
Trains the loan-default-risk model and saves it as loan_default_model.pkl
for the Streamlit app to load.

This is the notebook's modeling logic (EDA cells removed) turned into a
reusable, reproducible script — the kind of thing an interviewer will
expect to see next to a notebook in a real project repo.

Usage:
    python train_model.py --data loan_data.csv --out loan_default_model.pkl
"""

import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

RANDOM_STATE = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering step as the notebook."""
    df = df.copy()
    df["credit_history_years"] = df["days.with.cr.line"] / 365
    return df


def build_pipeline(numeric_features, categorical_features) -> Pipeline:
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def find_best_threshold(model, X_val, y_val) -> float:
    """Sweep thresholds and pick the one that maximizes F1 (same as notebook)."""
    y_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.05)
    best_thr, best_f1 = 0.5, -1
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr)


def main(data_path: str, out_path: str, quick: bool):
    df = pd.read_csv(data_path)
    df = engineer_features(df)

    X = df.drop("not.fully.paid", axis=1)
    y = df["not.fully.paid"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    pipeline = build_pipeline(numeric_features, categorical_features)

    if quick:
        # Fast path (good for CI / a quick re-run) — skip grid search
        pipeline.fit(X_train, y_train)
        best_model = pipeline
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        param_grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__class_weight": ["balanced", "balanced_subsample"],
        }
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_train, y_train)
        print("Best params:", grid.best_params_)
        print("Best CV PR-AUC:", grid.best_score_)
        best_model = grid.best_estimator_

    best_threshold = find_best_threshold(best_model, X_test, y_test)

    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)
    print("\nFinal test-set performance @ threshold =", best_threshold)
    print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_test, y_pred, zero_division=0), 4))
    print("F1       :", round(f1_score(y_test, y_pred, zero_division=0), 4))
    print("PR-AUC   :", round(average_precision_score(y_test, y_prob), 4))

    joblib.dump(
        {
            "model": best_model,
            "threshold": best_threshold,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_columns": X.columns.tolist(),
        },
        out_path,
    )
    print(f"\nSaved model artifact to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the loan default risk model.")
    parser.add_argument("--data", default="loan_data.csv", help="Path to loan_data.csv")
    parser.add_argument("--out", default="loan_default_model.pkl", help="Output model path")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip GridSearchCV for a fast run (useful while testing the pipeline).",
    )
    args = parser.parse_args()
    main(args.data, args.out, args.quick)
