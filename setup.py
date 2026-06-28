"""
Usage:
    python setup.py [--data Data/accepted.csv] [--n_jobs -1]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    DATA_RAW, DATA_PROCESSED_DIR, PARAMS_FILE,
    MINORITY_FRACS, RANDOM_STATE, KNN_IMPUTER_NEIGHBORS,
)
from src.preprocessing import load_and_clean, engineer_features, create_imbalanced_dataset


PARAM_GRIDS = {
    "XGBClassifier": {
        "clf__n_estimators":     [400, 800],
        "clf__max_depth":        [4, 6],
        "clf__learning_rate":    [0.05, 0.1],
        "clf__min_child_weight": [1, 10],
        "clf__reg_lambda":       [1, 5],
        "clf__subsample":        [0.8],
        "clf__colsample_bytree": [0.8],
        "clf__random_state":     [RANDOM_STATE],
        "clf__eval_metric":      ["logloss"],
    },
    "LogisticRegression": {
        "clf__C":          [0.01, 0.1, 1, 10],
        "clf__penalty":    ["l1", "l2"],
        "clf__solver":     ["saga"],
        "clf__max_iter":   [2000],
        "clf__random_state": [RANDOM_STATE],
    },
    "RandomForestClassifier": {
        "clf__n_estimators":     [500],
        "clf__max_depth":        [None, 16, 24],
        "clf__min_samples_leaf": [1, 5, 20],
        "clf__max_features":     ["sqrt", 0.3],
        "clf__random_state":     [RANDOM_STATE],
    },
}

ESTIMATORS = {
    "XGBClassifier": XGBClassifier(),
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(),
}


def preprocess_and_save(data_path: str):
    print(f"Loading {data_path} ...")
    df_raw = load_and_clean(data_path)
    df = engineer_features(df_raw)
    print(f"  Cleaned shape: {df.shape}")

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    for frac in MINORITY_FRACS:
        print(f"  Creating imbalance fraction {frac} ...")
        X, y, bias = create_imbalanced_dataset(df, frac)
        print(f"    Shape: {X.shape}, minority rate: {y.mean():.3f}")

        X.to_parquet(os.path.join(DATA_PROCESSED_DIR, f"X_full_{frac}.parquet"), index=False)
        y.to_frame("target").to_parquet(os.path.join(DATA_PROCESSED_DIR, f"y_full_{frac}.parquet"), index=False)

        with open(os.path.join(DATA_PROCESSED_DIR, f"bias_{frac}.json"), "w") as f:
            json.dump(bias, f, indent=2)

        biased = {col: v for col, v in bias.items() if v["p_value"] < 0.05}
        print(f"    Biased features (KS p<0.05): {len(biased)} / {len(bias)}")

    # Save feature names (needed for SHAP / NN input_dim)
    feat_names = list(X.columns)
    with open(os.path.join(DATA_PROCESSED_DIR, "feature_names.json"), "w") as f:
        json.dump(feat_names, f, indent=2)
    print(f"  Saved {len(feat_names)} feature names.")


def run_gridsearch(n_jobs: int = -1):
    # Use 10% imbalance for gridsearch
    frac = 0.1
    X = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, f"X_full_{frac}.parquet"))
    y = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, f"y_full_{frac}.parquet")).squeeze()

    os.makedirs(os.path.dirname(PARAMS_FILE), exist_ok=True)
    best_params = {}

    for model_name, estimator in ESTIMATORS.items():
        print(f"\nGridSearch for {model_name} ...")
        pipe = Pipeline([
            ("imputer", KNNImputer(n_neighbors=KNN_IMPUTER_NEIGHBORS, weights="distance")),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ])
        search = GridSearchCV(
            pipe,
            PARAM_GRIDS[model_name],
            scoring="average_precision",
            refit=True,
            n_jobs=n_jobs,
            cv=5,
            verbose=1,
        )
        search.fit(X, y)
        print(f"  Best score: {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}")

        # Strip "clf__" prefix from param keys
        best_params[model_name] = {
            k.removeprefix("clf__"): v
            for k, v in search.best_params_.items()
        }

    with open(PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved best params to {PARAMS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_RAW, help="Path to accepted.csv")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Cores for GridSearchCV (-1 = all)")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip preprocessing, go straight to gridsearch")
    args = parser.parse_args()

    if not args.skip_preprocess:
        preprocess_and_save(args.data)

    run_gridsearch(n_jobs=args.n_jobs)
    print("\nSetup complete. Ready to launch HQ task array.")
