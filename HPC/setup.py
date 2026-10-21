# Prepare prevalence datasets and tune model hyperparameters.
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    DATA_RAW, DATA_PROCESSED_DIR, PARAMS_FILE,
    MINORITY_FRACS, RANDOM_STATE, TUNING_FRACTION,
)
from src.preprocessing import (
    load_and_prepare, make_pools, create_imbalanced_dataset, save_dataset,
    load_dataset, load_test_ids, split_train_test, write_json,
)
from src.pipeline_utils import make_cv


# Define the model hyperparameter grids
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
        "clf__C":            [0.01, 0.1, 1, 10],
        "clf__penalty":      ["l1", "l2"],
        "clf__solver":       ["saga"],
        "clf__max_iter":     [2000],
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

# Use one model thread per parallel grid-search fit
ESTIMATORS = {
    "XGBClassifier": XGBClassifier(n_jobs=1),
    "LogisticRegression": LogisticRegression(),
    "RandomForestClassifier": RandomForestClassifier(n_jobs=1),
}


# Prepare and save the borrower pools and prevalence datasets
def preprocess_and_save(data_path: str):
    print(f"Loading {data_path} ...")
    df, pricing_df, report = load_and_prepare(data_path)
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    train_ids, test_ids = make_pools(df)
    pd.DataFrame({"borrower_id": test_ids}).to_parquet(os.path.join(DATA_PROCESSED_DIR, "test_ids.parquet"))
    pd.DataFrame({"borrower_id": train_ids}).to_parquet(os.path.join(DATA_PROCESSED_DIR, "train_ids.parquet"))
    report["n_train_pool"], report["n_test_pool"] = len(train_ids), len(test_ids)
    print(f"  Pools: {len(train_ids):,} training and {len(test_ids):,} test borrowers")

    for frac in MINORITY_FRACS:
        X, y, bias = create_imbalanced_dataset(df, frac, train_ids, test_ids)
        save_dataset(DATA_PROCESSED_DIR, frac, X, y, pricing_df.loc[X.index])
        write_json(os.path.join(DATA_PROCESSED_DIR, f"bias_{frac}.json"), bias)
        test_share = X.index.isin(test_ids).mean()
        print(f"  Prevalence {frac}: {X.shape}, default rate {y.mean():.4f}, test share {test_share:.3f}")
        report.setdefault("datasets", {})[str(frac)] = {
            "shape": list(X.shape), "default_rate": float(y.mean()), "test_share": float(test_share),
        }

    feat_names = list(X.columns)
    write_json(os.path.join(DATA_PROCESSED_DIR, "feature_names.json"), feat_names)
    write_json(os.path.join(DATA_PROCESSED_DIR, "preprocessing_report.json"), report)
    print(f"  Saved {len(feat_names)} feature names. Implied-rate check: {report['implied_rate_check']}")


# Check that tuning borrowers are absent from every test pool
def check_pools():
    test_ids = load_test_ids(DATA_PROCESSED_DIR)
    X_t, y_t = load_dataset(DATA_PROCESSED_DIR, TUNING_FRACTION)
    X_tune, _, _, _ = split_train_test(X_t, y_t, test_ids)
    for frac in MINORITY_FRACS:
        X, y = load_dataset(DATA_PROCESSED_DIR, frac)
        _, X_te, _, _ = split_train_test(X, y, test_ids)
        assert X_tune.index.intersection(X_te.index).empty, f"Tuning borrowers in the {frac} test set"
    print("  Pool check passed: tuning borrowers are disjoint from every test set.")


# Tune model hyperparameters on the training pool
def run_gridsearch(n_jobs: int = -1, n_jobs_rf: int = 4):
    test_ids = load_test_ids(DATA_PROCESSED_DIR)
    X, y = load_dataset(DATA_PROCESSED_DIR, TUNING_FRACTION)
    X_tune, _, y_tune, _ = split_train_test(X, y, test_ids)
    print(f"Tuning on the training pool of the {TUNING_FRACTION} dataset: {X_tune.shape}")

    os.makedirs(os.path.dirname(PARAMS_FILE), exist_ok=True)
    best_params = {}
    for model_name, estimator in ESTIMATORS.items():
        t0 = time.perf_counter()
        print(f"\nGridSearch for {model_name} ...")
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", RobustScaler()),
            ("clf", estimator),
        ])
        # Limit concurrent Random Forest fits to reduce memory use
        jobs = n_jobs_rf if model_name == "RandomForestClassifier" else n_jobs
        search = GridSearchCV(
            pipe, PARAM_GRIDS[model_name], scoring="average_precision",
            cv=make_cv(), refit=False, n_jobs=jobs, verbose=1,
        )
        search.fit(X_tune, y_tune)
        print(f"  Best score: {search.best_score_:.4f}  ({time.perf_counter() - t0:.0f}s)")
        print(f"  Best params: {search.best_params_}")
        best_params[model_name] = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}

    with open(PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved best params to {PARAMS_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_RAW, help="Path to accepted.csv")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Parallel fits in GridSearchCV (-1 = all cores)")
    parser.add_argument("--n_jobs_rf", type=int, default=4,
                        help="Parallel Random Forest fits (memory-bound: each can take tens of GB)")
    parser.add_argument("--skip_preprocess", action="store_true", help="Reuse data/processed")
    parser.add_argument("--skip_gridsearch", action="store_true", help="Only preprocess")
    args = parser.parse_args()

    if not args.skip_preprocess:
        preprocess_and_save(args.data)
    check_pools()
    if not args.skip_gridsearch:
        run_gridsearch(n_jobs=args.n_jobs, n_jobs_rf=args.n_jobs_rf)
    print("\nSetup complete. Ready to launch the HQ task array.")
