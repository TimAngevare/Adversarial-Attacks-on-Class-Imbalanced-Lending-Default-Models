# Prepare lending data while preserving the original borrower IDs.
import json
import locale
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE, TEST_SIZE, MISSING_THRESHOLD, NEAR_CONSTANT_THRESHOLD,
    HIGH_CARDINALITY_COUNT, CORR_DROP_COLS, DROP_FOR_LEAKAGE, DROP_FOR_PROXY_RISK,
    PRICING_COLUMNS, DROP_MANUAL, EMP_LENGTH_MAP, LOG_FEATURES,
)


# Prepare predictors and target while keeping pricing columns separate
def load_and_prepare(filepath: str, verbose: bool = True):
    log = print if verbose else (lambda *a, **k: None)
    report = {}

    df = pd.read_csv(filepath, low_memory=False)
    report["raw_shape"] = list(df.shape)
    df = df[df.loan_status.isin(["Fully Paid", "Charged Off"])]
    log(f"Resolved loans: {len(df):,}")

    # Drop columns with more than 10% missing values
    missing_ratio = df.isnull().mean()
    cols_above_missing = missing_ratio[missing_ratio > MISSING_THRESHOLD].index.tolist()
    df = df.drop(columns=cols_above_missing)
    report["dropped_missing"] = cols_above_missing

    # Create the FICO midpoint and drop selected correlated columns
    df["fico_mid"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df = df.drop(columns=["fico_range_low", "fico_range_high"], errors="ignore")
    corr_cols = [c for c in CORR_DROP_COLS if c in df.columns]
    df = df.drop(columns=corr_cols)
    report["dropped_correlated"] = corr_cols

    # Drop near-constant columns
    near_constant = [
        col for col in df.columns
        if df[col].value_counts(normalize=True, dropna=False).iloc[0] > NEAR_CONSTANT_THRESHOLD
    ]
    df = df.drop(columns=near_constant)
    report["dropped_near_constant"] = near_constant

    # Drop categorical columns with too many unique values
    cat_cols = df.select_dtypes(include=["object"]).columns
    high_unique = [c for c in cat_cols if df[c].nunique(dropna=True) > HIGH_CARDINALITY_COUNT]
    df = df.drop(columns=high_unique)
    report["dropped_high_cardinality"] = high_unique

    # Remove leakage and keep pricing columns for sensitivity analysis
    pricing_df = df[[c for c in PRICING_COLUMNS if c in df.columns]].copy()
    if "int_rate" in pricing_df.columns and pricing_df["int_rate"].dtype == object:
        pricing_df["int_rate"] = pd.to_numeric(
            pricing_df["int_rate"].astype(str).str.rstrip("%").str.strip(), errors="coerce"
        )
    term_months = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    report["implied_rate_check"] = implied_rate_check(df["loan_amnt"], term_months, pricing_df)

    df = df.drop(columns=[c for c in DROP_FOR_LEAKAGE if c in df.columns])
    df = df.drop(columns=[c for c in DROP_FOR_PROXY_RISK if c in df.columns])
    df = df.drop(columns=[c for c in DROP_MANUAL if c in df.columns])

    # Map loan status to the binary default target
    df = df.rename(columns={"loan_status": "target"})
    df["target"] = df["target"].map({"Charged Off": 1, "Fully Paid": 0})

    # Encode employment length, loan term and credit history
    df["emp_length"] = df["emp_length"].map(EMP_LENGTH_MAP)
    df["term"] = df["term"].fillna("36 months")
    df["term"] = df["term"].str.strip().str.replace(" months", "").astype(int)
    try:
        locale.setlocale(locale.LC_TIME, "C")
    except locale.Error:
        pass
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    reference_date = df["earliest_cr_line"].max()
    df["credit_history_years"] = (reference_date - df["earliest_cr_line"]).dt.days / 365.25
    df = df.drop(columns=["earliest_cr_line"])

    # Transform skewed features and remove rows without a loan amount
    for col in LOG_FEATURES:
        df[col] = np.log1p(df[col])
    df = df.dropna(subset=["loan_amnt"])
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    report["one_hot_encoded"] = categorical_cols

    leaks = set(DROP_FOR_PROXY_RISK) & set(df.columns)
    if leaks:
        raise RuntimeError(f"Pricing columns still present: {leaks}")

    report["final_shape"] = list(df.shape)
    report["base_default_rate"] = float(df["target"].mean())
    report["feature_names"] = [c for c in df.columns if c != "target"]
    log(f"Final feature matrix: {df.shape[0]:,} rows x {df.shape[1] - 1} features, "
        f"default rate {df['target'].mean():.3f}")
    return df, pricing_df, report


# Compare the interest rate implied by installment with int_rate
def implied_rate_check(loan_amnt, term_months, pricing_df) -> dict:
    if not {"installment", "int_rate"}.issubset(pricing_df.columns):
        return {"skipped": "installment or int_rate not available"}
    check = pd.DataFrame({
        "P": loan_amnt, "n": term_months,
        "A": pricing_df["installment"], "r": pricing_df["int_rate"],
    }).dropna()
    check = check[(check["P"] > 0) & (check["A"] > 0)]
    P, n, A = (check[c].to_numpy(dtype=float) for c in ["P", "n", "A"])
    low, high = np.full(len(P), 1e-8), np.full(len(P), 0.10)
    for _ in range(60):
        mid = (low + high) / 2
        too_high = P * mid / (1 - (1 + mid) ** -n) > A
        high = np.where(too_high, mid, high)
        low = np.where(too_high, low, mid)
    implied = 1200 * (low + high) / 2
    error = implied - check["r"].to_numpy(dtype=float)
    return {
        "loans_checked": int(len(check)),
        "correlation": float(np.corrcoef(implied, check["r"])[0, 1]) if len(check) > 1 else None,
        "median_abs_error_pp": float(np.median(np.abs(error))) if len(check) else None,
        "share_within_0.1pp": float(np.mean(np.abs(error) < 0.1)) if len(check) else None,
    }


# Compare feature distributions using KS statistics
def calc_bias(df_original: pd.DataFrame, df_subset: pd.DataFrame) -> dict:
    results = {}
    for col in df_original.select_dtypes(include="number").columns:
        stat, p = ks_2samp(df_original[col], df_subset[col])
        if pd.notna(stat):
            results[col] = {"ks_stat": round(float(stat), 4), "p_value": round(float(p), 4)}
    return results


# Assign borrowers to fixed stratified training and test pools
def make_pools(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    train_ids, test_ids = train_test_split(
        df.index.to_numpy(), test_size=TEST_SIZE, stratify=df["target"], random_state=random_state
    )
    return pd.Index(train_ids), pd.Index(test_ids)


# Subsample one class to reach the target prevalence
def _sample_to_prevalence(pool: pd.DataFrame, frac: float, seed: int):
    majority = pool[pool["target"] == 0]
    minority = pool[pool["target"] == 1]
    pool_fraction = len(minority) / len(pool)
    if frac <= pool_fraction:
        n_minority = int(round(frac / (1 - frac) * len(majority)))
        return majority, minority.sample(n=n_minority, random_state=seed)
    n_majority = int(round((1 - frac) / frac * len(minority)))
    return majority.sample(n=n_majority, random_state=seed), minority


# Sample each borrower pool to the target prevalence
def create_imbalanced_dataset(df, frac, train_ids, test_ids, random_state: int = RANDOM_STATE):
    parts = []
    for pool_ids in (train_ids, test_ids):
        parts.extend(_sample_to_prevalence(df.loc[pool_ids], frac, random_state))
    subsample = pd.concat(parts).sample(frac=1, random_state=random_state)
    bias = calc_bias(df, subsample)
    X = subsample.drop(columns="target")
    y = subsample["target"]
    return X, y, bias


# Save predictors, target and matched pricing data as Parquet
def save_dataset(directory, frac, X, y, pricing):
    X.to_parquet(os.path.join(directory, f"X_full_{frac}.parquet"))
    y.to_frame("target").to_parquet(os.path.join(directory, f"y_full_{frac}.parquet"))
    pricing.to_parquet(os.path.join(directory, f"pricing_{frac}.parquet"))


# Load predictors and target and check that borrower IDs match
def load_dataset(directory, frac):
    X = pd.read_parquet(os.path.join(directory, f"X_full_{frac}.parquet"))
    y = pd.read_parquet(os.path.join(directory, f"y_full_{frac}.parquet"))["target"]
    if not X.index.equals(y.index):
        raise RuntimeError(f"X and y of prevalence {frac} do not line up")
    return X, y


# Read the borrower IDs assigned to the test pool
def load_test_ids(directory) -> pd.Index:
    return pd.Index(pd.read_parquet(os.path.join(directory, "test_ids.parquet"))["borrower_id"])


# Split observations using their fixed borrower pools
def split_train_test(X, y, test_ids):
    is_test = X.index.isin(test_ids)
    return X[~is_test], X[is_test], y[~is_test], y[is_test]


# Hold out a stratified validation split from the training pool
def make_validation_split(X, y, random_state: int = RANDOM_STATE):
    from src.config import VALIDATION_SIZE
    return train_test_split(X, y, test_size=VALIDATION_SIZE, stratify=y, random_state=random_state)


# Save preprocessing information as JSON
def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
