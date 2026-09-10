import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE

KEEP_COLS = [
    "loan_amnt", "term", "installment", "grade", "emp_length",
    "home_ownership", "annual_inc", "verification_status", "issue_d",
    "loan_status", "purpose", "addr_state", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "initial_list_status",
    "application_type", "mort_acc", "pub_rec_bankruptcies",
]


def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, low_memory=False)
    df = df[df.loan_status.isin(["Fully Paid", "Charged Off"])]

    missing_pct = df.isnull().mean() * 100
    df = df.drop(columns=missing_pct[missing_pct > 50].index, errors="ignore")

    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available]

    df = df.rename(columns={"loan_status": "target"})
    df["target"] = df["target"].map({"Charged Off": 1, "Fully Paid": 0})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Individual application"] = (df["application_type"] == "INDIVIDUAL").astype(np.int8)
    df.drop(columns=["application_type"], inplace=True)

    emp_map = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
        "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
        "10+ years": 10,
    }
    df["emp_length"] = df["emp_length"].map(emp_map)

    df["term"] = (
        df["term"].fillna("36 months").str.strip().str.replace(" months", "", regex=False).astype(int)
    )

    df.drop("grade", axis=1, inplace=True)

    dummies = ["verification_status", "purpose", "initial_list_status", "home_ownership", "addr_state"]
    dummies = [c for c in dummies if c in df.columns]
    df = pd.get_dummies(df, columns=dummies, drop_first=True)

    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    reference = pd.Timestamp("2018-01-01")
    df["credit_history_yrs"] = (reference - df["earliest_cr_line"]).dt.days / 365.25
    df.drop(columns="earliest_cr_line", inplace=True)

    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_mid"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
        df.drop(columns=["fico_range_low", "fico_range_high"], inplace=True)

    df["annual_inc"] = np.log1p(df["annual_inc"])
    df["revol_bal"] = np.log1p(df["revol_bal"])

    df.drop("issue_d", axis=1, inplace=True, errors="ignore")

    df = df.dropna(subset=["loan_amnt"])
    df = df.dropna(subset=["target"])

    # Ensure no boolean columns (parquet/sklearn compatibility)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(np.int8)

    return df


def calc_bias(df_original: pd.DataFrame, df_subset: pd.DataFrame) -> dict:
    results = {}
    numeric_cols = df_original.select_dtypes(include="number").columns.difference(["target"])
    for col in numeric_cols:
        a = df_original[col].dropna().values
        b = df_subset[col].dropna().values
        stat, p = ks_2samp(a, b)
        results[col] = {"ks_stat": round(float(stat), 4), "p_value": round(float(p), 4)}
    return results


def create_imbalanced_dataset(
    df: pd.DataFrame, frac: float, random_state: int = RANDOM_STATE
) -> tuple:
    majority = df[df["target"] == 0]
    minority = df[df["target"] == 1]
    natural_frac = len(minority) / len(df)

    if frac <= natural_frac:
        n_minority = int(round(frac / (1 - frac) * len(majority)))
        min_part = minority.sample(n=n_minority, random_state=random_state)
        maj_part = majority
    else:
        n_majority = int(round((1 - frac) / frac * len(minority)))
        maj_part = majority.sample(n=n_majority, random_state=random_state)
        min_part = minority

    dataset = (
        pd.concat([maj_part, min_part])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    bias = calc_bias(df, dataset)
    X = dataset.drop(columns="target")
    y = dataset["target"]
    return X, y, bias


def make_train_test_split(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)


def make_validation_split(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE):
    """Carve a stratified validation set out of the training set (20%).

    Used to calibrate the F1-optimal decision threshold for the adversarial
    attacks without touching the held-out test set.
    """
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
