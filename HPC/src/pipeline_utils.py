# Build model pipelines and calculate CV, SHAP and final-model results.
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import geometric_mean_score

from src.config import (
    RANDOM_STATE, N_FOLDS, N_SHAP_MAX, N_SHAP_NN, N_SHAP_NN_BG, DEFAULT_CLASS,
)
from src.models_def import apply_class_weight
from src.preprocessing import make_validation_split


# Convert neural-network input to float32
def _to_float32(X):
    return np.asarray(X, dtype=np.float32)


# Apply mean imputation, robust scaling, optional resampling and classification
def build_pipeline(estimator, sampler) -> ImbPipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", RobustScaler()),
    ]
    if sampler is not None:
        steps.append(("sampler", sampler))
    if estimator.__class__.__name__ == "NeuralNetClassifier":
        steps.append(("to_float32", FunctionTransformer(_to_float32)))
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


# Create the shuffled stratified cross-validation folds
def make_cv():
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# Select the threshold with the highest default-class F1

def best_f1_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return thresholds[np.argmax(f1[:-1])]


# Run stratified cross-validation and retain fitted models
def run_cv(pipe, X, y, n_jobs: int = 1):
    return cross_validate(
        pipe, X, y, cv=make_cv(),
        scoring=("roc_auc", "average_precision"),
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        n_jobs=n_jobs,
    )


# Score validation folds at thresholds selected from training-fold predictions
def compute_cv_metrics(cv_out, imbalance_ratio, model_name, sampler_name, X, y) -> dict:
    y_arr = np.asarray(y)
    y_pred = np.empty_like(y_arr)
    fold_thresholds, train_f1s, test_f1s = [], [], []
    X_iloc = X.iloc if hasattr(X, "iloc") else X

    for est, train_idx, test_idx in zip(
        cv_out["estimator"], cv_out["indices"]["train"], cv_out["indices"]["test"]
    ):
        proba_train = est.predict_proba(X_iloc[train_idx])[:, 1]
        thr = best_f1_threshold(y_arr[train_idx], proba_train)
        proba_test = est.predict_proba(X_iloc[test_idx])[:, 1]
        y_pred[test_idx] = (proba_test >= thr).astype(y_arr.dtype)
        fold_thresholds.append(float(thr))
        train_f1s.append(f1_score(y_arr[train_idx], (proba_train >= thr).astype(y_arr.dtype), average="macro"))
        test_f1s.append(f1_score(y_arr[test_idx], y_pred[test_idx], average="macro"))

    cm = confusion_matrix(y_arr, y_pred, normalize="true")
    train_f1 = float(np.mean(train_f1s))
    test_f1 = float(np.mean(test_f1s))

    return {
        "Model": model_name,
        "Sampler": sampler_name,
        "Imbalance ratio": imbalance_ratio,
        "fit time (s)": round(float(cv_out["fit_time"].mean()), 2),
        "Score time (s)": round(float(cv_out["score_time"].mean()), 2),
        "Threshold (mean)": round(float(np.mean(fold_thresholds)), 4),
        "F1 (mean)": test_f1,
        "ROC-AUC (mean)": float(cv_out["test_roc_auc"].mean()),
        "PR-AUC (mean)": float(cv_out["test_average_precision"].mean()),
        "Recall (mean)": float(recall_score(y_arr, y_pred, average="macro")),
        "Precision (mean)": float(precision_score(y_arr, y_pred, average="macro")),
        "G-Mean (mean)": float(geometric_mean_score(y_arr, y_pred)),
        "Train F1 (mean)": train_f1,
        "F1 gap (train-test)": round(train_f1 - test_f1, 3),
        "TP": float(cm[1, 1]),
        "FP": float(cm[0, 1]),
        "TN": float(cm[0, 0]),
        "FN": float(cm[1, 0]),
    }


# Apply preprocessing without resampling for SHAP

def _preprocess_for_shap(pipe, X):
    X_t = X
    for _, step in pipe.steps[:-1]:
        if hasattr(step, "fit_resample"):
            continue
        X_t = step.transform(X_t)
    return np.asarray(X_t)


# Select SHAP values for the default class
def _get_sv_class1(sv):
    if isinstance(sv, list) and len(sv) == 2:
        return sv[1]
    if hasattr(sv, "values"):
        v = sv.values
        return v[:, :, 1] if v.ndim == 3 else v
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        return sv[:, :, 1]
    return sv


# Calculate SHAP importance for the selected fold
def compute_shap(pipe, train_idx, test_idx, model_name, sampler_name, imbalance_ratio, X, feat_names):
    rng = np.random.default_rng(RANDOM_STATE)
    clf = pipe.named_steps["clf"]
    X_iloc = X.iloc if hasattr(X, "iloc") else X

    try:
        if model_name in ("XGBClassifier", "RandomForestClassifier"):
            X_te = _preprocess_for_shap(pipe, X_iloc[test_idx])
            idx = rng.choice(len(X_te), min(N_SHAP_MAX, len(X_te)), replace=False)
            sv = _get_sv_class1(shap.TreeExplainer(clf).shap_values(X_te[idx]))

        elif model_name == "LogisticRegression":
            X_tr = _preprocess_for_shap(pipe, X_iloc[train_idx])
            X_te = _preprocess_for_shap(pipe, X_iloc[test_idx])
            idx = rng.choice(len(X_te), min(N_SHAP_MAX, len(X_te)), replace=False)
            sv = _get_sv_class1(shap.LinearExplainer(clf, X_tr).shap_values(X_te[idx]))

        elif model_name == "NeuralNetClassifier":
            X_te_raw = pd.DataFrame(np.asarray(X_iloc[test_idx]), columns=feat_names).astype(np.float32)
            X_tr_raw = pd.DataFrame(np.asarray(X_iloc[train_idx]), columns=feat_names).astype(np.float32)
            ei = rng.choice(len(X_te_raw), min(N_SHAP_NN, len(X_te_raw)), replace=False)
            bi = rng.choice(len(X_tr_raw), min(N_SHAP_NN_BG, len(X_tr_raw)), replace=False)

            # Get neural-network default probabilities for SHAP
            def nn_predict(X_raw):
                frame = pd.DataFrame(X_raw, columns=feat_names).astype(np.float32)
                return pipe.predict_proba(frame)[:, 1]

            sv = shap.KernelExplainer(nn_predict, X_tr_raw.iloc[bi]).shap_values(
                X_te_raw.iloc[ei], nsamples=100, l1_reg="num_features(10)"
            )
        else:
            return None

        importances = pd.Series(np.abs(sv).mean(axis=0), index=feat_names)
        row = {"Model": model_name, "Sampler": sampler_name, "Imbalance ratio": imbalance_ratio}
        row.update(importances.to_dict())
        return row

    except Exception as e:
        print(f"  SHAP skipped ({model_name}+{sampler_name} IR={imbalance_ratio}): {e}")
        return None


# Find the prediction column for the default class

def default_probability_index(model) -> int:
    classes = np.asarray(model.classes_)
    matches = np.where(classes == DEFAULT_CLASS)[0]
    if len(matches) != 1:
        raise ValueError(f"Default class {DEFAULT_CLASS} was not found in {classes}.")
    return int(matches[0])


# Get default probabilities for all observations
def default_probabilities(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, default_probability_index(model)]


# Select the validation threshold with the highest default-class F1
def get_threshold(model, X_validation, y_validation) -> float:
    probabilities = default_probabilities(model, X_validation)
    precision, recall, thresholds = precision_recall_curve(
        y_validation, probabilities, pos_label=DEFAULT_CLASS,
    )
    if len(thresholds) == 0:
        return 0.50
    f1_scores = (
        2.0 * precision[:-1] * recall[:-1]
        / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    )
    return float(thresholds[np.argmax(f1_scores)])


# Fit the model and select the threshold using a validation split
def fit_final_model(base_model, sampler, sampler_name, X_train_val, y_train_val):
    X_fit, X_val, y_fit, y_val = make_validation_split(X_train_val, y_train_val)
    y_fit_arr = np.asarray(y_fit).astype(np.int64)

    estimator = apply_class_weight(clone(base_model), sampler_name, y_fit_arr)
    if estimator.__class__.__name__ == "NeuralNetClassifier":
        estimator.set_params(module__input_dim=X_fit.shape[1])
        torch.manual_seed(RANDOM_STATE)
        X_fit = X_fit.astype(np.float32)

    pipeline = build_pipeline(estimator, clone(sampler) if sampler is not None else None)
    pipeline.fit(X_fit, y_fit_arr)
    threshold = get_threshold(pipeline, X_val, y_val)
    return {
        "pipeline": pipeline, "threshold": threshold,
        "X_fit": X_fit, "y_fit": y_fit, "X_val": X_val, "y_val": y_val,
    }
