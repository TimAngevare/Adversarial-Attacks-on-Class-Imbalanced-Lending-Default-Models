import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, f1_score,
    precision_score, recall_score, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import geometric_mean_score

from src.config import (
    RANDOM_STATE, N_FOLDS, N_SHAP_MAX, N_SHAP_NN, N_SHAP_NN_BG,
    N_ATTACK_CASES, ATTACK_MAX_ITER, N_CHANGES, MUTABLE_FEATURES,
    DIR_GROUP_SPECS, FAVOURABLE_LABEL,
)


def _to_float32(X):
    return np.asarray(X, dtype=np.float32)


def build_pipeline(estimator, sampler) -> ImbPipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if sampler is not None:
        steps.append(("sampler", sampler))
    if estimator.__class__.__name__ == "NeuralNetClassifier":
        steps.append(("to_float32", FunctionTransformer(_to_float32)))
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


def best_f1_threshold(y_true, y_proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


# ---------------------------------------------------------------------------
# CV metrics
# ---------------------------------------------------------------------------

def run_cv(pipe, X, y, n_jobs: int = 1):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_validate(
        pipe, X, y, cv=skf,
        scoring=("roc_auc", "average_precision"),
        return_train_score=True,
        return_estimator=True,
        return_indices=True,
        n_jobs=n_jobs,
    )


def compute_cv_metrics(cv_out, imbalance_ratio, model_name, sampler_name, X, y) -> dict:
    y_arr = np.asarray(y)
    y_pred = np.empty_like(y_arr)
    fold_thresholds, train_f1s, test_f1s = [], [], []

    for est, train_idx, test_idx in zip(
        cv_out["estimator"],
        cv_out["indices"]["train"],
        cv_out["indices"]["test"],
    ):
        X_iloc = X.iloc if hasattr(X, "iloc") else X
        proba_tr = est.predict_proba(X_iloc[train_idx])[:, 1]
        thr = best_f1_threshold(y_arr[train_idx], proba_tr)
        proba_te = est.predict_proba(X_iloc[test_idx])[:, 1]
        y_pred[test_idx] = (proba_te >= thr).astype(y_arr.dtype)
        fold_thresholds.append(thr)
        train_f1s.append(f1_score(y_arr[train_idx], (proba_tr >= thr).astype(y_arr.dtype), average="macro"))
        test_f1s.append(f1_score(y_arr[test_idx], y_pred[test_idx], average="macro"))

    cm = confusion_matrix(y_arr, y_pred, normalize="true")
    train_f1 = float(np.mean(train_f1s))
    test_f1 = float(np.mean(test_f1s))

    return {
        "Model": model_name,
        "Sampler": sampler_name,
        "Imbalance ratio": imbalance_ratio,
        "fit_time_mean": round(float(cv_out["fit_time"].mean()), 2),
        "score_time_mean": round(float(cv_out["score_time"].mean()), 2),
        "Threshold (mean)": round(float(np.mean(fold_thresholds)), 4),
        "F1 (mean)": round(test_f1, 4),
        "ROC-AUC (mean)": round(float(cv_out["test_roc_auc"].mean()), 4),
        "PR-AUC (mean)": round(float(cv_out["test_average_precision"].mean()), 4),
        "Recall (mean)": round(float(recall_score(y_arr, y_pred, average="macro")), 4),
        "Precision (mean)": round(float(precision_score(y_arr, y_pred, average="macro")), 4),
        "G-Mean (mean)": round(float(geometric_mean_score(y_arr, y_pred)), 4),
        "Train F1 (mean)": round(train_f1, 4),
        "F1 gap (train-test)": round(train_f1 - test_f1, 4),
        "CM_TP": float(cm[1, 1]),
        "CM_FP": float(cm[0, 1]),
        "CM_TN": float(cm[0, 0]),
        "CM_FN": float(cm[1, 0]),
    }


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def _preprocess_for_shap(pipe, X):
    X_t = np.asarray(X) if not hasattr(X, "iloc") else X
    for _, step in pipe.steps[:-1]:
        if hasattr(step, "fit_resample"):
            continue
        X_t = step.transform(X_t)
    return np.asarray(X_t)


def _get_sv_class1(sv):
    if isinstance(sv, list) and len(sv) == 2:
        return sv[1]
    if hasattr(sv, "values"):
        v = sv.values
        return v[:, :, 1] if v.ndim == 3 else v
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def compute_shap(
    pipe, train_idx, test_idx, model_name, sampler_name, imbalance_ratio, X, feat_names
) -> dict | None:
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
            X_te_raw = pd.DataFrame(X_iloc[test_idx], columns=feat_names).astype(np.float32)
            X_tr_raw = pd.DataFrame(X_iloc[train_idx], columns=feat_names).astype(np.float32)
            ei = rng.choice(len(X_te_raw), min(N_SHAP_NN, len(X_te_raw)), replace=False)
            bi = rng.choice(len(X_tr_raw), min(N_SHAP_NN_BG, len(X_tr_raw)), replace=False)

            def nn_predict(X_raw):
                df = pd.DataFrame(X_raw, columns=feat_names).astype(np.float32)
                return pipe.predict_proba(df)[:, 1]

            sv = shap.KernelExplainer(nn_predict, X_tr_raw.iloc[bi]).shap_values(
                X_te_raw.iloc[ei], nsamples=100, l1_reg="num_features(10)"
            )
        else:
            return None

        importances = pd.Series(np.abs(sv).mean(axis=0), index=feat_names)
        row = {
            "Model": model_name,
            "Sampler": sampler_name,
            "Imbalance ratio": imbalance_ratio,
        }
        row.update(importances.to_dict())
        return row

    except Exception as e:
        print(f"  SHAP skipped ({model_name}+{sampler_name} IR={imbalance_ratio}): {e}")
        return None


def _get_group_threshold(feature: str, spec: dict, X_reference: pd.DataFrame):
    if spec["method"] == "threshold":
        return spec["threshold"]
    return X_reference[feature].quantile(spec.get("quantile", 0.50))


def compute_dataset_dir(
    pipe: ImbPipeline,
    X_tr: pd.DataFrame,
    y_tr,
    group_specs: dict = DIR_GROUP_SPECS,
    favourable_label: int = FAVOURABLE_LABEL,
) -> pd.DataFrame:

    imputer = clone(pipe.named_steps["imputer"])
    scaler = clone(pipe.named_steps["scaler"])

    X_imp = imputer.fit_transform(X_tr)
    X_scaled = scaler.fit_transform(X_imp)

    sampler_step = pipe.named_steps.get("sampler")
    if sampler_step is not None:
        X_res, y_res = clone(sampler_step).fit_resample(X_scaled, np.asarray(y_tr))
    else:
        X_res, y_res = X_scaled, np.asarray(y_tr)

    # Inverse-transform back to original (imputed, unscaled) feature space
    # so that group thresholds (emp_length < 5, annual_inc quantile) stay valid.
    X_orig = pd.DataFrame(scaler.inverse_transform(X_res), columns=X_tr.columns)
    y_res = pd.Series(y_res)

    rows = []
    for feature, spec in group_specs.items():
        if feature not in X_orig.columns:
            continue
        # Threshold is derived from the ORIGINAL unscaled X_tr
        threshold = _get_group_threshold(feature, spec, X_tr)
        unpriv_mask = X_orig[feature] < threshold
        groups = unpriv_mask.map({True: spec["unprivileged_name"], False: spec["privileged_name"]})

        tmp = pd.DataFrame({"group": groups, "y": y_res})
        gt = tmp.groupby("group").apply(
            lambda g: pd.Series({
                "n_samples": len(g),
                "favourable_label_rate": (g["y"] == favourable_label).mean(),
                "default_label_rate": (g["y"] != favourable_label).mean(),
            }),
            include_groups=False,
        )

        priv = spec["privileged_name"]
        unpriv = spec["unprivileged_name"]
        priv_rate = gt.loc[priv, "favourable_label_rate"] if priv in gt.index else np.nan
        unpriv_rate = gt.loc[unpriv, "favourable_label_rate"] if unpriv in gt.index else np.nan
        dir_score = unpriv_rate / priv_rate if (pd.notna(priv_rate) and priv_rate > 0) else np.nan

        for group_name, vals in gt.iterrows():
            rows.append({
                "Feature": feature,
                "Group": group_name,
                "Role": "unprivileged" if group_name == unpriv else "privileged",
                "Threshold used": threshold,
                "DIR": dir_score,
                **vals.to_dict(),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Adversarial attacks
# ---------------------------------------------------------------------------

def constrained_random_attack(
    model, x_original: pd.DataFrame, X_reference: pd.DataFrame,
    mutable_features: list, desired_class: int = 0,
    max_iter: int = ATTACK_MAX_ITER, n_changes: int = N_CHANGES,
    random_state: int = RANDOM_STATE,
) -> dict:
    rng = np.random.default_rng(random_state)
    x_original = x_original.copy()
    original_prob = model.predict_proba(x_original)[:, 1][0]

    best_x = x_original.copy()
    best_prob = original_prob
    best_pred = model.predict(x_original)[0]
    best_changes = {}

    for _ in range(max_iter):
        x_cand = x_original.copy()
        chosen = rng.choice(mutable_features, size=min(n_changes, len(mutable_features)), replace=False)
        changes = {}
        for feat in chosen:
            old_val = x_cand[feat].iloc[0]
            new_val = X_reference[feat].sample(n=1, random_state=int(rng.integers(0, 1_000_000))).values[0]
            x_cand.loc[:, feat] = new_val
            if old_val != new_val:
                changes[feat] = {"old": old_val, "new": new_val}

        pred = model.predict(x_cand)[0]
        prob = model.predict_proba(x_cand)[:, 1][0]

        if prob < best_prob:
            best_prob, best_x, best_pred, best_changes = prob, x_cand.copy(), pred, changes.copy()

        if pred == desired_class:
            return {
                "success": 1,
                "cf_default_prob": prob,
                "changed_features": int((x_original.iloc[0] != x_cand.iloc[0]).sum()),
                "prob_reduction": original_prob - prob,
            }

    return {
        "success": 0,
        "cf_default_prob": best_prob,
        "changed_features": int((x_original.iloc[0] != best_x.iloc[0]).sum()),
        "prob_reduction": original_prob - best_prob,
    }


def run_adversarial_attacks(
    pipe, X_tr: pd.DataFrame, X_te: pd.DataFrame,
    y_te, y_pred_te,
    mutable_features: list = None,
    n_cases: int = N_ATTACK_CASES,
    max_iter: int = ATTACK_MAX_ITER,
    n_changes: int = N_CHANGES,
    random_state: int = RANDOM_STATE,
) -> dict:
    if mutable_features is None:
        mutable_features = MUTABLE_FEATURES
    feats = [f for f in mutable_features if f in X_tr.columns]

    y_te_arr = np.asarray(y_te)
    y_pred_arr = np.asarray(y_pred_te)
    mask = (y_te_arr == 1) & (y_pred_arr == 1)

    X_cand = X_te[mask]
    n_actual = min(n_cases, len(X_cand))

    if n_actual == 0:
        return {
            "N attacked": 0,
            "Attack success rate": np.nan,
            "Mean changed features": np.nan,
            "Mean prob reduction (success)": np.nan,
            "Mean prob reduction (all)": np.nan,
        }

    X_cand = X_cand.sample(n=n_actual, random_state=random_state)
    case_results = []
    for idx in X_cand.index:
        x0 = X_te.loc[[idx]]
        case_results.append(
            constrained_random_attack(
                pipe, x0, X_tr, feats,
                desired_class=0,
                max_iter=max_iter,
                n_changes=n_changes,
                random_state=random_state + int(idx),
            )
        )

    rdf = pd.DataFrame(case_results)
    succ = rdf["success"] == 1
    return {
        "N attacked": n_actual,
        "Attack success rate": round(float(rdf["success"].mean()), 4),
        "Mean changed features": round(float(rdf.loc[succ, "changed_features"].mean()), 4) if succ.any() else np.nan,
        "Mean prob reduction (success)": round(float(rdf.loc[succ, "prob_reduction"].mean()), 4) if succ.any() else np.nan,
        "Mean prob reduction (all)": round(float(rdf["prob_reduction"].mean()), 4),
    }
