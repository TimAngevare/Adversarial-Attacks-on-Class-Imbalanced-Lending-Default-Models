import os
import warnings

# Pre-create ART's config dir BEFORE importing `art` below. art/config.py runs
# `os.makedirs("~/.art")` without exist_ok=True at import time; when many array
# tasks start concurrently they race on it and all but one crash with
# FileExistsError -> NameError: name '_config' is not defined. Creating it
# atomically here makes ART's `os.path.exists` check pass and skip its makedirs.
os.makedirs(os.path.expanduser("~/.art"), exist_ok=True)

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
    N_ATTACK_CASES, MUTABLE_FEATURES,
    GA_SOL_PER_POP, GA_NUM_PARENTS_MATING, GA_NUM_GENERATIONS, GA_N_RUNS, GA_BETA,
    ZOO_MAX_ITER, ZOO_LEARNING_RATE, ZOO_BINARY_SEARCH_STEPS,
    ZOO_INITIAL_CONST, ZOO_NB_PARALLEL,
    DIR_GROUP_SPECS, FAVOURABLE_LABEL,
)

from src.ga_attack import GAdvExample
from art.attacks.evasion import ZooAttack
from art.estimators.classification import BlackBoxClassifier



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

def _default_prob(cf_rows: np.ndarray, feature_names: list, model) -> np.ndarray:
        df = pd.DataFrame(cf_rows, columns=feature_names)
        return model.predict_proba(df)[:, 1]

def constrained_ga_attack(
    model, x_original: pd.DataFrame, X_reference: pd.DataFrame,
    mutable_features: list, desired_class: int = 0,
    sol_per_pop: int = GA_SOL_PER_POP,
    num_parents_mating: int = GA_NUM_PARENTS_MATING,
    num_generations: int = GA_NUM_GENERATIONS,
    n_runs: int = GA_N_RUNS,
    beta: float = GA_BETA,
    random_state: int = RANDOM_STATE,
) -> dict:
  
    feature_names = list(X_reference.columns)
    mutable_set = set(mutable_features)
    
    black_list = [i for i, f in enumerate(feature_names) if f not in mutable_set]

    X_ref_np = np.asarray(X_reference.values, dtype=float)
    x_np = np.asarray(x_original.values, dtype=float).reshape(-1)

    original_prob = float(model.predict_proba(x_original)[:, 1][0])

    
    np.random.seed(random_state)

    ga = GAdvExample(
        feature_names=feature_names,
        cat_vars_ohe=None,
        sol_per_pop=sol_per_pop,
        num_parents_mating=num_parents_mating,
        num_generations=num_generations,
        n_runs=n_runs,
        black_list=black_list,
        beta=beta,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        _, _, x_success = ga.attack(model, x_np, X_ref_np)

    if len(x_success) > 0:
        cfs = np.asarray(x_success, dtype=float)
        probs = _default_prob(cfs, feature_names, model)
        n_changed = (np.abs(cfs - x_np) > 0).sum(axis=1)
        # Prefer the most proximal counterfactual: fewest changed features,
        # then the lowest remaining default probability.
        best = np.lexsort((probs, n_changed))[0]
        return {
            "success": 1,
            "cf_default_prob": float(probs[best]),
            "changed_features": int(n_changed[best]),
            "prob_reduction": original_prob - float(probs[best]),
        }

    return {
        "success": 0,
        "cf_default_prob": original_prob,
        "changed_features": 0,
        "prob_reduction": 0.0,
    }

def predict_fn(x_mut: np.ndarray, base: np.ndarray, mutable_idx: np.ndarray, feature_names: list, model) -> np.ndarray:
    x_mut = np.asarray(x_mut, dtype=float)
    if x_mut.ndim == 1:
        x_mut = x_mut.reshape(1, -1)
    X = np.tile(base, (x_mut.shape[0], 1))
    X[:, mutable_idx] = x_mut
    return model.predict_proba(pd.DataFrame(X, columns=feature_names))

def constrained_zoo_attack(
    model, x_original: pd.DataFrame, X_reference: pd.DataFrame,
    mutable_features: list, desired_class: int = 0,
    max_iter: int = ZOO_MAX_ITER,
    learning_rate: float = ZOO_LEARNING_RATE,
    binary_search_steps: int = ZOO_BINARY_SEARCH_STEPS,
    initial_const: float = ZOO_INITIAL_CONST,
    nb_parallel: int = ZOO_NB_PARALLEL,
    random_state: int = RANDOM_STATE,
) -> dict:

    feature_names = list(X_reference.columns)
    mutable_set = set(mutable_features)
    mutable_idx = np.array(
        [i for i, f in enumerate(feature_names) if f in mutable_set], dtype=int
    )
    n_mut = len(mutable_idx)

    base = np.asarray(x_original.values, dtype=float).reshape(-1)
    original_prob = float(model.predict_proba(x_original)[:, 1][0])

    x_mut0 = base[mutable_idx].astype(np.float32)
    train_low = X_reference.iloc[:, mutable_idx].min().to_numpy(dtype=np.float32)
    train_high = X_reference.iloc[:, mutable_idx].max().to_numpy(dtype=np.float32)
    
    low = np.minimum(train_low, x_mut0)
    high = np.maximum(train_high, x_mut0)
    high = np.where(high <= low, low + 1e-6, high)

    if nb_parallel is None:
        nb_parallel = n_mut

    classifier = BlackBoxClassifier(
        lambda x_mut: predict_fn(x_mut, base, mutable_idx, feature_names, model),
        input_shape=(n_mut,),
        nb_classes=2,
        clip_values=(low, high),
    )

    attack = ZooAttack(
        classifier=classifier,
        targeted=False,
        learning_rate=learning_rate,
        max_iter=max_iter,
        binary_search_steps=binary_search_steps,
        initial_const=initial_const,
        abort_early=True,
        use_resize=False,
        use_importance=False,
        nb_parallel=max(1, min(nb_parallel, n_mut)),
        batch_size=1,
        verbose=False,
    )

    np.random.seed(random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_mut_adv = attack.generate(x=x_mut0.reshape(1, -1))

    X_adv = base.copy()
    X_adv[mutable_idx] = np.asarray(x_mut_adv, dtype=float).reshape(-1)
    adv_df = pd.DataFrame(X_adv.reshape(1, -1), columns=feature_names)
    adv_prob = float(model.predict_proba(adv_df)[:, 1][0])
    adv_pred = int(model.predict(adv_df)[0])
    changed = int((np.abs(X_adv[mutable_idx] - base[mutable_idx]) > 1e-8).sum())

    return {
        "success": 1 if adv_pred == desired_class else 0,
        "cf_default_prob": adv_prob,
        "changed_features": changed,
        "prob_reduction": original_prob - adv_prob,
    }


def _summarise_cases(case_results: list) -> dict:
    rdf = pd.DataFrame(case_results)
    succ = rdf["success"] == 1
    return {
        "Attack success rate": round(float(rdf["success"].mean()), 4),
        "Mean changed features": round(float(rdf.loc[succ, "changed_features"].mean()), 4) if succ.any() else np.nan,
        "Mean prob reduction (success)": round(float(rdf.loc[succ, "prob_reduction"].mean()), 4) if succ.any() else np.nan,
        "Mean prob reduction (all)": round(float(rdf["prob_reduction"].mean()), 4),
    }


# Attack methods compared in the results: name -> per-instance attack function.
ATTACK_METHODS = {
    "GA": constrained_ga_attack,
    "ZOO": constrained_zoo_attack,
}


def run_adversarial_attacks(
    pipe, X_tr: pd.DataFrame, X_te: pd.DataFrame,
    y_te, y_pred_te,
    mutable_features: list = None,
    n_cases: int = N_ATTACK_CASES,
    methods: dict = None,
    random_state: int = RANDOM_STATE,
) -> dict:
    
    if methods is None:
        methods = ATTACK_METHODS
    if mutable_features is None:
        mutable_features = MUTABLE_FEATURES

    feats = [f for f in mutable_features if f in X_tr.columns]

    metric_names = [
        "Attack success rate", "Mean changed features",
        "Mean prob reduction (success)", "Mean prob reduction (all)",
    ]

    y_te_arr = np.asarray(y_te)
    y_pred_arr = np.asarray(y_pred_te)
    mask = (y_te_arr == 1) & (y_pred_arr == 1)

    X_cand = X_te[mask]
    n_actual = min(n_cases, len(X_cand))

    if n_actual == 0:
        out = {"N attacked": 0}
        for name in methods:
            out.update({f"{m} ({name})": np.nan for m in metric_names})
        return out

    X_cand = X_cand.sample(n=n_actual, random_state=random_state)

    out = {"N attacked": n_actual}
    for name, attack_fn in methods.items():
        case_results = []
        for idx in X_cand.index:
            x0 = X_te.loc[[idx]]
            case_results.append(
                attack_fn(
                    pipe, x0, X_tr, feats,
                    desired_class=0,
                    random_state=random_state + int(idx),
                )
            )
        summary = _summarise_cases(case_results)
        out.update({f"{m} ({name})": summary[m] for m in metric_names})

    return out
