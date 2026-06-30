import time
import warnings

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
    RANDOM_STATE, N_FOLDS,
    N_SHAP_MAX, N_SHAP_NN, N_SHAP_NN_BG,
    N_ATTACK_CASES, DEFAULT_CLASS, MAX_CHANGED_ACTIONS,
    ACTION_SPECS, LOG_FEATURES, ATTACK_REQUIRED_COLUMNS,
    LOWER_Q, UPPER_Q, PROXIMITY_PENALTY,
    GA_POP_SIZE, GA_GENERATIONS, GA_RESTARTS,
    GA_MUTATION_PROB, GA_MUTATION_SCALE, GA_ELITE_FRAC,
    ZOO_MAX_ITER, ZOO_RESTARTS, ZOO_LEARNING_RATE, ZOO_FD_STEP,
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


# Fixed decision threshold: predictions use the model's default 0.5 cutoff
# (no per-fold threshold tuning).
DECISION_THRESHOLD = 0.5


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
    train_f1s, test_f1s = [], []

    for est, train_idx, test_idx in zip(
        cv_out["estimator"],
        cv_out["indices"]["train"],
        cv_out["indices"]["test"],
    ):
        X_iloc = X.iloc if hasattr(X, "iloc") else X
        proba_tr = est.predict_proba(X_iloc[train_idx])[:, 1]
        proba_te = est.predict_proba(X_iloc[test_idx])[:, 1]
        y_pred[test_idx] = (proba_te >= DECISION_THRESHOLD).astype(y_arr.dtype)
        train_f1s.append(f1_score(y_arr[train_idx], (proba_tr >= DECISION_THRESHOLD).astype(y_arr.dtype), average="macro"))
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
        "Threshold (mean)": DECISION_THRESHOLD,
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
#
# Both attacks search for a realistic, *actionable* counterfactual: a sparse,
# direction-constrained change to a few mutable loan features (see ACTION_SPECS)
# that pushes the model's default probability below a calibrated threshold.
# The search happens in a normalised [0, 1] "action" space; vector_to_instance
# decodes it back to feature space and repairs dependent features.
# ---------------------------------------------------------------------------


def to_raw(value, feature):
    """Decode a model-space value to its raw scale (undo log1p for LOG_FEATURES)."""
    value = float(value)
    return max(np.expm1(value), 0.0) if feature in LOG_FEATURES else value


def to_model(value, feature):
    """Encode a raw value back to model space (apply log1p for LOG_FEATURES)."""
    value = max(float(value), 0.0)
    return np.log1p(value) if feature in LOG_FEATURES else value


def get_action_features(columns) -> list:
    return [feature for feature in ACTION_SPECS if feature in columns]


def project_action_vector(z: np.ndarray) -> np.ndarray:
    """Clip to [0, 1] and keep at most MAX_CHANGED_ACTIONS non-zero entries."""
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    active = np.flatnonzero(z > 1e-12)
    if len(active) > MAX_CHANGED_ACTIONS:
        keep = active[np.argsort(z[active])[-MAX_CHANGED_ACTIONS:]]
        mask = np.zeros(len(z), dtype=bool)
        mask[keep] = True
        z = np.where(mask, z, 0.0)
    return z


def get_bounds(x_original, X_reference, action_features) -> dict:
    """Per-feature (low, high) raw-scale bounds from direction + max_change, clamped
    to training quantiles."""
    bounds = {}
    for feature in action_features:
        original = to_raw(x_original.iloc[0][feature], feature)
        values = X_reference[feature].astype(float).to_numpy()
        if feature in LOG_FEATURES:
            values = np.expm1(values)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            bounds[feature] = (original, original)
            continue
        q_low = float(np.nanquantile(values, LOWER_Q))
        q_high = float(np.nanquantile(values, UPPER_Q))
        direction = ACTION_SPECS[feature]["direction"]
        max_change = ACTION_SPECS[feature]["max_change"]
        if direction == "increase":
            low = original
            high = min(original * (1 + max_change), q_high)
            if high < low:
                high = low
        else:  # decrease
            low = max(original * (1 - max_change), q_low, 0.0)
            high = original
            if low > high:
                low = high
        bounds[feature] = (float(low), float(high))
    return bounds


def repair_features(x_adv, x_original):
    """Keep dependent features consistent after an action: installment tracks the
    loan amount, dti tracks income, and revol_util tracks the balance."""
    eps = 1e-10
    if {"loan_amnt", "installment"}.issubset(x_adv.columns):
        old_loan = max(float(x_original.iloc[0]["loan_amnt"]), eps)
        old_installment = float(x_original.iloc[0]["installment"])
        new_loan = float(x_adv.iloc[0]["loan_amnt"])
        x_adv.loc[:, "installment"] = old_installment * new_loan / old_loan
    if {"annual_inc", "dti"}.issubset(x_adv.columns):
        old_income = max(to_raw(x_original.iloc[0]["annual_inc"], "annual_inc"), eps)
        new_income = max(to_raw(x_adv.iloc[0]["annual_inc"], "annual_inc"), eps)
        old_dti = float(x_original.iloc[0]["dti"])
        x_adv.loc[:, "dti"] = max(0.0, old_dti * old_income / new_income)
    if {"revol_bal", "revol_util"}.issubset(x_adv.columns):
        old_balance = to_raw(x_original.iloc[0]["revol_bal"], "revol_bal")
        old_util = float(x_original.iloc[0]["revol_util"])
        if old_balance > eps and old_util > eps:
            credit_limit = old_balance / (old_util / 100.0)
            new_balance = to_raw(x_adv.iloc[0]["revol_bal"], "revol_bal")
            x_adv.loc[:, "revol_util"] = np.clip(100.0 * new_balance / credit_limit, 0.0, 100.0)
    return x_adv


def vector_to_instance(z, x_original, bounds, action_features):
    """Decode a normalised action vector into a concrete adversarial instance."""
    z_effective = project_action_vector(z)
    x_adv = x_original.copy(deep=True)
    for i, feature in enumerate(action_features):
        original = to_raw(x_original.iloc[0][feature], feature)
        low, high = bounds[feature]
        if ACTION_SPECS[feature]["direction"] == "increase":
            new_value = original + z_effective[i] * (high - original)
        else:
            new_value = original - z_effective[i] * (original - low)
        new_value = np.clip(new_value, low, high)
        x_adv.loc[:, feature] = to_model(new_value, feature)
    x_adv = repair_features(x_adv, x_original)
    return x_adv, z_effective


def default_probability_index(model) -> int:
    classes = np.asarray(model.classes_)
    matches = np.where(classes == DEFAULT_CLASS)[0]
    if len(matches) != 1:
        raise ValueError(f"Default class {DEFAULT_CLASS} was not found in {classes}.")
    return int(matches[0])


def default_probabilities(model, X) -> np.ndarray:
    class_index = default_probability_index(model)
    return model.predict_proba(X)[:, class_index]


def default_probability(model, X, query_counter=None) -> float:
    if query_counter is not None:
        query_counter["count"] += 1
    return float(default_probabilities(model, X)[0])


def get_threshold(model, X_validation, y_validation) -> float:
    """F1-optimal default-probability threshold on a held-out validation set."""
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


def changed_features(x_adv, x_original, action_features) -> int:
    return int(sum(
        abs(to_raw(x_adv.iloc[0][feature], feature)
            - to_raw(x_original.iloc[0][feature], feature)) > 1e-8
        for feature in action_features
    ))


def evaluate_candidate(z, model, x_original, bounds, action_features, query_counter):
    """Objective = default probability + PROXIMITY_PENALTY * mean perturbation."""
    x_adv, z_effective = vector_to_instance(z, x_original, bounds, action_features)
    adversarial_probability = default_probability(model, x_adv, query_counter)
    perturbation_cost = np.mean(np.abs(z_effective))
    score = adversarial_probability + PROXIMITY_PENALTY * perturbation_cost
    return score, adversarial_probability, x_adv, z_effective


def create_result(
    method, original_probability, adversarial_probability,
    x_original, x_adv, action_features, threshold, query_counter, runtime_seconds,
) -> dict:
    result = {
        "Method": method,
        "Success": int(adversarial_probability < threshold),
        "Original default probability": original_probability,
        "Adversarial default probability": adversarial_probability,
        "Probability reduction": original_probability - adversarial_probability,
        "Changed features": changed_features(x_adv, x_original, action_features),
        "Queries": int(query_counter["count"]),
        "Runtime seconds": runtime_seconds,
    }
    for feature in action_features:
        before = to_raw(x_original.iloc[0][feature], feature)
        after = to_raw(x_adv.iloc[0][feature], feature)
        result[f"{feature}_before"] = before
        result[f"{feature}_after"] = after
        result[f"{feature}_relative_change"] = (after - before) / before if before != 0 else np.nan
    return result


def permute_attack(model, x_original, X_reference, action_features, threshold, seed=RANDOM_STATE) -> dict:
    """PermuteAttack-inspired genetic search over the normalised action space."""
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    query_counter = {"count": 0}

    bounds = get_bounds(x_original, X_reference, action_features)
    original_probability = default_probability(model, x_original, query_counter)

    d = len(action_features)
    best_score = np.inf
    best_probability = original_probability
    best_x_adv = x_original.copy(deep=True)
    n_elite = max(2, int(GA_POP_SIZE * GA_ELITE_FRAC))

    for _ in range(GA_RESTARTS):
        population = rng.uniform(0.0, 1.0, size=(GA_POP_SIZE, d))
        population[0] = np.zeros(d)
        population = np.asarray([project_action_vector(z) for z in population])

        for _ in range(GA_GENERATIONS):
            evaluations = []
            for z in population:
                score, probability, x_adv, z_effective = evaluate_candidate(
                    z, model, x_original, bounds, action_features, query_counter,
                )
                evaluations.append({"score": score, "probability": probability,
                                    "x_adv": x_adv, "z": z_effective})
            evaluations.sort(key=lambda item: item["score"])

            if evaluations[0]["score"] < best_score:
                best_score = evaluations[0]["score"]
                best_probability = evaluations[0]["probability"]
                best_x_adv = evaluations[0]["x_adv"].copy(deep=True)

            successful = [item for item in evaluations if item["probability"] < threshold]
            if successful:
                selected = min(successful, key=lambda item: item["score"])
                return create_result(
                    "PermuteAttack-inspired GA", original_probability,
                    selected["probability"], x_original, selected["x_adv"],
                    action_features, threshold, query_counter, time.perf_counter() - start,
                )

            elites = np.asarray([item["z"] for item in evaluations[:n_elite]])
            children = [elite.copy() for elite in elites]
            while len(children) < GA_POP_SIZE:
                parent_1 = elites[rng.integers(0, len(elites))]
                parent_2 = elites[rng.integers(0, len(elites))]
                crossover_mask = rng.random(d) < 0.5
                child = np.where(crossover_mask, parent_1, parent_2)
                mutation_mask = rng.random(d) < GA_MUTATION_PROB
                child[mutation_mask] += rng.normal(0.0, GA_MUTATION_SCALE, size=mutation_mask.sum())
                if rng.random() < 0.30:
                    child[rng.integers(0, d)] = 0.0
                children.append(project_action_vector(child))
            population = np.asarray(children)

    return create_result(
        "PermuteAttack-inspired GA", original_probability, best_probability,
        x_original, best_x_adv, action_features, threshold,
        query_counter, time.perf_counter() - start,
    )


def zoo_attack(model, x_original, X_reference, action_features, threshold, seed=RANDOM_STATE) -> dict:
    """ZOO-inspired finite-difference attack with an Adam update on the action vector."""
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    query_counter = {"count": 0}

    bounds = get_bounds(x_original, X_reference, action_features)
    original_probability = default_probability(model, x_original, query_counter)

    d = len(action_features)
    best_score = np.inf
    best_probability = original_probability
    best_x_adv = x_original.copy(deep=True)

    for restart in range(ZOO_RESTARTS):
        z = np.zeros(d) if restart == 0 else rng.uniform(0.0, 0.25, size=d)
        z = project_action_vector(z)
        m = np.zeros(d)
        v = np.zeros(d)

        current_score, current_probability, current_x_adv, z = evaluate_candidate(
            z, model, x_original, bounds, action_features, query_counter,
        )
        if current_score < best_score:
            best_score = current_score
            best_probability = current_probability
            best_x_adv = current_x_adv.copy(deep=True)

        for iteration in range(1, ZOO_MAX_ITER + 1):
            gradient = np.zeros(d)
            for j in range(d):
                z_plus = z.copy()
                z_minus = z.copy()
                z_plus[j] = min(1.0, z_plus[j] + ZOO_FD_STEP)
                z_minus[j] = max(0.0, z_minus[j] - ZOO_FD_STEP)
                f_plus, _, _, _ = evaluate_candidate(
                    z_plus, model, x_original, bounds, action_features, query_counter,
                )
                f_minus, _, _, _ = evaluate_candidate(
                    z_minus, model, x_original, bounds, action_features, query_counter,
                )
                denominator = max(z_plus[j] - z_minus[j], 1e-8)
                gradient[j] = (f_plus - f_minus) / denominator

            beta_1, beta_2 = 0.90, 0.999
            m = beta_1 * m + (1.0 - beta_1) * gradient
            v = beta_2 * v + (1.0 - beta_2) * gradient ** 2
            m_hat = m / (1.0 - beta_1 ** iteration)
            v_hat = v / (1.0 - beta_2 ** iteration)
            proposal = z - ZOO_LEARNING_RATE * m_hat / (np.sqrt(v_hat) + 1e-8)
            proposal = project_action_vector(proposal)

            proposal_score, proposal_probability, proposal_x_adv, proposal_z = evaluate_candidate(
                proposal, model, x_original, bounds, action_features, query_counter,
            )
            if proposal_score <= current_score:
                z = proposal_z
                current_score = proposal_score
                current_probability = proposal_probability
                current_x_adv = proposal_x_adv

            if current_score < best_score:
                best_score = current_score
                best_probability = current_probability
                best_x_adv = current_x_adv.copy(deep=True)

            if current_probability < threshold:
                return create_result(
                    "ZOO-inspired finite difference", original_probability,
                    current_probability, x_original, current_x_adv,
                    action_features, threshold, query_counter, time.perf_counter() - start,
                )

    return create_result(
        "ZOO-inspired finite difference", original_probability, best_probability,
        x_original, best_x_adv, action_features, threshold,
        query_counter, time.perf_counter() - start,
    )


# Attack methods compared in the results: column-suffix -> per-instance function.
ATTACK_METHODS = {
    "GA": permute_attack,    # PermuteAttack-inspired genetic algorithm
    "ZOO": zoo_attack,       # ZOO-inspired finite-difference (Adam)
}

_ATTACK_METRIC_NAMES = [
    "Attack success rate", "Mean changed features",
    "Mean prob reduction (success)", "Mean prob reduction (all)",
    "Median queries", "Median runtime",
]


def _summarise_cases(case_results: list) -> dict:
    rdf = pd.DataFrame(case_results)
    succ = rdf["Success"] == 1
    return {
        "Attack success rate": round(float(rdf["Success"].mean()), 4),
        "Mean changed features": round(float(rdf.loc[succ, "Changed features"].mean()), 4) if succ.any() else np.nan,
        "Mean prob reduction (success)": round(float(rdf.loc[succ, "Probability reduction"].mean()), 4) if succ.any() else np.nan,
        "Mean prob reduction (all)": round(float(rdf["Probability reduction"].mean()), 4),
        "Median queries": round(float(rdf["Queries"].median()), 2),
        "Median runtime": round(float(rdf["Runtime seconds"].median()), 4),
    }


def run_adversarial_attacks(
    pipe, X_ref: pd.DataFrame, X_val: pd.DataFrame, y_val,
    X_te: pd.DataFrame, y_te,
    n_cases: int = N_ATTACK_CASES,
    methods: dict = None,
    required_columns: list = None,
    random_state: int = RANDOM_STATE,
) -> dict:
    if methods is None:
        methods = ATTACK_METHODS
    if required_columns is None:
        required_columns = ATTACK_REQUIRED_COLUMNS

    action_features = get_action_features(X_ref.columns)

    # F1-optimal decision threshold from the held-out validation set, then pick
    # eligible targets: truly-defaulting, predicted-default, complete rows.
    threshold = get_threshold(pipe, X_val, y_val)
    test_probs = default_probabilities(pipe, X_te)
    y_te_arr = np.asarray(y_te)
    mask = (y_te_arr == DEFAULT_CLASS) & (test_probs >= threshold)

    avail_required = [c for c in required_columns if c in X_te.columns]
    if avail_required:
        mask &= X_te[avail_required].notna().all(axis=1).to_numpy()

    X_cand = X_te[mask]
    n_actual = min(n_cases, len(X_cand))

    if n_actual == 0 or len(action_features) == 0:
        out = {"N attacked": 0, "Threshold": round(float(threshold), 4)}
        for name in methods:
            out.update({f"{m} ({name})": np.nan for m in _ATTACK_METRIC_NAMES})
        return out

    X_cand = X_cand.sample(n=n_actual, random_state=random_state)

    out = {"N attacked": n_actual, "Threshold": round(float(threshold), 4)}
    for name, attack_fn in methods.items():
        case_results = []
        for idx in X_cand.index:
            x0 = X_te.loc[[idx]]
            case_results.append(
                attack_fn(pipe, x0, X_ref, action_features, threshold,
                          seed=random_state + int(idx))
            )
        summary = _summarise_cases(case_results)
        out.update({f"{m} ({name})": summary[m] for m in _ATTACK_METRIC_NAMES})

    return out
