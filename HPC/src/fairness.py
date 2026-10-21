# Calculate fairness metrics from final-model test predictions.
import numpy as np
import pandas as pd

from src.config import (
    DIR_GROUP_SPECS, FAVOURABLE_LABEL, N_BOOTSTRAP, FAIRNESS_CI, RANDOM_STATE,
)


# Get the fixed or training-based group threshold
def _get_group_threshold(feature, spec, X_reference):
    if spec["method"] == "threshold":
        return spec["threshold"]
    if spec["method"] == "quantile":
        return float(X_reference[feature].quantile(spec.get("quantile", 0.50)))
    raise ValueError(f"Unknown group method {spec['method']!r}")


# Assign group membership and exclude missing values
def make_group_masks(values, spec, threshold):
    values = np.asarray(values, dtype=float)
    known = np.isfinite(values)
    if spec.get("unprivileged_if", "lower") == "lower":
        unprivileged = values < threshold
    else:
        unprivileged = values >= threshold
    return known & unprivileged, known & ~unprivileged, ~known


# Return [TP, FN, FP, TN] with default as the positive class
def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.array([
        np.sum((y_true == 1) & (y_pred == 1)),
        np.sum((y_true == 1) & (y_pred == 0)),
        np.sum((y_true == 0) & (y_pred == 1)),
        np.sum((y_true == 0) & (y_pred == 0)),
    ], dtype=float)


# Calculate favourable prediction rates, TPR, FPR and default rates
def rates_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    tp, fn, fp, tn = np.moveaxis(counts, -1, 0)
    n = tp + fn + fp + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        favourable = (fn + tn) / n
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        default_rate = (tp + fn) / n
    return favourable, tpr, fpr, default_rate


# Bootstrap confusion counts using multinomial draws
def bootstrap_counts(counts, n_bootstrap, rng):
    n = int(counts.sum())
    if n == 0:
        return np.full((n_bootstrap, 4), np.nan)
    return rng.multinomial(n, counts / n, size=n_bootstrap).astype(float)


# Calculate the interval from finite bootstrap estimates
def _interval(samples, ci):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return np.nan, np.nan
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(low), float(high)


# Calculate group fairness metrics and bootstrap intervals
def evaluate_group_fairness(y_true, y_pred, X_eval, X_reference, group_specs=DIR_GROUP_SPECS,
                            favourable_label=FAVOURABLE_LABEL, n_bootstrap=N_BOOTSTRAP,
                            ci=FAIRNESS_CI, seed=RANDOM_STATE):
    if favourable_label != 0:
        raise ValueError("The rates below assume that the favourable label is 0 (non-default).")
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    rng = np.random.default_rng(seed)

    rows = []
    for feature, spec in group_specs.items():
        if feature not in X_eval.columns:
            print(f"Skipping {feature}: not found in X_eval")
            continue
        threshold = _get_group_threshold(feature, spec, X_reference)
        unpriv_mask, priv_mask, missing = make_group_masks(X_eval[feature], spec, threshold)

        counts_u = confusion_counts(y_true[unpriv_mask], y_pred[unpriv_mask])
        counts_p = confusion_counts(y_true[priv_mask], y_pred[priv_mask])
        fav_u, tpr_u, fpr_u, dr_u = rates_from_counts(counts_u)
        fav_p, tpr_p, fpr_p, dr_p = rates_from_counts(counts_p)
        boot_u = rates_from_counts(bootstrap_counts(counts_u, n_bootstrap, rng))
        boot_p = rates_from_counts(bootstrap_counts(counts_p, n_bootstrap, rng))

        with np.errstate(divide="ignore", invalid="ignore"):
            dir_score = fav_u / fav_p if fav_p > 0 else np.nan
            dir_boot = boot_u[0] / boot_p[0]
        dir_low, dir_high = _interval(dir_boot, ci)
        tpr_low, tpr_high = _interval(boot_u[1] - boot_p[1], ci)
        fpr_low, fpr_high = _interval(boot_u[2] - boot_p[2], ci)

        rows.append({
            "Feature": feature,
            "Threshold used": threshold,
            "Unprivileged group": spec["unprivileged_name"],
            "Privileged group": spec["privileged_name"],
            "n unprivileged": int(counts_u.sum()),
            "n privileged": int(counts_p.sum()),
            "n missing": int(missing.sum()),
            "Default rate (unprivileged)": dr_u,
            "Default rate (privileged)": dr_p,
            "Favourable rate (unprivileged)": fav_u,
            "Favourable rate (privileged)": fav_p,
            "DIR": dir_score,
            "DIR CI low": dir_low,
            "DIR CI high": dir_high,
            "TPR (unprivileged)": tpr_u,
            "TPR (privileged)": tpr_p,
            "TPR gap": tpr_u - tpr_p,
            "TPR gap CI low": tpr_low,
            "TPR gap CI high": tpr_high,
            "FPR (unprivileged)": fpr_u,
            "FPR (privileged)": fpr_p,
            "FPR gap": fpr_u - fpr_p,
            "FPR gap CI low": fpr_low,
            "FPR gap CI high": fpr_high,
        })
    return pd.DataFrame(rows)
