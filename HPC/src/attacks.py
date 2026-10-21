# Run constrained attacks and reference calculations.
import itertools
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import (
    RANDOM_STATE, DEFAULT_CLASS, N_ATTACK_CASES, MAX_CHANGED_ACTIONS, ACTION_SPECS,
    LOG_FEATURES, ATTACK_REQUIRED_COLUMNS, DEPENDENT_FEATURES, CHANGE_TOL,
    LOWER_Q, UPPER_Q, PROXIMITY_PENALTY, ATTACK_OBJECTIVE,
    GA_POP_SIZE, GA_GENERATIONS, GA_RESTARTS, GA_MUTATION_PROB, GA_MUTATION_SCALE, GA_ELITE_FRAC,
    ZOO_MAX_ITER, ZOO_RESTARTS, ZOO_LEARNING_RATE, ZOO_FD_STEP,
    GRID_LEVELS, GRID_MAX_ROWS_PER_CALL, SEPARABILITY_TOL,
)
from src.models_def import set_model_threads
from src.pipeline_utils import default_probabilities, default_probability_index


# Convert feature values back to their original scale
def to_raw(value, feature):
    value = float(value)
    return max(np.expm1(value), 0.0) if feature in LOG_FEATURES else value


# Convert feature values to the model input scale
def to_model(value, feature):
    value = max(float(value), 0.0)
    return np.log1p(value) if feature in LOG_FEATURES else value


# Convert batched feature values between raw and model scales
def to_raw_array(values, feature):
    values = np.asarray(values, dtype=float)
    return np.maximum(np.expm1(values), 0.0) if feature in LOG_FEATURES else values


# Convert batched feature values to the model input scale
def to_model_array(values, feature):
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    return np.log1p(values) if feature in LOG_FEATURES else values


# Select mutable features present in the data
def get_action_features(columns):
    return [feature for feature in ACTION_SPECS if feature in columns]


# Clip actions to [0, 1] and keep at most MAX_CHANGED_ACTIONS changes
def project_action_matrix(Z):
    Z = np.clip(np.atleast_2d(np.asarray(Z, dtype=float)), 0.0, 1.0).copy()
    Z[Z <= 1e-12] = 0.0
    d = Z.shape[1]
    too_many = (Z > 0).sum(axis=1) > MAX_CHANGED_ACTIONS
    if too_many.any():
        rows = Z[too_many]
        drop = np.argsort(rows, axis=1, kind="stable")[:, : d - MAX_CHANGED_ACTIONS]
        np.put_along_axis(rows, drop, 0.0, axis=1)
        Z[too_many] = rows
    return Z


def project_action_vector(z):
    return project_action_matrix(z)[0]


# Calculate training quantiles for the attack bounds
def make_attack_context(X_reference, action_features):
    quantiles = {}
    for feature in action_features:
        values = to_raw_array(X_reference[feature].to_numpy(), feature)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            quantiles[feature] = (np.nan, np.nan)
        else:
            quantiles[feature] = (
                float(np.quantile(values, LOWER_Q)),
                float(np.quantile(values, UPPER_Q)),
            )
    return {"action_features": list(action_features), "quantiles": quantiles}


# Bound relative changes using training quantiles while allowing no change
def get_bounds(X_rows, context):
    features = context["action_features"]
    n = len(X_rows)
    if not features:
        empty = np.empty((n, 0))
        return empty, empty, empty

    original = np.column_stack(
        [to_raw_array(X_rows[feature].to_numpy(), feature) for feature in features]
    )
    low = original.copy()
    high = original.copy()

    for j, feature in enumerate(features):
        q_low, q_high = context["quantiles"][feature]
        if not np.isfinite(q_low):
            continue
        spec = ACTION_SPECS[feature]
        o = original[:, j]
        if spec["direction"] == "increase":
            high[:, j] = np.maximum(np.minimum(o * (1 + spec["max_change"]), q_high), o)
        else:
            floor = np.maximum(np.maximum(o * (1 - spec["max_change"]), q_low), 0.0)
            low[:, j] = np.minimum(floor, o)

    return original, low, high


# Apply feature changes and update their dependent features
def apply_actions(X_rows, Z, row_index, bounds, context):
    features = context["action_features"]
    Z = project_action_matrix(Z)
    row_index = np.asarray(row_index, dtype=int)
    original, low, high = (b[row_index] for b in bounds)

    X_adv = X_rows.iloc[row_index].reset_index(drop=True)
    changed = np.zeros(Z.shape, dtype=bool)

    for j, feature in enumerate(features):
        o = original[:, j]
        if ACTION_SPECS[feature]["direction"] == "increase":
            new = o + Z[:, j] * (high[:, j] - o)
        else:
            new = o - Z[:, j] * (o - low[:, j])
        new = np.clip(new, low[:, j], high[:, j])

        moved = np.abs(new - o) > CHANGE_TOL * np.maximum(1.0, np.abs(o))
        mask = (Z[:, j] > 0) & moved
        changed[:, j] = mask
        if not mask.any():
            continue

        column = X_adv[feature].to_numpy(dtype=float, copy=True)
        column[mask] = to_model_array(new[mask], feature)
        X_adv[feature] = column

        for dependent, rule in DEPENDENT_FEATURES.get(feature, []):
            if dependent not in X_adv.columns:
                continue
            dep = X_adv[dependent].to_numpy(dtype=float, copy=True)
            if rule == "proportional":
                ok = mask & (o > 0) & np.isfinite(dep)
                ratio = np.divide(new, o, out=np.ones_like(new), where=o > 0)
            else:  # "inverse"
                ok = mask & (new > 0) & np.isfinite(dep)
                ratio = np.divide(o, new, out=np.ones_like(new), where=new > 0)
            dep[ok] = np.maximum(dep[ok] * ratio[ok], 0.0)
            X_adv[dependent] = dep

    Z_effective = np.where(changed, Z, 0.0)
    return X_adv, Z_effective, changed


# Convert default probabilities to log-odds
def to_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


# Count the directly mutable features that changed
def changed_features(x_adv, x_original, action_features):
    return int(
        sum(
            abs(
                to_raw(x_adv.iloc[0][feature], feature)
                - to_raw(x_original.iloc[0][feature], feature)
            ) > 1e-8
            for feature in action_features
        )
    )


# Combine default log-odds with the action-size penalty
def attack_objective(probabilities, Z_effective, objective=None):
    objective = objective or ATTACK_OBJECTIVE
    costs = np.asarray(Z_effective, dtype=float).mean(axis=1)
    if objective == "logit":
        base = to_logit(probabilities)
    elif objective == "probability":
        base = np.asarray(probabilities, dtype=float)
    else:
        raise ValueError(f"Unknown objective {objective!r}")
    return base + PROXIMITY_PENALTY * costs, costs


# Evaluate candidate actions in one model call
def evaluate_candidates(Z, model, x_original, bounds, context, query_counter, objective=None):
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    X_adv, Z_effective, changed = apply_actions(
        x_original, Z, np.zeros(len(Z), dtype=int), bounds, context
    )
    probabilities = default_probabilities(model, X_adv)
    query_counter["count"] += len(Z)
    scores, costs = attack_objective(probabilities, Z_effective, objective)
    return scores, probabilities, costs, Z_effective, changed, X_adv


# Track the cheapest success and the lowest default probability
class SearchTracker:

    def __init__(self, threshold):
        self.threshold = threshold
        self.success = None
        self.lowest = None

    @staticmethod
    def _pick(i, probabilities, costs, changed, X_adv):
        return {
            "probability": float(probabilities[i]),
            "cost": float(costs[i]),
            "changed": int(np.asarray(changed)[i].sum()),
            "x_adv": X_adv.iloc[[i]].copy(),
        }

    # Update the cheapest success and lowest probability found
    def update(self, probabilities, costs, changed, X_adv):
        probabilities = np.asarray(probabilities, dtype=float)
        costs = np.asarray(costs, dtype=float)
        n_changed = np.asarray(changed).sum(axis=1)

        hits = np.flatnonzero(probabilities < self.threshold)
        if len(hits):
            order = np.lexsort((probabilities[hits], n_changed[hits], costs[hits]))
            i = hits[order[0]]
            if self.success is None or costs[i] < self.success["cost"] - 1e-12:
                self.success = self._pick(i, probabilities, costs, changed, X_adv)

        i = int(np.argmin(probabilities))
        if self.lowest is None or probabilities[i] < self.lowest["probability"]:
            self.lowest = self._pick(i, probabilities, costs, changed, X_adv)

    @property
    def found(self):
        return self.success is not None

    def chosen(self):
        return self.success if self.found else self.lowest


# Record the cheapest success or lowest-probability candidate
def create_result(method, original_probability, tracker, x_original, query_counter,
                  runtime_seconds, action_features):
    chosen = tracker.chosen()
    adversarial_probability = chosen["probability"]
    threshold = tracker.threshold

    result = {
        "Method": method,
        "Success": int(adversarial_probability < threshold),
        "Original default probability": original_probability,
        "Adversarial default probability": adversarial_probability,
        "Lowest probability found": tracker.lowest["probability"],
        "Probability reduction": original_probability - adversarial_probability,
        "Log-odds reduction": float(to_logit(original_probability) - to_logit(adversarial_probability)),
        "Changed features": chosen["changed"],
        "Action cost": chosen["cost"],
        "Queries": int(query_counter["count"]),
        "Runtime seconds": runtime_seconds,
        "Threshold": threshold,
    }

    for feature in action_features:
        before = to_raw(x_original.iloc[0][feature], feature)
        after = to_raw(chosen["x_adv"].iloc[0][feature], feature)
        result[f"{feature}_before"] = before
        result[f"{feature}_after"] = after
        result[f"{feature}_relative_change"] = (
            (after - before) / before if before != 0 else np.nan
        )

    return result


# Search for feasible changes using a genetic algorithm
def permute_attack(
    model,
    x_original,
    context,
    threshold,
    seed=RANDOM_STATE,
    objective=None,
):
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    query_counter = {"count": 0}

    action_features = context["action_features"]
    d = len(action_features)
    bounds = get_bounds(x_original, context)
    tracker = SearchTracker(threshold)

    # Evaluate the unchanged application
    _, probabilities, costs, _, changed, X_adv = evaluate_candidates(
        np.zeros((1, d)), model, x_original, bounds, context, query_counter, objective
    )
    original_probability = float(probabilities[0])
    tracker.update(probabilities, costs, changed, X_adv)

    n_elite = max(2, int(GA_POP_SIZE * GA_ELITE_FRAC))

    def finish():
        return create_result(
            method="GA",
            original_probability=original_probability,
            tracker=tracker,
            x_original=x_original,
            query_counter=query_counter,
            runtime_seconds=time.perf_counter() - start,
            action_features=action_features,
        )

    for _ in range(GA_RESTARTS):
        population = rng.uniform(0.0, 1.0, size=(GA_POP_SIZE, d))
        population[0] = 0.0
        population = project_action_matrix(population)

        for _ in range(GA_GENERATIONS):
            # Evaluate the population in one model call
            scores, probabilities, costs, Z_effective, changed, X_adv = evaluate_candidates(
                population, model, x_original, bounds, context, query_counter, objective
            )
            tracker.update(probabilities, costs, changed, X_adv)

            # Stop when the population contains a successful change
            if tracker.found:
                return finish()

            order = np.argsort(scores, kind="stable")
            elites = Z_effective[order[:n_elite]]

            children = [elite.copy() for elite in elites]
            while len(children) < GA_POP_SIZE:
                parent_1 = elites[rng.integers(0, len(elites))]
                parent_2 = elites[rng.integers(0, len(elites))]

                crossover_mask = rng.random(d) < 0.5
                child = np.where(crossover_mask, parent_1, parent_2)

                mutation_mask = rng.random(d) < GA_MUTATION_PROB
                child[mutation_mask] += rng.normal(
                    loc=0.0,
                    scale=GA_MUTATION_SCALE,
                    size=mutation_mask.sum(),
                )

                if rng.random() < 0.30:
                    child[rng.integers(0, d)] = 0.0

                children.append(child)

            population = project_action_matrix(np.asarray(children))

    return finish()


# Search for feasible changes using finite differences
def zoo_attack(
    model,
    x_original,
    context,
    threshold,
    seed=RANDOM_STATE,
    objective=None,
):
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    query_counter = {"count": 0}

    action_features = context["action_features"]
    d = len(action_features)
    bounds = get_bounds(x_original, context)
    tracker = SearchTracker(threshold)

    # Evaluate candidates and update the search tracker
    def evaluate(Z):
        scores, probabilities, costs, Z_effective, changed, X_adv = evaluate_candidates(
            Z, model, x_original, bounds, context, query_counter, objective
        )
        # Check every evaluated candidate for success
        tracker.update(probabilities, costs, changed, X_adv)
        return scores, probabilities, Z_effective

    def finish():
        return create_result(
            method="ZOO",
            original_probability=original_probability,
            tracker=tracker,
            x_original=x_original,
            query_counter=query_counter,
            runtime_seconds=time.perf_counter() - start,
            action_features=action_features,
        )

    _, probabilities, _ = evaluate(np.zeros((1, d)))
    original_probability = float(probabilities[0])

    beta_1, beta_2 = 0.90, 0.999

    for restart in range(ZOO_RESTARTS):
        if restart == 0:
            z = np.zeros(d)
        else:
            z = rng.uniform(low=0.0, high=0.25, size=d)

        scores, _, Z_effective = evaluate(z[None, :])
        z = Z_effective[0]
        current_score = scores[0]
        if tracker.found:
            return finish()

        m = np.zeros(d)
        v = np.zeros(d)

        for iteration in range(1, ZOO_MAX_ITER + 1):
            # Estimate gradients using paired finite-difference probes
            probes = np.repeat(z[None, :], 2 * d, axis=0)
            for j in range(d):
                probes[2 * j, j] = min(1.0, z[j] + ZOO_FD_STEP)
                probes[2 * j + 1, j] = max(0.0, z[j] - ZOO_FD_STEP)

            probe_scores, _, _ = evaluate(probes)
            if tracker.found:
                return finish()

            step = np.maximum(
                probes[0::2, :].diagonal() - probes[1::2, :].diagonal(),
                1e-8,
            )
            gradient = (probe_scores[0::2] - probe_scores[1::2]) / step

            m = beta_1 * m + (1.0 - beta_1) * gradient
            v = beta_2 * v + (1.0 - beta_2) * gradient**2
            m_hat = m / (1.0 - beta_1**iteration)
            v_hat = v / (1.0 - beta_2**iteration)

            proposal = z - ZOO_LEARNING_RATE * m_hat / (np.sqrt(v_hat) + 1e-8)
            proposal_scores, _, proposal_Z = evaluate(proposal[None, :])
            if tracker.found:
                return finish()

            if proposal_scores[0] <= current_score:
                z = proposal_Z[0]
                current_score = proposal_scores[0]

    return finish()


# Build feasible grid actions with no change listed first
def make_action_grid(n_actions, levels=None, max_changed=None):
    levels = GRID_LEVELS if levels is None else levels
    max_changed = MAX_CHANGED_ACTIONS if max_changed is None else max_changed
    values = np.linspace(0.0, 1.0, levels)
    grid = np.array(list(itertools.product(values, repeat=n_actions)), dtype=float)
    grid = grid[(grid > 0).sum(axis=1) <= max_changed]
    is_zero = (grid == 0).all(axis=1)
    return np.vstack([grid[is_zero], grid[~is_zero]])


# Evaluate grid candidates in batches for each borrower
def grid_attack(model, X_rows, context, threshold, levels=None,
                max_rows_per_call=None):
    action_features = context["action_features"]
    grid = make_action_grid(len(action_features), levels)
    k = len(grid)
    max_rows_per_call = GRID_MAX_ROWS_PER_CALL if max_rows_per_call is None else max_rows_per_call
    per_call = max(1, max_rows_per_call // k)

    records = []
    for start in range(0, len(X_rows), per_call):
        t0 = time.perf_counter()
        chunk = X_rows.iloc[start:start + per_call]
        m = len(chunk)
        bounds = get_bounds(chunk, context)

        X_adv, Z_effective, changed = apply_actions(
            chunk,
            np.tile(grid, (m, 1)),
            np.repeat(np.arange(m), k),
            bounds,
            context,
        )
        p = default_probabilities(model, X_adv).reshape(m, k)
        cost = Z_effective.mean(axis=1).reshape(m, k)
        n_changed = changed.sum(axis=1).reshape(m, k)

        success = p < threshold
        any_success = success.any(axis=1)

        # Select the cheapest success or the lowest-probability candidate
        j_success = np.zeros(m, dtype=int)
        for i in np.flatnonzero(any_success):
            hits = np.flatnonzero(success[i])
            order = np.lexsort((p[i, hits], n_changed[i, hits], cost[i, hits]))
            j_success[i] = hits[order[0]]
        j_lowest = p.argmin(axis=1)
        j = np.where(any_success, j_success, j_lowest)

        rows = np.arange(m)
        selected = rows * k + j
        p0 = p[:, 0]
        p_adv = p[rows, j]
        elapsed = time.perf_counter() - t0

        record = pd.DataFrame({
            "Method": "Grid",
            "Test row": chunk.index.to_numpy(),
            "Success": any_success.astype(int),
            "Original default probability": p0,
            "Adversarial default probability": p_adv,
            "Lowest probability found": p.min(axis=1),
            "Probability reduction": p0 - p_adv,
            "Log-odds reduction": to_logit(p0) - to_logit(p_adv),
            "Changed features": n_changed[rows, j],
            "Action cost": cost[rows, j],
            "Queries": k,
            "Runtime seconds": elapsed / max(m, 1),
            "Threshold": threshold,
        })

        original = bounds[0]
        for i_feature, feature in enumerate(action_features):
            before = original[:, i_feature]
            after = to_raw_array(X_adv[feature].to_numpy()[selected], feature)
            record[f"{feature}_before"] = before
            record[f"{feature}_after"] = after
            record[f"{feature}_relative_change"] = np.divide(
                after - before, before, out=np.full(m, np.nan), where=before != 0
            )

        records.append(record)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


# Check whether the pipeline uses Logistic Regression
def is_separable_model(pipeline):
    return pipeline.steps[-1][1].__class__.__name__ == "LogisticRegression"


# Read the fitted coefficients and account for feature scaling
def linear_logit_coefficients(model, features):
    if hasattr(model, "logit_coefficients"):
        coefficients = model.logit_coefficients()
        return {f: float(coefficients.get(f, 0.0)) for f in features}
    if not hasattr(model, "named_steps") or not is_separable_model(model):
        raise RuntimeError("The exact reference needs a Logistic Regression pipeline.")
    unexpected = [name for name, _ in model.steps if name not in ("imputer", "scaler", "sampler", "clf")]
    if unexpected:
        raise RuntimeError(f"Unexpected pipeline steps for the exact reference: {unexpected}")
    names = list(model.feature_names_in_)
    clf = model.named_steps["clf"]
    sign = 1.0 if default_probability_index(model) == 1 else -1.0
    coef = sign * np.asarray(clf.coef_[0], dtype=float)
    scaler = model.named_steps.get("scaler")
    scale = np.ones(len(names))
    if scaler is not None and getattr(scaler, "scale_", None) is not None:
        scale = np.asarray(scaler.scale_, dtype=float)
    return {f: float(coef[names.index(f)] / scale[names.index(f)]) for f in features if f in names}


# Find the best feature value using endpoints and stationary points
def _best_single_change(x0, low, high, a, dep0, b, rule, log_feature):
    t = np.log1p if log_feature else (lambda v: v)
    has_dep = rule is not None
    b = np.where(np.isfinite(dep0) & (x0 > 0), b, 0.0) if has_dep else np.zeros_like(x0)
    dep0 = np.nan_to_num(dep0) if has_dep else np.zeros_like(x0)

    # Calculate the log-odds change for one feature and its dependent
    def delta(x):
        out = a * (t(x) - t(x0))
        if rule == "proportional":
            out = out + b * dep0 * (np.divide(x, x0, out=np.ones_like(x), where=x0 > 0) - 1.0)
        elif rule == "inverse":
            out = out + b * dep0 * (np.divide(x0, x, out=np.ones_like(x), where=x > 0) - 1.0)
        return out

    candidates = [x0.copy(), low.copy(), high.copy()]
    with np.errstate(divide="ignore", invalid="ignore"):
        if log_feature and rule == "proportional":
            # a / (1 + x) + B = 0  with  B = b * dep0 / x0
            B = np.divide(b * dep0, x0, out=np.zeros_like(x0), where=x0 > 0)
            candidates.append(np.where(B != 0, -a / B - 1.0, np.nan))
        elif log_feature and rule == "inverse":
            # a / (1 + x) - C / x^2 = 0  ->  a x^2 - C x - C = 0  with  C = b * dep0 * x0
            C = b * dep0 * x0
            disc = C**2 + 4 * a * C
            root = np.sqrt(np.where(disc >= 0, disc, np.nan))
            denominator = np.where(a != 0, 2 * a, np.nan)
            candidates.append((C + root) / denominator)
            candidates.append((C - root) / denominator)
        elif not log_feature and rule == "inverse":
            # a - C / x^2 = 0  ->  x = sqrt(C / a)
            ratio = np.divide(b * dep0 * x0, a, out=np.full_like(x0, np.nan), where=a != 0)
            candidates.append(np.sqrt(np.where(ratio > 0, ratio, np.nan)))
        # Monotone terms only need endpoint candidates

    X = np.column_stack(candidates)
    inside = np.isfinite(X) & (X >= low[:, None]) & (X <= high[:, None])
    X = np.where(inside, X, x0[:, None])
    D = np.column_stack([delta(X[:, k]) for k in range(X.shape[1])])
    k_best = np.nanargmin(np.where(np.isfinite(D), D, np.inf), axis=1)
    rows = np.arange(len(x0))
    return X[rows, k_best], D[rows, k_best]


# Calculate and verify the coefficient-based LR reference
def separable_reference_attack(model, X_rows, context, threshold, max_rows_per_call=None):
    action_features = context["action_features"]
    d = len(action_features)
    dependents = {f: (DEPENDENT_FEATURES.get(f) or [(None, None)])[0] for f in action_features}
    needed = action_features + [dep for dep, _ in dependents.values() if dep is not None]
    coefficients = linear_logit_coefficients(model, needed)
    max_rows_per_call = GRID_MAX_ROWS_PER_CALL if max_rows_per_call is None else max_rows_per_call
    per_call = max(1, max_rows_per_call // (d + 2))

    records = []
    for start in range(0, len(X_rows), per_call):
        t0 = time.perf_counter()
        chunk = X_rows.iloc[start:start + per_call]
        m = len(chunk)
        original, low, high = get_bounds(chunk, context)

        z_best = np.zeros((m, d))
        delta = np.zeros((m, d))
        for j, feature in enumerate(action_features):
            dep, rule = dependents[feature]
            dep0 = (chunk[dep].to_numpy(dtype=float) if dep is not None and dep in chunk.columns
                    else np.full(m, np.nan))
            x_best, d_best = _best_single_change(
                original[:, j], low[:, j], high[:, j],
                coefficients.get(feature, 0.0), dep0,
                coefficients.get(dep, 0.0) if dep is not None else 0.0,
                rule if dep is not None and dep in chunk.columns else None,
                feature in LOG_FEATURES,
            )
            if ACTION_SPECS[feature]["direction"] == "increase":
                span = high[:, j] - original[:, j]
                z = np.divide(x_best - original[:, j], span, out=np.zeros(m), where=span > 0)
            else:
                span = original[:, j] - low[:, j]
                z = np.divide(original[:, j] - x_best, span, out=np.zeros(m), where=span > 0)
            z_best[:, j] = np.clip(z, 0.0, 1.0)
            delta[:, j] = np.minimum(d_best, 0.0)          # "no change" is always allowed

        # Combine up to MAX_CHANGED_ACTIONS beneficial changes
        order = np.argsort(delta, axis=1, kind="stable")
        keep = np.zeros((m, d), dtype=bool)
        np.put_along_axis(keep, order[:, :MAX_CHANGED_ACTIONS], True, axis=1)
        keep &= delta < 0
        Z_final = np.where(keep, z_best, 0.0)

        # Verify the original, individual optima and combined change
        Z_all = np.zeros((m, d + 2, d))
        for j in range(d):
            Z_all[:, 1 + j, j] = z_best[:, j]
        Z_all[:, d + 1] = Z_final
        X_adv, Z_eff, changed = apply_actions(
            chunk, Z_all.reshape(-1, d), np.repeat(np.arange(m), d + 2),
            (original, low, high), context,
        )
        logits = to_logit(default_probabilities(model, X_adv)).reshape(m, d + 2)
        logit_0 = logits[:, 0]
        predicted = np.column_stack(
            [logit_0 + delta[:, j] * (z_best[:, j] > 0) for j in range(d)]
            + [logit_0 + np.where(keep, delta, 0.0).sum(axis=1)]
        )
        gap = np.abs(logits[:, 1:] - predicted)
        if np.nanmax(gap, initial=0.0) > SEPARABILITY_TOL:
            raise RuntimeError(
                f"Closed-form log-odds differ from the model (gap {np.nanmax(gap):.2e}); "
                "the exact reference only applies to Logistic Regression."
            )

        final_rows = np.arange(m) * (d + 2) + d + 1
        p_0 = 1 / (1 + np.exp(-logit_0))
        p_final = 1 / (1 + np.exp(-logits[:, -1]))
        elapsed = time.perf_counter() - t0
        record = pd.DataFrame({
            "Method": "Exact (LR)",
            "Test row": chunk.index.to_numpy(),
            "Success": (p_final < threshold).astype(int),
            "Original default probability": p_0,
            "Adversarial default probability": p_final,
            "Lowest probability found": p_final,
            "Probability reduction": p_0 - p_final,
            "Log-odds reduction": logit_0 - logits[:, -1],
            "Changed features": changed.reshape(m, d + 2, d)[:, -1].sum(axis=1),
            "Action cost": Z_eff.reshape(m, d + 2, d)[:, -1].mean(axis=1),
            "Queries": d + 2,
            "Runtime seconds": elapsed / max(m, 1),
            "Threshold": threshold,
        })
        for i_feature, feature in enumerate(action_features):
            before = original[:, i_feature]
            after = to_raw_array(X_adv[feature].to_numpy()[final_rows], feature)
            record[f"{feature}_before"] = before
            record[f"{feature}_after"] = after
            record[f"{feature}_relative_change"] = np.divide(
                after - before, before, out=np.full(m, np.nan), where=before != 0
            )
        records.append(record)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


SEARCH_ATTACKS = {
    "GA": permute_attack,   # PermuteAttack-inspired genetic algorithm
    "ZOO": zoo_attack,      # ZOO-inspired finite differences (Adam)
}
EXACT_METHOD = "Exact (LR)"   # exact reference, only for Logistic Regression (Section 7.4)
ATTACK_METHODS = [EXACT_METHOD, "Grid"] + list(SEARCH_ATTACKS)


# Calculate the Wilson interval for the success rate
def wilson_interval(successes, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


# Summarise attack success, eligibility and post-attack recall
def summarise_attack_records(records, clean, methods=None):
    methods = methods or ATTACK_METHODS
    n_defaults = clean["N test defaults"]
    n_caught = clean["N caught (clean)"]
    n_eligible = clean["N eligible"]

    rows = []
    for method in methods:
        r = records[records["Method"] == method] if len(records) else pd.DataFrame()
        n_attacked = len(r)
        n_success = int(r["Success"].sum()) if n_attacked else 0
        asr = n_success / n_attacked if n_attacked else np.nan
        asr_low, asr_high = wilson_interval(n_success, n_attacked)

        # Calculate recall remaining after the attack
        def post_recall(rate):
            if n_defaults == 0:
                return np.nan
            if n_eligible == 0:
                return n_caught / n_defaults
            return (n_caught - rate * n_eligible) / n_defaults

        succeeded = r[r["Success"] == 1] if n_attacked else r
        rows.append({
            **clean,
            "Method": method,
            "N attacked": n_attacked,
            "N successful": n_success,
            "Attack success rate": asr,
            "ASR CI low": asr_low,
            "ASR CI high": asr_high,
            "Post-attack recall": post_recall(asr),
            "Post-attack recall basis": (
                "all eligible" if n_attacked == n_eligible
                else f"estimated from {n_attacked} of {n_eligible} eligible"
            ),
            "Post-attack recall CI low": post_recall(asr_high),
            "Post-attack recall CI high": post_recall(asr_low),
            "Mean changed features (success)": succeeded["Changed features"].mean() if len(succeeded) else np.nan,
            "Mean action cost (success)": succeeded["Action cost"].mean() if len(succeeded) else np.nan,
            "Mean prob reduction (success)": succeeded["Probability reduction"].mean() if len(succeeded) else np.nan,
            "Mean prob reduction (all)": r["Probability reduction"].mean() if n_attacked else np.nan,
            "Median queries": r["Queries"].median() if n_attacked else np.nan,
            "Median runtime": r["Runtime seconds"].median() if n_attacked else np.nan,
        })
    return pd.DataFrame(rows)


# Run references and sampled search attacks for one fitted configuration
def run_attacks_for_configuration(pipeline, threshold, X_reference, X_test, y_test,
                                  configuration_number, search_attacks=None,
                                  n_search_cases="default", n_workers=1, start_method="spawn"):
    info = {}
    search_attacks = SEARCH_ATTACKS if search_attacks is None else search_attacks
    n_search_cases = N_ATTACK_CASES if n_search_cases == "default" else n_search_cases

    action_features = get_action_features(X_test.columns)
    if len(action_features) != len(ACTION_SPECS):
        raise ValueError("One or more configured action features are missing.")
    context = make_attack_context(X_reference, action_features)

    y = np.asarray(y_test).astype(int)
    p_test = default_probabilities(pipeline, X_test)
    is_default = y == DEFAULT_CLASS
    caught = is_default & (p_test >= threshold)
    complete = X_test[ATTACK_REQUIRED_COLUMNS].notna().all(axis=1).to_numpy()
    eligible = caught & complete

    clean = {
        "Threshold": threshold,
        "N test defaults": int(is_default.sum()),
        "N caught (clean)": int(caught.sum()),
        "N eligible": int(eligible.sum()),
        "Clean recall": caught.sum() / is_default.sum() if is_default.sum() else np.nan,
    }

    X_eligible = X_test.loc[eligible]
    parts = []

    t0 = time.perf_counter()
    grid_records = grid_attack(pipeline, X_eligible, context, threshold)
    info["grid_seconds"] = time.perf_counter() - t0
    if len(grid_records):
        # Check that zero action reproduces the original probabilities
        if not np.allclose(grid_records["Original default probability"], p_test[eligible], atol=1e-6):
            raise RuntimeError("Grid attack does not reproduce the clean test probabilities.")

    methods = ["Grid"] + list(search_attacks)
    if is_separable_model(pipeline):
        t0 = time.perf_counter()
        exact_records = separable_reference_attack(pipeline, X_eligible, context, threshold)
        info["exact_seconds"] = time.perf_counter() - t0
        methods = [EXACT_METHOD] + methods
    else:
        exact_records = pd.DataFrame()

    if n_search_cases is None or n_search_cases >= len(X_eligible):
        search_index = X_eligible.index
    else:
        search_index = X_eligible.sample(
            n=n_search_cases, random_state=RANDOM_STATE + configuration_number
        ).index

    for reference in (exact_records, grid_records):
        if len(reference):
            reference["In search sample"] = reference["Test row"].isin(search_index)
            reference["Seed"] = np.nan
            parts.append(reference)

    # Assign each borrower a seed before distributing search jobs
    jobs = []
    for method, attack_fn in search_attacks.items():
        method_offset = list(SEARCH_ATTACKS).index(method) if method in SEARCH_ATTACKS else 0
        for case_number, idx in enumerate(search_index):
            seed = (RANDOM_STATE + 100000 * configuration_number
                    + 50000 * method_offset + case_number)
            jobs.append((method, idx, seed))

    t0 = time.perf_counter()
    results = run_search_jobs(pipeline, X_test.loc[list(search_index)], context, threshold,
                              jobs, search_attacks, n_workers, start_method, info)
    info["search_seconds"] = time.perf_counter() - t0
    info["search_jobs"] = len(jobs)

    for method in search_attacks:
        rows = [r for r in results if r["Method"] == method]
        if rows:
            parts.append(pd.DataFrame(rows))

    records = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    summary = summarise_attack_records(records, clean, methods)
    return records, summary, info


# Run GA and ZOO searches in worker processes

_WORKER = {}


# Set each search worker to one model thread
def _init_worker(pipeline, X_cases, context, threshold, search_attacks):
    os.environ["OMP_NUM_THREADS"] = "1"
    limits = threadpool_limits(1)   # kept alive for the life of the worker
    set_model_threads(pipeline.steps[-1][1], 1)
    _WORKER.update(limits=limits, pipeline=pipeline, X_cases=X_cases, context=context,
                   threshold=threshold, search_attacks=search_attacks)


# Run one borrower search using the worker model and assigned seed
def _run_job(job):
    method, idx, seed = job
    w = _WORKER
    t0 = time.perf_counter()
    result = w["search_attacks"][method](
        w["pipeline"], w["X_cases"].loc[[idx]], w["context"], w["threshold"], seed=seed
    )
    result.update({"Test row": idx, "In search sample": True, "Seed": seed,
                   "Worker pid": os.getpid(), "Job seconds": time.perf_counter() - t0})
    return result


# Run search jobs in parallel and record any fallback to serial execution
def run_search_jobs(pipeline, X_cases, context, threshold, jobs, search_attacks,
                    n_workers, start_method, info):
    if not jobs:
        return []
    n_workers = max(1, min(int(n_workers or 1), len(jobs)))
    info["search_workers"] = n_workers
    info["search_start_method"] = start_method if n_workers > 1 else "serial"

    if n_workers > 1:
        try:
            ctx = mp.get_context(start_method)
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx, initializer=_init_worker,
                initargs=(pipeline, X_cases, context, threshold, search_attacks),
            ) as pool:
                return list(pool.map(_run_job, jobs, chunksize=1))
        except Exception:
            info["search_pool_error"] = traceback.format_exc()
            print("Attack pool failed; running GA/ZOO serially instead:\n" + info["search_pool_error"])
            info["search_start_method"] = "serial (pool failed)"
            info["search_workers"] = 1

    # Run serial searches with one model thread
    clf = pipeline.steps[-1][1]
    n_jobs_before = clf.get_params().get("n_jobs") if hasattr(clf, "get_params") else None
    set_model_threads(clf, 1)
    try:
        _WORKER.update(pipeline=pipeline, X_cases=X_cases, context=context,
                       threshold=threshold, search_attacks=search_attacks)
        return [_run_job(job) for job in jobs]
    finally:
        if n_jobs_before not in (None, 1):
            set_model_threads(clf, n_jobs_before)
