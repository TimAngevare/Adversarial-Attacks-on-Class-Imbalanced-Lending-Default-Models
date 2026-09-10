"""
Phase B: One HQ task per experiment.
Each experiment is (imbalance_ratio, model, sampler) — fully self-contained.
"""

import argparse
import json
import os
import sys
import traceback
from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import (
    DATA_PROCESSED_DIR, RESULTS_DIR, PARAMS_FILE,
    MINORITY_FRACS, MODEL_NAMES, SAMPLER_NAMES,
    RANDOM_STATE, N_FOLDS, ZOO_SKIP_MODELS,
)
from src.models_def import get_models, apply_class_weight
from src.samplers_def import get_samplers
from src.pipeline_utils import (
    build_pipeline, run_cv, compute_cv_metrics,
    compute_shap, compute_dataset_dir, run_adversarial_attacks,
    ATTACK_METHODS,
)
from src.preprocessing import make_train_test_split, make_validation_split


# Build experiment grid once
_GRID = list(product(
    range(len(MINORITY_FRACS)),
    range(len(MODEL_NAMES)),
    range(len(SAMPLER_NAMES)),
))
TOTAL_EXPERIMENTS = len(_GRID)  # 5 × 4 × 5 = 100


def get_experiment_config(experiment_id: int) -> tuple:
    fi, mi, si = _GRID[experiment_id]
    return MINORITY_FRACS[fi], MODEL_NAMES[mi], SAMPLER_NAMES[si]


def run_experiment(experiment_id: int, n_cores: int = 8):
    frac, model_name, sampler_name = get_experiment_config(experiment_id)
    out_dir = os.path.join(RESULTS_DIR, f"exp_{experiment_id:04d}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[exp {experiment_id:04d}] frac={frac}, model={model_name}, sampler={sampler_name}")

    X_full = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, f"X_full_{frac}.parquet"))
    y_full = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, f"y_full_{frac}.parquet")).squeeze()

    with open(os.path.join(DATA_PROCESSED_DIR, "feature_names.json")) as f:
        feat_names = json.load(f)

    with open(PARAMS_FILE) as f:
        best_params = json.load(f)

    X_tr, X_te, y_tr, y_te = make_train_test_split(X_full, y_full)

    n_features = X_tr.shape[1]

    model_cores = n_cores if model_name != "NeuralNetClassifier" else 1  # skorch doesn't support parallel CV
    models = get_models(n_features, n_cores=model_cores, best_params=best_params)
    samplers = get_samplers(n_jobs=n_cores)

    is_nn = model_name == "NeuralNetClassifier"
    model = clone(models[model_name]) if not is_nn else models[model_name]
    sampler = samplers[sampler_name]

    model = apply_class_weight(model, sampler_name, y_tr)

    if is_nn:
        X_cv = X_tr.astype(np.float32)
        y_cv = y_tr.to_numpy().astype(np.int64)
    else:
        X_cv = X_tr
        y_cv = y_tr

    cv_n_jobs = 1
    pipe = build_pipeline(model, sampler)

    dir_df = compute_dataset_dir(pipe, X_tr, y_tr)
    dir_df.insert(0, "experiment_id", experiment_id)
    dir_df.insert(1, "Imbalance ratio", frac)
    dir_df.insert(2, "Model", model_name)
    dir_df.insert(3, "Sampler", sampler_name)
    dir_df.to_csv(os.path.join(out_dir, "dir.csv"), index=False)

    # Parallel CV folds don't work reliably with skorch; use n_jobs=1 for NN
    cv_out = run_cv(pipe, X_cv, y_cv, n_jobs=cv_n_jobs)

    metrics = compute_cv_metrics(cv_out, frac, model_name, sampler_name, X_cv, y_cv)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    print(f"  PR-AUC={metrics['PR-AUC (mean)']:.4f}  F1={metrics['F1 (mean)']:.4f}")

   
    last_pipe = cv_out["estimator"][-1]
    last_train_idx = cv_out["indices"]["train"][-1]
    last_test_idx = cv_out["indices"]["test"][-1]

    shap_row = compute_shap(
        last_pipe, last_train_idx, last_test_idx,
        model_name, sampler_name, frac, X_cv, feat_names,
    )
    if shap_row:
        pd.DataFrame([shap_row]).to_csv(os.path.join(out_dir, "shap.csv"), index=False)


    del cv_out

    # Refit for adversarial attacks. Hold a validation split out of the fit so the
    # F1-optimal threshold that selects attack targets is calibrated on unseen data.
    X_fit, X_val, y_fit, y_val = make_validation_split(X_tr, y_tr)

    model2 = clone(models[model_name]) if not is_nn else models[model_name]
    model2 = apply_class_weight(model2, sampler_name, y_fit)
    sampler2 = get_samplers(n_jobs=n_cores)[sampler_name]
    pipe_final = build_pipeline(model2, sampler2)

    if is_nn:
        pipe_final.fit(X_fit.astype(np.float32), y_fit.to_numpy().astype(np.int64))
    else:
        pipe_final.fit(X_fit, y_fit)

    # ------------------------------------------------------------------ adversarial attacks
    # Skip ZOO for models flagged in ZOO_SKIP_MODELS (its query budget blows
    # through the wall-time there); GA still runs.
    attack_methods = {
        name: fn for name, fn in ATTACK_METHODS.items()
        if not (name == "ZOO" and model_name in ZOO_SKIP_MODELS)
    }
    attack_row = run_adversarial_attacks(
        pipe_final, X_fit, X_val, y_val, X_te, y_te, methods=attack_methods,
    )
    attack_row = {
        "experiment_id": experiment_id,
        "Imbalance ratio": frac,
        "Model": model_name,
        "Sampler": sampler_name,
        **attack_row,
    }
    pd.DataFrame([attack_row]).to_csv(os.path.join(out_dir, "attacks.csv"), index=False)
    print(
        f"  Attack success rate: GA={attack_row.get('Attack success rate (GA)', 'N/A')} "
        f"ZOO={attack_row.get('Attack success rate (ZOO)', 'N/A')}"
    )

    # ------------------------------------------------------------------ metadata
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "experiment_id": experiment_id,
            "imbalance_ratio": frac,
            "model": model_name,
            "sampler": sampler_name,
            "n_cores": n_cores,
            "status": "ok",
        }, f, indent=2)

    print(f"[exp {experiment_id:04d}] DONE → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", type=int, required=True,
                        help=f"0 to {TOTAL_EXPERIMENTS - 1}")
    parser.add_argument("--n_cores", type=int, default=8,
                        help="CPU cores allocated to this task (matches --cpus in hq submit)")
    args = parser.parse_args()

    if args.experiment_id < 0 or args.experiment_id >= TOTAL_EXPERIMENTS:
        sys.exit(f"experiment_id must be 0-{TOTAL_EXPERIMENTS - 1}, got {args.experiment_id}")

    out_dir = os.path.join(RESULTS_DIR, f"exp_{args.experiment_id:04d}")
    os.makedirs(out_dir, exist_ok=True)

    try:
        run_experiment(args.experiment_id, n_cores=args.n_cores)
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        err_path = os.path.join(out_dir, "error.txt")
        with open(err_path, "w") as f:
            f.write(tb)
        sys.exit(1)
