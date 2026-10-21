# Run one pricing sensitivity configuration on the training pool.
import argparse
import json
import os
import sys
import traceback
from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from src.config import (
    DATA_PROCESSED_DIR, RESULTS_DIR, PARAMS_FILE, MINORITY_FRACS,
    PRICING_VARIANTS, PRICING_SENSITIVITY_MODELS, PRICING_SENSITIVITY_SAMPLERS,
)

_GRID = list(product(MINORITY_FRACS, list(PRICING_VARIANTS),
                     PRICING_SENSITIVITY_MODELS, PRICING_SENSITIVITY_SAMPLERS))


# Evaluate one pricing sensitivity configuration and save its scores
def run_task(task_id: int, n_cores: int, out_dir: str, attempt):
    from main import configure_threads
    configure_threads(n_cores)
    from src.diagnostics import runtime_report, StageTimer
    from src.models_def import get_models, apply_class_weight
    from src.samplers_def import get_samplers
    from src.pipeline_utils import build_pipeline, make_cv
    from src.preprocessing import load_dataset, load_test_ids, split_train_test

    frac, variant, model_name, sampler_name = _GRID[task_id]
    print(f"[pricing {task_id:04d}] frac={frac}, variant={variant}, model={model_name}, sampler={sampler_name}")
    attempt.update(runtime=runtime_report(n_cores, label=f"pricing {task_id:04d}"))
    timer = StageTimer(n_cores)

    with timer.stage("cv"):
        X, y = load_dataset(DATA_PROCESSED_DIR, frac)
        extra = PRICING_VARIANTS[variant]
        if extra:
            pricing = pd.read_parquet(os.path.join(DATA_PROCESSED_DIR, f"pricing_{frac}.parquet"))
            X = X.join(pricing[extra])
        X_tr, _, y_tr, _ = split_train_test(X, y, load_test_ids(DATA_PROCESSED_DIR))

        with open(PARAMS_FILE) as f:
            best_params = json.load(f)
        model = get_models(X_tr.shape[1], n_cores=n_cores, best_params=best_params)[model_name]
        estimator = apply_class_weight(clone(model), sampler_name, y_tr)
        sampler = get_samplers(n_jobs=n_cores)[sampler_name]
        X_cv, y_cv = X_tr, y_tr
        if model_name == "NeuralNetClassifier":
            estimator.set_params(module__input_dim=X_tr.shape[1])
            X_cv, y_cv = X_tr.astype(np.float32), y_tr.to_numpy().astype(np.int64)
        pipe = build_pipeline(estimator, sampler)
        cv_out = cross_validate(pipe, X_cv, y_cv, cv=make_cv(),
                                scoring=("roc_auc", "average_precision"), n_jobs=1)

    row = {"Imbalance ratio": frac, "Variant": variant, "Model": model_name, "Sampler": sampler_name,
           "ROC-AUC (mean)": float(cv_out["test_roc_auc"].mean()),
           "PR-AUC (mean)": float(cv_out["test_average_precision"].mean())}
    pd.DataFrame([row]).to_csv(os.path.join(out_dir, "pricing.csv"), index=False)
    print(f"  ROC-AUC={row['ROC-AUC (mean)']:.4f}  PR-AUC={row['PR-AUC (mean)']:.4f}")
    attempt.stage_done("pricing")
    attempt.record_timings(timer.summary())
    attempt.finish("ok")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int)
    parser.add_argument("--n_cores", type=int, default=None)
    parser.add_argument("--count", action="store_true", help="Print the number of tasks and exit")
    args = parser.parse_args()

    if args.count:
        print(len(_GRID))
        sys.exit(0)
    if args.task_id is None or not 0 <= args.task_id < len(_GRID):
        sys.exit(f"task_id must be 0-{len(_GRID) - 1}")

    from main import default_cores
    n_cores = args.n_cores or default_cores()
    out_dir = os.path.join(RESULTS_DIR, f"pricing_{args.task_id:04d}")
    from src.attempts import Attempt
    frac, variant, model_name, sampler_name = _GRID[args.task_id]
    attempt = Attempt(out_dir, identity={"task_id": args.task_id, "imbalance_ratio": frac,
                                         "variant": variant, "model": model_name, "sampler": sampler_name},
                      stages=["pricing"])
    try:
        run_task(args.task_id, n_cores, out_dir, attempt)
    except BaseException:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        attempt.finish("failed", error=tb)
        sys.exit(1)
