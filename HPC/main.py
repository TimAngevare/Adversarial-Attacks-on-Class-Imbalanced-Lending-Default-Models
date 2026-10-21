# Run cross-validation, fairness and attacks for one experiment.

import argparse
import json
import os
import sys
import time
import traceback
from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

from src.config import (
    DATA_PROCESSED_DIR, RESULTS_DIR, PARAMS_FILE,
    MINORITY_FRACS, MODEL_NAMES, SAMPLER_NAMES,
    RANDOM_STATE, ZOO_SKIP_MODELS, FAVOURABLE_LABEL, ATTACK_START_METHOD,
)
from src.config import KMEANS_SMOTE_FALLBACK_LABEL
from src.diagnostics import runtime_report, StageTimer, FamilyMemoryMonitor, write_json
from src.attempts import Attempt

# Map experiment IDs to prevalence, model and sampler in a fixed order
_GRID = list(product(
    range(len(MINORITY_FRACS)),
    range(len(MODEL_NAMES)),
    range(len(SAMPLER_NAMES)),
))
TOTAL_EXPERIMENTS = len(_GRID)  # 5 x 4 x 5 = 100
ALL_STAGES = ["cv", "final", "attacks"]
_THREAD_LIMITS = None


# Map the experiment ID to its prevalence, model and sampler
def get_experiment_config(experiment_id: int) -> tuple:
    fi, mi, si = _GRID[experiment_id]
    return MINORITY_FRACS[fi], MODEL_NAMES[mi], SAMPLER_NAMES[si]


# Limit library threads to the cores assigned to the task
def configure_threads(n_cores: int):
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(var, str(n_cores))
    from threadpoolctl import threadpool_limits
    global _THREAD_LIMITS
    _THREAD_LIMITS = threadpool_limits(n_cores)   # BLAS/OpenMP already loaded by numpy
    import torch
    torch.set_num_threads(n_cores)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass   # already set in this process
    torch.manual_seed(RANDOM_STATE)


CLI_TO_RECORDED = {"cv": ["cv", "shap"], "final": ["fairness"], "attacks": ["attacks"]}


# Run the selected experiment stages and record their outputs
def run_experiment(experiment_id: int, n_cores: int, stages, out_dir: str,
                   attack_workers: int, n_search_cases, attempt):
    frac, model_name, sampler_name = get_experiment_config(experiment_id)
    configuration_number = experiment_id + 1

    print(f"[exp {experiment_id:04d}] run {attempt.run_id}: frac={frac}, model={model_name}, sampler={sampler_name}")
    attempt.update(n_cores=n_cores, attack_workers_requested=attack_workers,
                   runtime=runtime_report(n_cores, label=f"exp {experiment_id:04d} ({model_name})"))

    # Set thread limits before importing numerical libraries
    from src.models_def import get_models, apply_class_weight
    from src.samplers_def import get_samplers, FALLBACK_EVENTS
    from src.pipeline_utils import (
        build_pipeline, run_cv, compute_cv_metrics, compute_shap,
        fit_final_model, default_probabilities,
    )
    from src.fairness import evaluate_group_fairness
    from src.attacks import run_attacks_for_configuration, SEARCH_ATTACKS
    from src.preprocessing import load_dataset, load_test_ids, split_train_test

    memory = FamilyMemoryMonitor().start()
    timer = StageTimer(n_cores, memory_monitor=memory)

    # Label stages that used the optional SMOTE replacement
    def labelled_keys(fallbacks_before):
        used = len(FALLBACK_EVENTS) > fallbacks_before
        label = KMEANS_SMOTE_FALLBACK_LABEL if used else sampler_name
        return {"Imbalance ratio": frac, "Model": model_name, "Sampler": label}, used

    with timer.stage("load"):
        X_full, y_full = load_dataset(DATA_PROCESSED_DIR, frac)
        test_ids = load_test_ids(DATA_PROCESSED_DIR)
        with open(os.path.join(DATA_PROCESSED_DIR, "feature_names.json")) as f:
            feat_names = json.load(f)
        with open(PARAMS_FILE) as f:
            best_params = json.load(f)
        X_tr, X_te, y_tr, y_te = split_train_test(X_full, y_full, test_ids)
        attempt.update(n_train_pool=len(X_tr), n_test_pool=len(X_te))

    is_nn = model_name == "NeuralNetClassifier"
    models = get_models(X_tr.shape[1], n_cores=n_cores, best_params=best_params)
    base_model = models[model_name]

    # Run cross-validation and collect SHAP importances
    if "cv" in stages:
        before = len(FALLBACK_EVENTS)
        with timer.stage("cv"):
            model = apply_class_weight(clone(base_model), sampler_name, y_tr)
            sampler = get_samplers(n_jobs=n_cores)[sampler_name]
            if is_nn:
                X_cv, y_cv = X_tr.astype(np.float32), y_tr.to_numpy().astype(np.int64)
            else:
                X_cv, y_cv = X_tr, y_tr
            pipe = build_pipeline(model, sampler)
            # Run folds sequentially with model threads using the assigned cores
            cv_out = run_cv(pipe, X_cv, y_cv, n_jobs=1)
            keys, fell_back = labelled_keys(before)
            metrics = compute_cv_metrics(cv_out, frac, model_name, keys["Sampler"], X_cv, y_cv)
            pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
            print(f"  CV: PR-AUC={metrics['PR-AUC (mean)']:.4f}  F1={metrics['F1 (mean)']:.4f}")
        attempt.stage_done("cv", kmeans_smote_fallback=fell_back)

        with timer.stage("shap"):
            shap_row = compute_shap(
                cv_out["estimator"][-1], cv_out["indices"]["train"][-1], cv_out["indices"]["test"][-1],
                model_name, keys["Sampler"], frac, X_cv, feat_names,
            )
            if shap_row:
                pd.DataFrame([shap_row]).to_csv(os.path.join(out_dir, "shap.csv"), index=False)
        if shap_row:
            attempt.stage_done("shap", kmeans_smote_fallback=fell_back)
        del cv_out

    # Fit the final model and evaluate group fairness
    final = None
    if "final" in stages or "attacks" in stages:
        before = len(FALLBACK_EVENTS)
        with timer.stage("final_fit"):
            sampler = get_samplers(n_jobs=n_cores)[sampler_name]
            final = fit_final_model(base_model, sampler, sampler_name, X_tr, y_tr)
            print(f"  Final model threshold: {final['threshold']:.4f}")
        final_keys, final_fell_back = labelled_keys(before)

    if "final" in stages:
        with timer.stage("fairness"):
            y_te_arr = y_te.to_numpy().astype(int)
            p_test = default_probabilities(final["pipeline"], X_te)
            y_pred = (p_test >= final["threshold"]).astype(int)
            fair = evaluate_group_fairness(y_te_arr, y_pred, X_te, final["X_fit"])
            for position, (k, v) in enumerate(final_keys.items()):
                fair.insert(position, k, v)
            fair.to_csv(os.path.join(out_dir, "fairness.csv"), index=False)
            pd.DataFrame([{
                **final_keys,
                "Threshold": final["threshold"],
                "N test": len(y_te_arr),
                "N test defaults": int(y_te_arr.sum()),
                "Test ROC-AUC": roc_auc_score(y_te_arr, p_test),
                "Test PR-AUC": average_precision_score(y_te_arr, p_test),
                "Test recall (default)": recall_score(y_te_arr, y_pred, zero_division=0),
                "Test precision (default)": precision_score(y_te_arr, y_pred, zero_division=0),
                "Favourable rate (all)": float((y_pred == FAVOURABLE_LABEL).mean()),
            }]).to_csv(os.path.join(out_dir, "final_metrics.csv"), index=False)
        attempt.stage_done("fairness", kmeans_smote_fallback=final_fell_back)

    # Attack eligible defaults using the final model
    if "attacks" in stages:
        with timer.stage("attacks"):
            search_attacks = {
                name: fn for name, fn in SEARCH_ATTACKS.items()
                if not (name == "ZOO" and model_name in ZOO_SKIP_MODELS)
            }
            kwargs = {} if n_search_cases is None else {"n_search_cases": n_search_cases}
            records, summary, info = run_attacks_for_configuration(
                pipeline=final["pipeline"], threshold=final["threshold"],
                X_reference=final["X_fit"], X_test=X_te, y_test=y_te,
                configuration_number=configuration_number,
                search_attacks=search_attacks,
                n_workers=attack_workers,
                start_method=ATTACK_START_METHOD.get(model_name, "spawn"),
                **kwargs,
            )
            for position, (k, v) in enumerate(final_keys.items()):
                records.insert(position, k, v)
                summary.insert(position, k, v)
            summary.insert(3, "Configuration", configuration_number)
            if "Test row" in records.columns:
                records["Test row"] = records["Test row"].astype("int64")
            summary.to_csv(os.path.join(out_dir, "attack_summary.csv"), index=False)
            records.to_parquet(os.path.join(out_dir, "attack_records.parquet"), index=False)
            attack_info = {k: v for k, v in info.items() if k != "search_pool_error"}
            attempt.update(attack_info=attack_info)
            if "search_pool_error" in info:
                attempt.update(attack_pool_error=info["search_pool_error"])
            grid = summary[summary["Method"] == "Grid"].iloc[0]
            print(f"  Attacks: eligible={grid['N eligible']}, clean recall={grid['Clean recall']:.3f}, "
                  f"grid ASR={grid['Attack success rate']:.3f}, post-attack recall={grid['Post-attack recall']:.3f}")
            print(f"  Attack timings: {attack_info}")
        attempt.stage_done("attacks", kmeans_smote_fallback=final_fell_back)

    memory.stop()
    attempt.record_timings(timer.summary())
    attempt.update(kmeans_smote_fallbacks=list(FALLBACK_EVENTS))
    attempt.finish("ok")
    print(f"[exp {experiment_id:04d}] DONE -> {out_dir}")


# Get the assigned core count from HyperQueue or CPU affinity
def default_cores():
    if os.environ.get("HQ_CPUS"):
        return len(os.environ["HQ_CPUS"].split(","))
    return len(os.sched_getaffinity(0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_id", type=int, required=True,
                        help=f"0 to {TOTAL_EXPERIMENTS - 1}")
    parser.add_argument("--n_cores", type=int, default=None,
                        help="Cores for this task (default: the cores HQ assigned, HQ_CPUS)")
    parser.add_argument("--attack_workers", type=int, default=None,
                        help="Processes for GA/ZOO (default: n_cores; 1 = serial)")
    parser.add_argument("--stages", nargs="+", default=ALL_STAGES, choices=ALL_STAGES,
                        help="Run only some stages, e.g. --stages final attacks after a CV-only run")
    parser.add_argument("--out_dir", default=None,
                        help="Default: results_v2/exp_XXXX (the scaling and smoke tests write elsewhere)")
    parser.add_argument("--n_search_cases", type=int, default=None,
                        help="Override N_ATTACK_CASES for GA/ZOO (e.g. a quick test)")
    args = parser.parse_args()

    if args.experiment_id < 0 or args.experiment_id >= TOTAL_EXPERIMENTS:
        sys.exit(f"experiment_id must be 0-{TOTAL_EXPERIMENTS - 1}, got {args.experiment_id}")

    n_cores = args.n_cores or default_cores()
    attack_workers = args.attack_workers or n_cores
    configure_threads(n_cores)

    out_dir = args.out_dir or os.path.join(RESULTS_DIR, f"exp_{args.experiment_id:04d}")
    frac, model_name, sampler_name = get_experiment_config(args.experiment_id)
    recorded = [r for stage in args.stages for r in CLI_TO_RECORDED[stage]]
    attempt = Attempt(out_dir,
                      identity={"experiment_id": args.experiment_id, "imbalance_ratio": frac,
                                "model": model_name, "sampler": sampler_name},
                      stages=recorded)

    try:
        run_experiment(args.experiment_id, n_cores, args.stages, out_dir,
                       attack_workers, args.n_search_cases, attempt)
    except BaseException:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        attempt.finish("failed", error=tb)
        sys.exit(1)
