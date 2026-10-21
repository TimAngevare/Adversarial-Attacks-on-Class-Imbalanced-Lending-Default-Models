# Collect completed task outputs and check the combined results.
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.config import (
    RESULTS_DIR, OUTPUTS_DIR, DATA_PROCESSED_DIR, MUTABLE_FEATURES, SEPARABILITY_TOL,
    KMEANS_SMOTE_FALLBACK_LABEL,
)
from src.attempts import STAGE_FILES, PIPELINE_VERSION
from main import TOTAL_EXPERIMENTS, get_experiment_config
from sensitivity import _GRID as PRICING_GRID

CONFIG_KEYS = ["Imbalance ratio", "Model", "Sampler"]


# Read the task metadata when available
def read_meta(directory):
    path = os.path.join(directory, "meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# Get the expected configuration for the task ID
def expected_identity(kind, task_id):
    if kind == "exp":
        frac, model, sampler = get_experiment_config(task_id)
        return {"experiment_id": task_id, "imbalance_ratio": frac, "model": model,
                "sampler": sampler, "pipeline_version": PIPELINE_VERSION}
    frac, variant, model, sampler = PRICING_GRID[task_id]
    return {"task_id": task_id, "imbalance_ratio": frac, "variant": variant, "model": model,
            "sampler": sampler, "pipeline_version": PIPELINE_VERSION}


# Check that each row matches the task configuration
def _check_rows(df, identity, path):
    allowed_samplers = {identity["sampler"], KMEANS_SMOTE_FALLBACK_LABEL}
    bad = []
    if "Model" in df and (df["Model"] != identity["model"]).any():
        bad.append("Model")
    if "Sampler" in df and (~df["Sampler"].isin(allowed_samplers)).any():
        bad.append("Sampler")
    if "Imbalance ratio" in df and not np.allclose(df["Imbalance ratio"].astype(float), identity["imbalance_ratio"]):
        bad.append("Imbalance ratio")
    if bad:
        raise ValueError(f"{path}: rows do not match the task configuration ({bad})")


# Collect outputs from tasks with valid identities and completed stages
def collect_stage(results_dir, kind, n_tasks, stage, filename, problems, provenance):
    frames = []
    reader = pd.read_parquet if filename.endswith(".parquet") else pd.read_csv
    for task_id in range(n_tasks):
        directory = os.path.join(results_dir, f"{kind}_{task_id:04d}")
        meta = read_meta(directory)
        if meta is None:
            continue
        identity = expected_identity(kind, task_id)
        if meta.get("identity") != identity:
            problems.append(f"{directory}: identity does not match this pipeline version; ignored")
            continue
        record = meta.get("stage_records", {}).get(stage)
        path = os.path.join(directory, filename)
        if record is None:
            continue
        if not os.path.exists(path):
            problems.append(f"{path}: stage {stage} recorded but file missing")
            continue
        df = reader(path)
        _check_rows(df, identity, path)
        provenance.append({"task": f"{kind}_{task_id:04d}", "stage": stage, "file": filename,
                           "run_id": record["run_id"], "finished": record.get("finished"),
                           "kmeans_smote_fallback": record.get("kmeans_smote_fallback", False)})
        frames.append(df)
    if not frames:
        print(f"  No completed {filename} files found.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# Summarise task status, stage times and resource use
def task_report(results_dir):
    rows = []
    for exp_id in range(TOTAL_EXPERIMENTS):
        frac, model, sampler = get_experiment_config(exp_id)
        d = os.path.join(results_dir, f"exp_{exp_id:04d}")
        row = {"experiment_id": exp_id, "Imbalance ratio": frac, "Model": model, "Sampler": sampler}
        m = read_meta(d)
        if m is None:
            row["status"] = "missing"
        elif m.get("identity") != expected_identity("exp", exp_id):
            row["status"] = "stale (other configuration or pipeline version)"
        else:
            status = m.get("status", "?")
            # Treat tasks without a recorded status as unfinished
            row["status"] = "running or killed" if status == "running" else status
            last = (m.get("attempts") or [{}])[-1]
            row["attempts"] = len(m.get("attempts", []))
            row["run id"] = last.get("run_id")
            row["error"] = last.get("error", "") if status == "failed" else ""
            row["completed stages"] = ",".join(sorted(m.get("stage_records", {})))
            row["host"] = m.get("runtime", {}).get("hostname")
            row["n_cores"] = m.get("n_cores")
            row["nproc"] = m.get("runtime", {}).get("nproc")
            row["torch_threads"] = m.get("runtime", {}).get("torch_num_threads")
            row["OMP_NUM_THREADS"] = m.get("runtime", {}).get("env", {}).get("OMP_NUM_THREADS")
            for stage, st in m.get("stages", {}).items():
                row[f"{stage} wall (s)"] = st.get("wall_seconds")
                row[f"{stage} CPU eff."] = st.get("cpu_efficiency")
                row[f"{stage} peak mem (GiB)"] = st.get("peak_family_memory_gib")
            row["total wall (h)"] = round(sum(st.get("wall_seconds", 0) for st in m.get("stages", {}).values()) / 3600, 2)
            row["peak memory task + workers (GiB)"] = m.get("peak_family_memory_gib")
            info = m.get("attack_info", {})
            row["attack workers"] = info.get("search_workers")
            row["attack start method"] = info.get("search_start_method")
            row["KMeansSMOTE fallbacks"] = len(m.get("kmeans_smote_fallbacks", []))
        rows.append(row)
    return pd.DataFrame(rows)


# Check SHAP columns against the prepared feature set
def check_shap_features(shap_wide):
    path = os.path.join(DATA_PROCESSED_DIR, "feature_names.json")
    if shap_wide.empty or not os.path.exists(path):
        return
    with open(path) as f:
        features = set(json.load(f))
    found = set(shap_wide.columns) - set(CONFIG_KEYS)
    if found != features:
        raise ValueError(f"SHAP features differ from feature_names.json: "
                         f"extra {sorted(found - features)}, missing {sorted(features - found)}")
    print("  SHAP features match feature_names.json.")


# Check LR attack scores against the coefficient-based reference
def check_attacks(records):
    if records.empty or "Exact (LR)" not in set(records["Method"]):
        print("  No exact-reference records to check.")
        return pd.DataFrame()
    row_keys = CONFIG_KEYS + ["Test row"]
    to_logit = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12)) - np.log1p(-np.clip(p, 1e-12, 1 - 1e-12))
    ref = (records[records["Method"] == "Exact (LR)"][row_keys + ["Success", "Lowest probability found"]]
           .rename(columns={"Success": "ref_success", "Lowest probability found": "ref_p"}))
    rows = []
    for method in ["Grid", "GA", "ZOO"]:
        m = records[records["Method"] == method].merge(ref, on=row_keys, how="inner")
        if m.empty:
            continue
        lower = to_logit(m["Lowest probability found"]) < to_logit(m["ref_p"]) - SEPARABILITY_TOL
        disagree = (m["Success"] == 1) & (m["ref_success"] == 0)
        rows.append({"Method": method, "Borrowers compared": len(m),
                     "Lower score than exact": int(lower.sum()),
                     "Success where exact fails": int((disagree & lower).sum()),
                     "Ties at the threshold": int((disagree & ~lower).sum())})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if len(out) and out["Lower score than exact"].sum() > 0:
        print("  WARNING: an attack beat the exact reference for Logistic Regression. Check the attack code.")
    else:
        print("  No attack beats the exact reference for Logistic Regression.")
    return out


# Plot each metric by sampler, model and prevalence
def plot_by_sampler(df, metric, ylabel, fname, methods=None):
    samplers = list(df["Sampler"].unique())
    models = list(df["Model"].unique())
    colors = {m: plt.cm.tab10(i % 10) for i, m in enumerate(models)}
    styles = {"Grid": "-", "GA": "--", "ZOO": ":", "Exact (LR)": "-."}
    ncols = 3
    nrows = -(-len(samplers) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, sampler in zip(axes, samplers):
        sub = df[df["Sampler"] == sampler]
        for model in models:
            for method in (methods or [None]):
                s = sub[sub["Model"] == model]
                if method is not None:
                    s = s[s["Method"] == method]
                s = s.sort_values("Imbalance ratio")
                if not s.empty:
                    ax.plot(s["Imbalance ratio"], s[metric], marker="o", color=colors[model],
                            linestyle=styles.get(method, "-"), label=f"{model} {method or ''}".strip())
        ax.set_xscale("log")
        ax.set_title(sampler)
        ax.set_xlabel("Default prevalence")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    for ax in axes[len(samplers):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight", dpi=130)
    plt.close(fig)


# Plot mean SHAP importance for each model
def plot_shap(shap_wide, fname, top_n=15):
    feat_cols = [c for c in shap_wide.columns if c not in CONFIG_KEYS]
    models = shap_wide["Model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        imp = shap_wide[shap_wide["Model"] == model][feat_cols].mean().nlargest(top_n)
        colors = ["tab:orange" if f in MUTABLE_FEATURES else "tab:blue" for f in imp.index]
        imp[::-1].plot(kind="barh", ax=ax, color=colors[::-1])
        ax.set_title(model)
        ax.set_xlabel("Mean |SHAP value|")
    fig.legend(handles=[Patch(facecolor="tab:orange", label="Adversarially mutable"),
                        Patch(facecolor="tab:blue", label="Not mutable")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight", dpi=130)
    plt.close(fig)


# Collect results, run consistency checks and save summary plots
def main(results_dir, outputs_dir):
    plots = os.path.join(outputs_dir, "plots")
    os.makedirs(plots, exist_ok=True)

    print("\n=== Task report ===")
    report = task_report(results_dir)
    report.to_csv(os.path.join(outputs_dir, "task_report.csv"), index=False)
    print(report["status"].value_counts().to_string())
    bad = report[report["status"] != "ok"]
    if len(bad):
        print("\nNot finished:")
        print(bad[[c for c in ["experiment_id", "Model", "Sampler", "Imbalance ratio", "status", "error"]
                   if c in bad.columns]].to_string(index=False))
    eff_cols = [c for c in report.columns if c.endswith("CPU eff.")]
    if eff_cols:
        print("\nMean CPU efficiency per stage and model (CPU seconds / (wall x cores)):")
        print(report.groupby("Model")[eff_cols].mean().round(2).to_string())
        print("\nSlowest tasks (hours):")
        print(report.sort_values("total wall (h)", ascending=False)
              [["experiment_id", "Model", "Sampler", "Imbalance ratio", "total wall (h)",
                "peak memory task + workers (GiB)"]]
              .head(10).to_string(index=False))
    if "KMeansSMOTE fallbacks" in report and report["KMeansSMOTE fallbacks"].fillna(0).sum() > 0:
        print("\nKMeansSMOTE fell back to SMOTE in:")
        print(report[report["KMeansSMOTE fallbacks"] > 0][["experiment_id", "Model", "Imbalance ratio"]].to_string(index=False))

    problems, provenance = [], []
    files = [
        ("exp", TOTAL_EXPERIMENTS, "cv", "metrics.csv", "model_results.csv"),
        ("exp", TOTAL_EXPERIMENTS, "shap", "shap.csv", "shap_importances.csv"),
        ("exp", TOTAL_EXPERIMENTS, "fairness", "fairness.csv", "fairness_results.csv"),
        ("exp", TOTAL_EXPERIMENTS, "fairness", "final_metrics.csv", "final_model_test_metrics.csv"),
        ("exp", TOTAL_EXPERIMENTS, "attacks", "attack_summary.csv", "attack_summary.csv"),
        ("exp", TOTAL_EXPERIMENTS, "attacks", "attack_records.parquet", "attack_records.parquet"),
        ("pricing", len(PRICING_GRID), "pricing", "pricing.csv", "pricing_sensitivity.csv"),
    ]
    tables = {}
    for kind, n_tasks, stage, src_name, dst_name in files:
        print(f"\nCollecting {src_name} ...")
        df = collect_stage(results_dir, kind, n_tasks, stage, src_name, problems, provenance)
        tables[dst_name] = df
        if not df.empty:
            if dst_name.endswith(".parquet"):
                df.to_parquet(os.path.join(outputs_dir, dst_name), index=False)
            else:
                df.to_csv(os.path.join(outputs_dir, dst_name), index=False)
            print(f"  {len(df):,} rows -> {dst_name}")
    records = tables["attack_records.parquet"]
    # Save the producing task, stage and run for each collected file
    pd.DataFrame(provenance).to_csv(os.path.join(outputs_dir, "provenance.csv"), index=False)
    if problems:
        print("\nIgnored or inconsistent files:")
        for p in dict.fromkeys(problems):   # unique, in order
            print("  " + p)

    fallback_rows = [name for name, df in tables.items()
                     if not df.empty and "Sampler" in df and (df["Sampler"] == KMEANS_SMOTE_FALLBACK_LABEL).any()]
    if fallback_rows:
        print(f"\nWARNING: rows labelled '{KMEANS_SMOTE_FALLBACK_LABEL}' in {fallback_rows}. "
              "They are not KMeans-SMOTE results.")

    print("\n=== Checks ===")
    check_shap_features(tables["shap_importances.csv"])
    check_attacks(records)

    mr = tables["model_results.csv"]
    if not mr.empty:
        plot_by_sampler(mr, "PR-AUC (mean)", "PR-AUC (CV)", os.path.join(plots, "pr_auc_vs_imbalance.png"))
    if not tables["shap_importances.csv"].empty:
        plot_shap(tables["shap_importances.csv"], os.path.join(plots, "shap_global.png"))
    summ = tables["attack_summary.csv"]
    if not summ.empty:
        plot_by_sampler(summ, "Attack success rate", "ASR (eligible defaults)",
                        os.path.join(plots, "attack_success_rate.png"), methods=["Grid", "GA", "ZOO"])
        plot_by_sampler(summ, "Post-attack recall", "Recall after attack (all test defaults)",
                        os.path.join(plots, "attack_post_attack_recall.png"), methods=["Grid"])
    fair = tables["fairness_results.csv"]
    if not fair.empty:
        for feature in fair["Feature"].unique():
            plot_by_sampler(fair[fair["Feature"] == feature], "DIR", f"DIR ({feature})",
                            os.path.join(plots, f"dir_{feature}.png"))

    print(f"\nAll outputs saved to {outputs_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--outputs_dir", default=OUTPUTS_DIR)
    args = parser.parse_args()
    main(args.results_dir, args.outputs_dir)
