"""
Phase C: Collect all per-experiment results into final CSVs and plots.
Run after all HQ tasks have completed.

Usage:
    python aggregate.py [--results_dir results] [--outputs_dir outputs]
"""
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

from src.config import RESULTS_DIR, OUTPUTS_DIR, MODEL_NAMES, SAMPLER_NAMES, MINORITY_FRACS


def collect_csv(results_dir: str, filename: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(results_dir, "exp_*", filename)))
    if not paths:
        print(f"  No {filename} files found.")
        return pd.DataFrame()
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"  Warning: could not read {p}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def report_failures(results_dir: str):
    error_files = sorted(glob.glob(os.path.join(results_dir, "exp_*", "error.txt")))
    if error_files:
        print(f"\n  {len(error_files)} failed experiments:")
        for ef in error_files:
            exp_id = os.path.basename(os.path.dirname(ef))
            meta_path = os.path.join(os.path.dirname(ef), "meta.json")
            info = ""
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    m = json.load(f)
                info = f"  {m.get('model','?')} + {m.get('sampler','?')} @ IR={m.get('imbalance_ratio','?')}"
            print(f"    {exp_id}{info}")
    else:
        print("  No failed experiments.")


def plot_pr_auc_vs_imbalance(model_results: pd.DataFrame, outputs_dir: str):
    ir_col, y_col = "Imbalance ratio", "PR-AUC (mean)"
    samplers = model_results["Sampler"].unique()
    models = model_results["Model"].unique()
    n = len(samplers)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, sampler in zip(axes, samplers):
        sub = model_results[model_results["Sampler"] == sampler]
        for model in models:
            m = sub[sub["Model"] == model].sort_values(ir_col)
            if not m.empty:
                ax.plot(m[ir_col], m[y_col], marker="o", label=model)
        ax.set_title(sampler)
        ax.set_xlabel("Imbalance Ratio")
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(outputs_dir, "plots", "pr_auc_vs_imbalance.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_attack_success(adf: pd.DataFrame, outputs_dir: str):
    ir_col = "Imbalance ratio"
    # Each attack method has its own "Attack success rate (<method>)" column.
    method_styles = [("GA", "-"), ("ZOO", "--")]
    methods = [
        (name, ls) for name, ls in method_styles
        if f"Attack success rate ({name})" in adf.columns
    ]
    if not methods:  # backward-compat with single-method results
        methods = [("", "-")]

    samplers = adf["Sampler"].unique()
    models = list(adf["Model"].unique())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_color = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(models)}

    n = len(samplers)
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, sampler in zip(axes, samplers):
        sub = adf[adf["Sampler"] == sampler]
        for model in models:
            m = sub[sub["Model"] == model].sort_values(ir_col)
            if m.empty:
                continue
            for name, ls in methods:
                col = f"Attack success rate ({name})" if name else "Attack success rate"
                if col in m:
                    ax.plot(m[ir_col], m[col], marker="o", linestyle=ls, color=model_color[model])
        ax.set_title(sampler)
        ax.set_xlabel("Imbalance ratio")
        ax.set_ylabel("Attack success rate")
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.set_visible(False)

    # Two legends: colour -> model, line style -> attack method.
    from matplotlib.lines import Line2D
    model_handles = [Line2D([0], [0], color=model_color[m], marker="o", label=m) for m in models]
    method_handles = [
        Line2D([0], [0], color="black", linestyle=ls, label=name)
        for name, ls in methods if name
    ]
    handles = model_handles + method_handles
    fig.legend(handles, [h.get_label() for h in handles],
               loc="upper center", ncol=len(handles), bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(os.path.join(outputs_dir, "plots", "attack_success_rate.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_shap(shap_wide: pd.DataFrame, outputs_dir: str, mutable_features: list, top_n: int = 15):
    feat_cols = [c for c in shap_wide.columns if c not in ("Model", "Sampler", "Imbalance ratio")]
    models_in_data = shap_wide["Model"].unique()

    fig, axes = plt.subplots(1, len(models_in_data), figsize=(6 * len(models_in_data), 6))
    axes = np.atleast_1d(axes)
    for ax, model_name in zip(axes, models_in_data):
        imp = shap_wide[shap_wide["Model"] == model_name][feat_cols].mean().nlargest(top_n)
        colors = ["tab:orange" if f in mutable_features else "tab:blue" for f in imp.index]
        imp[::-1].plot(kind="barh", ax=ax, color=colors[::-1])
        ax.set_title(model_name)
        ax.set_xlabel("Mean |SHAP value|")
    fig.legend(
        handles=[
            Patch(facecolor="tab:orange", label="Adversarially mutable"),
            Patch(facecolor="tab:blue",   label="Not mutable"),
        ],
        loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04),
    )
    fig.suptitle(f"Top-{top_n} Global SHAP Features per Model", y=1.06)
    plt.tight_layout()
    fig.savefig(os.path.join(outputs_dir, "plots", "shap_global.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def find_best_model(model_results: pd.DataFrame) -> str:
    avg = model_results.groupby("Model")["PR-AUC (mean)"].mean().sort_values(ascending=False)
    print("\n  Average PR-AUC by model (across all imbalances & samplers):")
    print(avg.to_string())
    return avg.index[0]


def main(results_dir: str, outputs_dir: str):
    os.makedirs(os.path.join(outputs_dir, "plots"), exist_ok=True)

    print("\n=== Collecting results ===")
    report_failures(results_dir)

    print("\nCollecting metrics ...")
    model_results = collect_csv(results_dir, "metrics.csv")
    if not model_results.empty:
        model_results.to_csv(os.path.join(outputs_dir, "model_results.csv"), index=False)
        print(f"  {len(model_results)} rows → model_results.csv")
        plot_pr_auc_vs_imbalance(model_results, outputs_dir)
        best_model = find_best_model(model_results)
        print(f"\n  Best model by average PR-AUC: {best_model}")

    print("\nCollecting SHAP ...")
    shap_wide = collect_csv(results_dir, "shap.csv")
    if not shap_wide.empty:
        shap_wide.to_csv(os.path.join(outputs_dir, "shap_importances.csv"), index=False)
        print(f"  {len(shap_wide)} rows → shap_importances.csv")
        try:
            from src.config import MUTABLE_FEATURES
            plot_shap(shap_wide, outputs_dir, MUTABLE_FEATURES)
        except Exception as e:
            print(f"  SHAP plot failed: {e}")

    print("\nCollecting DIR ...")
    dir_df = collect_csv(results_dir, "dir.csv")
    if not dir_df.empty:
        dir_df.to_csv(os.path.join(outputs_dir, "dir_fairness_results.csv"), index=False)
        print(f"  {len(dir_df)} rows → dir_fairness_results.csv")

    print("\nCollecting adversarial attacks ...")
    attack_df = collect_csv(results_dir, "attacks.csv")
    if not attack_df.empty:
        attack_df.to_csv(os.path.join(outputs_dir, "adversarial_attack_results.csv"), index=False)
        print(f"  {len(attack_df)} rows → adversarial_attack_results.csv")
        plot_attack_success(attack_df, outputs_dir)

    print(f"\nAll outputs saved to {outputs_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--outputs_dir", default=OUTPUTS_DIR)
    args = parser.parse_args()
    main(args.results_dir, args.outputs_dir)
