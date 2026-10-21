# Compare serial and parallel attack results.
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# Allow small score differences between prediction thread counts
ATOL = 1e-8
RTOL = 1e-8
# Exclude timing and process IDs from agreement checks
IGNORE = {"Runtime seconds", "Job seconds", "Worker pid"}


# Load attack records from a completed run
def load_run(path):
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        return None, None, "no meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("status") != "ok":
        last = (meta.get("attempts") or [{}])[-1]
        return meta, None, f"status {meta.get('status')}: {last.get('error', '')}"
    if "attacks" not in meta.get("stage_records", {}):
        return meta, None, "attacks stage has no record"
    records = pd.read_parquet(os.path.join(path, "attack_records.parquet"))
    return meta, records, None


# Compare borrower-level records within the score tolerance
def compare_records(a, b):
    key = [c for c in ["Method", "Borrower id", "Test row"] if c in a.columns]
    diffs = []
    if len(a) != len(b):
        diffs.append(f"row count {len(a)} vs {len(b)}")
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    if not a[key].equals(b[key]):
        diffs.append("different (method, borrower) pairs")
        return diffs
    for col in a.columns:
        if col in IGNORE or col not in b.columns:
            continue
        x, y = a[col], b[col]
        if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
            close = np.isclose(x.to_numpy(), y.to_numpy(), atol=ATOL, rtol=RTOL, equal_nan=True)
            if not close.all():
                worst = np.nanmax(np.abs(x.to_numpy() - y.to_numpy()))
                diffs.append(f"{col}: {int((~close).sum())} rows differ (max abs diff {worst:.3g})")
        else:
            same = (x.astype(str) == y.astype(str))
            if not same.all():
                diffs.append(f"{col}: {int((~same).sum())} rows differ")
    missing = sorted(set(a.columns) ^ set(b.columns))
    if missing:
        diffs.append(f"columns only in one run: {missing}")
    return diffs


# Read peak task memory for each recorded stage
def memory_by_stage(meta):
    stages = meta.get("stages", {})
    return {s: v.get("peak_family_memory_gib") for s, v in stages.items()}


# Compare completion, worker settings and results for a pair of runs
def check_pair(parallel_dir, serial_dir):
    row = {"parallel run": parallel_dir, "serial run": serial_dir}
    problems = []
    meta_p, rec_p, err_p = load_run(parallel_dir)
    meta_s, rec_s, err_s = load_run(serial_dir)
    if err_p:
        problems.append(f"parallel run: {err_p}")
    if err_s:
        problems.append(f"serial run: {err_s}")
    if meta_p:
        info = meta_p.get("attack_info", {})
        model = meta_p.get("identity", {}).get("model")
        row.update(model=model, workers=info.get("search_workers"),
                   start_method=info.get("search_start_method"),
                   search_seconds_parallel=info.get("search_seconds"),
                   peak_mem_parallel_gib=meta_p.get("peak_family_memory_gib"),
                   attack_mem_parallel_gib=memory_by_stage(meta_p).get("attacks"))
        if "attack_pool_error" in meta_p:
            problems.append("process pool failed and fell back to serial")
        from src.config import ATTACK_START_METHOD
        expected = ATTACK_START_METHOD.get(model, "spawn")
        if info.get("search_workers", 1) > 1 and info.get("search_start_method") != expected:
            problems.append(f"start method {info.get('search_start_method')}, expected {expected}")
        if info.get("search_workers", 1) <= 1:
            problems.append("parallel run used one worker")
    if meta_s:
        info = meta_s.get("attack_info", {})
        row.update(search_seconds_serial=info.get("search_seconds"),
                   peak_mem_serial_gib=meta_s.get("peak_family_memory_gib"))
    if rec_p is not None and rec_s is not None:
        diffs = compare_records(rec_p, rec_s)
        row["records compared"] = len(rec_p)
        problems.extend(diffs)
    row["ok"] = not problems
    row["problems"] = "; ".join(problems)
    return row


# Compare the requested runs and report any differences
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_dir", default="results/smoke")
    parser.add_argument("--a", help="Run directory (parallel)")
    parser.add_argument("--b", help="Run directory (serial)")
    args = parser.parse_args()

    if args.a and args.b:
        rows = [check_pair(args.a, args.b)]
    else:
        rows = []
        for serial in sorted(glob.glob(os.path.join(args.smoke_dir, "exp_*_workers_1"))):
            exp = re.search(r"exp_(\d+)_workers_1$", serial).group(1)
            parallel = [p for p in glob.glob(os.path.join(args.smoke_dir, f"exp_{exp}_workers_*"))
                        if not p.endswith("_workers_1")]
            if not parallel:
                rows.append({"serial run": serial, "ok": False, "problems": "no parallel run"})
            for p in parallel:
                rows.append(check_pair(p, serial))
    if not rows:
        sys.exit(f"No smoke runs found in {args.smoke_dir}")

    df = pd.DataFrame(rows)
    if "search_seconds_parallel" in df and "search_seconds_serial" in df:
        df["search speed-up"] = (df["search_seconds_serial"] / df["search_seconds_parallel"]).round(2)
    cols = [c for c in ["model", "workers", "start_method", "records compared", "search_seconds_serial",
                        "search_seconds_parallel", "search speed-up", "peak_mem_serial_gib",
                        "peak_mem_parallel_gib", "attack_mem_parallel_gib", "ok", "problems"] if c in df]
    with pd.option_context("display.width", 200, "display.max_colwidth", 120):
        print(df[cols].to_string(index=False))
    out = os.path.join(args.smoke_dir if not args.a else ".", "smoke_comparison.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")
    if not df["ok"].all():
        print("\nSMOKE TEST FAILED", file=sys.stderr)
        sys.exit(1)
    print("\nSmoke test passed: parallel and serial attacks agree.")


if __name__ == "__main__":
    main()
