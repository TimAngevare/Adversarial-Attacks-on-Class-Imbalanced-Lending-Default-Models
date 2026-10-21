# Report stage times, speed-up and efficiency for completed scaling runs.
import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--exp_id", type=int, default=60)
parser.add_argument("--scaling_dir", default="results/scaling")
args = parser.parse_args()

STAGE_RECORD = {"cv": "cv", "shap": "shap", "fairness": "fairness", "attacks": "attacks"}

exit_codes = {}
status_path = os.path.join(args.scaling_dir, f"status_exp_{args.exp_id}.tsv")
if os.path.exists(status_path):
    status = pd.read_csv(status_path, sep="\t", dtype=str)
    exit_codes = dict(zip(status["cores"].astype(int), status["exit_code"]))

completed, problems = [], []
dirs = glob.glob(os.path.join(args.scaling_dir, f"exp_{args.exp_id}_cores_*"))
seen = set()
for d in dirs:
    cores = int(re.search(r"cores_(\d+)$", d).group(1))
    seen.add(cores)
    meta_path = os.path.join(d, "meta.json")
    code = exit_codes.get(cores)
    if code is not None and code != "0":
        problems.append({"cores": cores, "problem": f"exit code {code}"})
        continue
    if not os.path.exists(meta_path):
        problems.append({"cores": cores, "problem": "no meta.json (run did not start)"})
        continue
    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("status") != "ok":
        last = (meta.get("attempts") or [{}])[-1]
        problems.append({"cores": cores, "problem": f"status {meta.get('status')}: {last.get('error', '')}"})
        continue
    if meta.get("identity", {}).get("experiment_id") != args.exp_id:
        problems.append({"cores": cores, "problem": "meta.json belongs to another experiment"})
        continue
    stages = meta.get("stages", {})
    records = meta.get("stage_records", {})
    missing = [s for s, rec in STAGE_RECORD.items() if s in stages and rec not in records]
    if missing:
        problems.append({"cores": cores, "problem": f"stages without valid output: {missing}"})
        continue
    row = {"cores": cores, "nproc": meta.get("runtime", {}).get("nproc"),
           "torch threads": meta.get("runtime", {}).get("torch_num_threads"),
           "peak memory task + workers (GiB)": meta.get("peak_family_memory_gib")}
    for stage, s in stages.items():
        row[stage] = s["wall_seconds"]
    completed.append(row)

for cores, code in exit_codes.items():
    if cores not in seen:
        problems.append({"cores": cores, "problem": f"no output folder (exit code {code})"})

if problems:
    print(f"Experiment {args.exp_id}: runs left out of the table")
    print(pd.DataFrame(problems).sort_values("cores").to_string(index=False))
    print()

if not completed:
    sys.exit(f"No completed runs in {args.scaling_dir} for experiment {args.exp_id}")

df = pd.DataFrame(completed).sort_values("cores").set_index("cores")
stages = [c for c in ["load", "cv", "shap", "final_fit", "fairness", "attacks"] if c in df.columns]
df["total"] = df[stages].sum(axis=1)
stages.append("total")

print(f"Experiment {args.exp_id}: wall seconds per stage (completed runs only)")
print(df[["nproc", "torch threads", "peak memory task + workers (GiB)"] + stages].round(2).to_string())

if 1 not in df.index:
    print("\nNo completed 1-core run, so no speed-up or efficiency. Rerun with 1 core included.")
    sys.exit(1 if problems else 0)

speedup = df.loc[1, stages] / df[stages]
efficiency = speedup.div(df.index.to_series(), axis=0)
print("\nSpeed-up vs 1 core")
print(speedup.round(2).to_string())
print("\nParallel efficiency (speed-up / cores; 1.0 = perfect)")
print(efficiency.round(2).to_string())
out = os.path.join(args.scaling_dir, f"scaling_exp_{args.exp_id}.csv")
pd.concat({"seconds": df[stages], "speedup": speedup, "efficiency": efficiency}, axis=1).to_csv(out)
print(f"\nSaved {out}")
if problems:
    pd.DataFrame(problems).to_csv(os.path.join(args.scaling_dir, f"scaling_exp_{args.exp_id}_left_out.csv"), index=False)
