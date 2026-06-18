#!/bin/bash
# Single HyperQueue task — runs one experiment.
# HQ_TASK_ID is set automatically by HyperQueue (0-99).
# --cpus=8 in hq submit means 8 cores are pinned to this task via taskset.

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.local/venv/bin/activate

N_CORES=6

echo "[$(date)] Starting experiment $HQ_TASK_ID on $(hostname) with $N_CORES cores"

python3.11 main.py --experiment_id=$HQ_TASK_ID --n_cores=$N_CORES

EXIT_CODE=$?
echo "[$(date)] Experiment $HQ_TASK_ID finished with exit code $EXIT_CODE"
exit $EXIT_CODE
