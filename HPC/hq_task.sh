#!/bin/bash
# Single HyperQueue task — runs one experiment.
# HQ_TASK_ID is set automatically by HyperQueue (0-99).

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.local/venv/bin/activate

# Increased the number of cores to 16 to speed up the experiments. This is possible because the HPC node has 192 cores and we are running 24 concurrent tasks (192/8 = 24). Each task can now use 16 cores without exceeding the total available cores.
N_CORES=16

# Check if the SLURM job successfully staged data to the NVMe drive
if [ -d "/scratch-node/$USER/$SLURM_JOB_ID/data/processed" ]; then
    export LOCAL_DATA_DIR="/scratch-node/$USER/$SLURM_JOB_ID/data/processed"
else
    export LOCAL_DATA_DIR="data/processed"
fi

echo "[$(date)] Starting experiment $HQ_TASK_ID on $(hostname) with $N_CORES cores"

python3.11 main.py --experiment_id=$HQ_TASK_ID --n_cores=$N_CORES

EXIT_CODE=$?
echo "[$(date)] Experiment $HQ_TASK_ID finished with exit code $EXIT_CODE"
exit $EXIT_CODE
