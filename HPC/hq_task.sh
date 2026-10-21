#!/bin/bash
# Run one HyperQueue experiment or pricing sensitivity task
# Usage: hq submit ... hq_task.sh [sensitivity.py]
# Submit from HPC/ after starting the HyperQueue server

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1

module load 2023 || exit 1
module load Python/3.11.3-GCCcore-12.3.0 || exit 1
source ~/.local/venv/bin/activate || exit 1

SCRIPT=${1:-main.py}

# Count the assigned HyperQueue cores, using 16 when HQ_CPUS is unset
if [ -n "$HQ_CPUS" ]; then
    N_CORES=$(echo "$HQ_CPUS" | tr ',' '\n' | wc -l)
else
    N_CORES=16
fi

# Limit numerical libraries to the assigned cores
export OMP_NUM_THREADS=$N_CORES
export MKL_NUM_THREADS=$N_CORES
export OPENBLAS_NUM_THREADS=$N_CORES
export NUMEXPR_NUM_THREADS=$N_CORES
export PYTHONUNBUFFERED=1

# Use the staged node-local data when available
if [ -d "/scratch-node/$USER/$SLURM_JOB_ID/data/processed" ]; then
    export LOCAL_DATA_DIR="/scratch-node/$USER/$SLURM_JOB_ID/data/processed"
else
    export LOCAL_DATA_DIR="data/processed"
fi

echo "[$(date)] Task $HQ_TASK_ID ($SCRIPT) on $(hostname)"
echo "  HQ_CPUS=$HQ_CPUS  HQ_PIN=$HQ_PIN  N_CORES=$N_CORES  nproc=$(nproc)"
echo "  affinity: $(taskset -cp $$ 2>/dev/null)"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS  LOCAL_DATA_DIR=$LOCAL_DATA_DIR"

if [ "$SCRIPT" = "sensitivity.py" ]; then
    python3.11 sensitivity.py --task_id="$HQ_TASK_ID" --n_cores="$N_CORES"
else
    # Pass optional stage selection to main.py
    python3.11 main.py --experiment_id="$HQ_TASK_ID" --n_cores="$N_CORES" ${STAGES:+--stages $STAGES}
fi

EXIT_CODE=$?
echo "[$(date)] Task $HQ_TASK_ID finished with exit code $EXIT_CODE"
exit $EXIT_CODE
