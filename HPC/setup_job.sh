#!/bin/bash
# Prepare data and tune hyperparameters on an exclusive genoa node
# Usage: mkdir -p logs && sbatch setup_job.sh
# Use --skip_gridsearch for preprocessing only
# Use --skip_preprocess for grid search only

#SBATCH --job-name=AAI_Setup
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p logs data/processed config

module load 2023 || exit 1
module load Python/3.11.3-GCCcore-12.3.0 || exit 1
source ~/.local/venv/bin/activate || exit 1

# Use one library thread per parallel grid-search fit
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "Starting setup on $(hostname)"
python3.11 setup.py --data Data/accepted.csv --n_jobs 32 "$@"
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "Setup FAILED with exit code $STATUS" >&2
    exit $STATUS
fi
echo "Setup complete."
