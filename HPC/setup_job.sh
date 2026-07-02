#!/bin/bash
# Optional: run setup.py as a SLURM job (uses all 192 cores for GridSearchCV).
# Alternatively run it interactively on the login node (slower).
#
# Usage:  sbatch setup_job.sh

#SBATCH --job-name=AAI_Setup
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

mkdir -p logs data/processed config

module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/.local/venv/bin/activate

echo "Starting setup on $(hostname)"
python3.11 setup.py --skip_preprocess --data Data/accepted.csv --n_jobs 32
echo "Setup complete."
