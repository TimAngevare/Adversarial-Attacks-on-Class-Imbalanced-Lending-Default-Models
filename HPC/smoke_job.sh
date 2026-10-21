#!/bin/bash
# Compare parallel and serial attacks before the full experiment run
# Default: Baseline at 20% prevalence for all four models
# Check completion, worker start methods, attack agreement and peak memory
# Usage: mkdir -p logs && sbatch smoke_job.sh
# Example: sbatch smoke_job.sh "90" 16 32

#SBATCH --job-name=AAI_Smoke
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=84G
#SBATCH --time=08:00:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p logs results/smoke

module load 2023 || exit 1
module load Python/3.11.3-GCCcore-12.3.0 || exit 1
source ~/.local/venv/bin/activate || exit 1

EXP_IDS=${1:-"80 85 90 95"}
N_WORKERS=${2:-16}
N_CASES=${3:-32}
N_CORES=${SLURM_CPUS_PER_TASK:-16}

export OMP_NUM_THREADS=$N_CORES MKL_NUM_THREADS=$N_CORES OPENBLAS_NUM_THREADS=$N_CORES NUMEXPR_NUM_THREADS=$N_CORES
export PYTHONUNBUFFERED=1
export LOCAL_DATA_DIR=data/processed

echo "Smoke test on $(hostname), job $SLURM_JOB_ID: experiments $EXP_IDS, $N_WORKERS workers, $N_CASES cases"
N_FAILED=0
for EXP in $EXP_IDS; do
    for W in "$N_WORKERS" 1; do
        OUT="results/smoke/exp_${EXP}_workers_${W}"
        rm -rf "$OUT"; mkdir -p "$OUT"
        echo "[$(date)] experiment $EXP, attack workers $W"
        START=$(date +%s)
        python3.11 main.py --experiment_id="$EXP" --n_cores="$N_CORES" \
            --attack_workers="$W" --n_search_cases="$N_CASES" \
            --stages attacks --out_dir="$OUT" \
            > "$OUT/stdout.log" 2> "$OUT/stderr.log"
        CODE=$?
        echo "  exit code $CODE, wall $(( $(date +%s) - START )) s"
        if [ $CODE -ne 0 ]; then
            N_FAILED=$(( N_FAILED + 1 ))
            tail -n 5 "$OUT/stderr.log" >&2
        fi
    done
done

python3.11 compare_attack_runs.py --smoke_dir results/smoke
COMPARE=$?
if [ "$N_FAILED" -gt 0 ] || [ "$COMPARE" -ne 0 ]; then
    echo "Smoke test FAILED ($N_FAILED failed runs, comparison exit code $COMPARE)" >&2
    exit 1
fi
echo "Smoke test passed."
