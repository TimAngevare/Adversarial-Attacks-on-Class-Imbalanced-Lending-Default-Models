#!/bin/bash
# Compare one experiment across different core counts
# Usage: sbatch scaling_job.sh [experiment_id] ["1 2 4 8"]
# Default: experiment 60, XGBoost + Baseline at 10% prevalence
# Run python3.11 scaling_report.py to inspect timings and efficiency
# Record failures separately and exclude them from speed-up calculations

#SBATCH --job-name=AAI_Scaling
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=42G
#SBATCH --time=10:00:00
#SBATCH --output=logs/scaling_%j.out
#SBATCH --error=logs/scaling_%j.err

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p logs results/scaling

module load 2023 || exit 1
module load Python/3.11.3-GCCcore-12.3.0 || exit 1
source ~/.local/venv/bin/activate || exit 1

# Select the experiment and core counts for the scaling comparison
EXP_ID=${1:-60}
CORE_COUNTS=${2:-"1 2 4 8"}
export PYTHONUNBUFFERED=1
export LOCAL_DATA_DIR=data/processed

STATUS_FILE="results/scaling/status_exp_${EXP_ID}.tsv"
printf "cores\texit_code\twall_seconds\tslurm_job\n" > "$STATUS_FILE"
N_FAILED=0

ALLOWED=$(python3.11 -c "import os; print(' '.join(map(str, sorted(os.sched_getaffinity(0)))))")
echo "Job $SLURM_JOB_ID on $(hostname); allowed cores: $ALLOWED"

for N in $CORE_COUNTS; do
    CPUS=$(echo $ALLOWED | tr ' ' '\n' | head -n "$N" | paste -sd, -)
    if [ "$(echo "$CPUS" | tr ',' '\n' | wc -l)" -lt "$N" ]; then
        echo "  only $(echo "$CPUS" | tr ',' '\n' | wc -l) cores allowed, skipping N=$N" >&2
        printf "%s\t%s\t%s\t%s\n" "$N" "skipped" "" "$SLURM_JOB_ID" >> "$STATUS_FILE"
        continue
    fi
    OUT="results/scaling/exp_${EXP_ID}_cores_${N}"
    rm -rf "$OUT"          # every scaling run starts from an empty folder
    mkdir -p "$OUT"
    echo "[$(date)] experiment $EXP_ID with $N cores on cpus $CPUS"
    export OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N NUMEXPR_NUM_THREADS=$N
    START=$(date +%s)
    TIMECMD=""
    [ -x /usr/bin/time ] && TIMECMD="/usr/bin/time -v"
    $TIMECMD taskset -c "$CPUS" python3.11 main.py \
        --experiment_id="$EXP_ID" --n_cores="$N" --out_dir="$OUT" \
        > "$OUT/stdout.log" 2> "$OUT/time_and_stderr.log"
    CODE=$?
    WALL=$(( $(date +%s) - START ))
    printf "%s\t%s\t%s\t%s\n" "$N" "$CODE" "$WALL" "$SLURM_JOB_ID" >> "$STATUS_FILE"
    echo "  exit code $CODE, wall $WALL s"
    if [ "$CODE" -ne 0 ]; then
        N_FAILED=$(( N_FAILED + 1 ))
        echo "  FAILED; last lines of $OUT/time_and_stderr.log:" >&2
        tail -n 5 "$OUT/time_and_stderr.log" >&2
    fi
done

python3.11 scaling_report.py --exp_id "$EXP_ID"
REPORT=$?
if [ "$N_FAILED" -gt 0 ]; then
    echo "$N_FAILED scaling run(s) failed; see $STATUS_FILE" >&2
    exit 1
fi
exit $REPORT
