#!/bin/bash
# Report SLURM efficiency, HyperQueue status and collected task results
# Usage: bash post_run_report.sh <slurm_jobid> [hq_job_id ...]
# Without HQ job IDs, inspect all HyperQueue jobs

JOBID=${1:?usage: post_run_report.sh <slurm_jobid> [hq_job_id ...]}
shift
HQJOBS="$*"

module load 2023 >/dev/null 2>&1
module load HyperQueue/0.19.0 >/dev/null 2>&1

echo "=== seff $JOBID (CPU efficiency of the whole worker job) ==="
seff "$JOBID"

echo
echo "=== sacct $JOBID ==="
sacct -j "$JOBID" --format=JobID,JobName%20,State,Elapsed,TotalCPU,CPUTime,AllocCPUS,MaxRSS,NodeList

echo
echo "=== HyperQueue jobs ==="
hq job list --all
if [ -z "$HQJOBS" ]; then
    HQJOBS=$(hq job list --all --output-mode=json 2>/dev/null \
        | python3 -c "import json,sys; print(' '.join(str(j['id']) for j in json.load(sys.stdin)))" 2>/dev/null)
fi
for HQJOB in $HQJOBS; do
    echo
    echo "=== HyperQueue job $HQJOB ==="
    hq job info "$HQJOB"
    echo "--- tasks that did not finish"
    hq task list "$HQJOB" 2>/dev/null | grep -i -E "fail|cancel" || echo "none"
done

echo
echo "=== Task report (per-task stage times, CPU efficiency, memory) ==="
module load Python/3.11.3-GCCcore-12.3.0 >/dev/null 2>&1
source ~/.local/venv/bin/activate
python3.11 aggregate.py || echo "aggregate.py failed" >&2
echo "Per-task details: outputs/task_report.csv; node snapshots: logs/monitor_${JOBID}.log"
