#!/bin/bash
# Start HyperQueue workers on one exclusive genoa node
# Start the HQ server and submit tasks before running sbatch
# Use task_report.csv to review stage times and memory estimates
# HyperQueue memory requests guide scheduling; they are not enforced limits

#SBATCH --job-name=AAI_HQ_Workers
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
mkdir -p logs

module load 2023 || exit 1
module load HyperQueue/0.19.0 || exit 1
module load OpenMPI/4.1.5-GCC-12.3.0 || exit 1

# Stage processed data on local storage when /scratch-node is available
SCRATCH_DIR="/scratch-node/$USER/$SLURM_JOB_ID"

if [ -d "/scratch-node" ]; then
    echo "Fast NVMe detected. Staging data to $SCRATCH_DIR/data/processed"
    mkdir -p "$SCRATCH_DIR/data"
    if ! cp -r data/processed "$SCRATCH_DIR/data/"; then
        echo "Staging to NVMe failed; tasks will read from network storage." >&2
        rm -rf "$SCRATCH_DIR/data/processed"
    fi
else
    echo "No NVMe drive detected on this node. Defaulting to network storage."
fi

# Record node load, memory and thread use every 10 minutes
INTERVAL=600 bash monitor_node.sh &
MONITOR_PID=$!

cleanup() {
    kill "$MONITOR_PID" 2>/dev/null
    if [ -d "$SCRATCH_DIR" ]; then
        echo "Cleaning up NVMe..."
        rm -rf "$SCRATCH_DIR"
    fi
}
trap cleanup EXIT

echo "Starting HQ workers on $(hostname) with $SLURM_CPUS_ON_NODE cores"
echo "Live view from the login node:  bash watch_node.sh $SLURM_JOB_ID"

# Reserve memory for the OS and monitoring when setting worker capacity
WORKER_MEM_MB=${WORKER_MEM_MB:-330000}

# Connect the worker to the server and process queued tasks
srun --overlap hq worker start --manager slurm --idle-timeout=5min \
    --resource "mem=sum(${WORKER_MEM_MB})"
STATUS=$?

if [ $STATUS -ne 0 ]; then
    echo "HQ worker exited with code $STATUS" >&2
fi
echo "HQ workers finished. After the run:  bash post_run_report.sh $SLURM_JOB_ID"
exit $STATUS
