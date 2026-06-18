#!/bin/bash
# SLURM job that starts HyperQueue workers on the allocated node.
# The workers fetch tasks submitted via hq_task.sh from the HQ server
# running on the login node.
#
# RESOURCE CALCULATION (192-core genoa node, 8 cores/task):
#   Concurrent tasks : 192 / 8  = 24
#   Total tasks      : 5 x 4 x 5 = 100
#   Rounds           : ceil(100/24) = 5
#   Estimated time   : 5 rounds x ~90 min worst-case = ~7.5h
#   Requested        : 20h (generous safety margin)
#
# HOW TO USE:
#   On the login node (before sbatch):
#     1.  module load 2023 && module load HyperQueue/0.19.0
#     2.  nohup hq server start &
#     3.  hq submit --array 0-99 \
#             --stdout=logs/%{TASK_ID}.out \
#             --stderr=logs/%{TASK_ID}.err \
#             --pin taskset --cpus=8 --time-limit=1300min \
#             hq_task.sh
#     4.  sbatch hyperq_job_hpc.sh

#SBATCH --job-name=AAI_HQ_Workers
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

mkdir -p logs

module load 2023
module load HyperQueue/0.19.0
module load OpenMPI/4.1.5-GCC-12.3.0

echo "Starting HQ workers on $(hostname) with $SLURM_CPUS_ON_NODE cores"

# Each worker connects to the HQ server and picks up queued tasks
srun --overlap hq worker start --manager slurm --idle-timeout=5min

echo "HQ workers finished."
