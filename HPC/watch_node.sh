#!/bin/bash
# Inspect a running SLURM job from the login node
# Usage: bash watch_node.sh <slurm_jobid> [command]
# Default: top -H; press q to close the view
# Reuse the existing allocation with srun --overlap

JOBID=${1:?usage: watch_node.sh <slurm_jobid> [command]}
shift
CMD=${*:-top -H}
srun --jobid="$JOBID" --overlap --pty $CMD
