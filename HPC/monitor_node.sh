#!/bin/bash
# Log node load, memory and thread use while the worker job runs
# Started by hyperq_job_hpc.sh; writes logs/monitor_<jobid>.log
# Set INTERVAL to change the sampling interval in seconds

INTERVAL=${INTERVAL:-600}
LOG="logs/monitor_${SLURM_JOB_ID:-local}.log"
mkdir -p logs

while true; do
    {
        echo "==================== $(date '+%Y-%m-%d %H:%M:%S') $(hostname) ===================="
        echo "--- load average (192 = all cores busy on a genoa node)"
        uptime
        echo "--- memory (GiB)"
        free -g
        echo "--- busiest threads: top -H (one line per thread, %CPU of one core)"
        top -H -b -n 1 -w 200 | head -45
        echo "--- per process: threads (nlwp), CPU %, RSS, the core it ran on last (psr)"
        ps -u "$USER" -o pid,nlwp,pcpu,rss,psr,etime,args --sort=-pcpu | head -30
        if command -v hq >/dev/null 2>&1; then
            echo "--- HyperQueue jobs"
            hq job list 2>/dev/null | head -20
        fi
    } >> "$LOG" 2>&1
    sleep "$INTERVAL"
done
