#!/bin/bash
# Submit four experiment arrays and the pricing sensitivity array
# Run from HPC/ after starting the HyperQueue server
# Check per-model memory estimates against task_report.csv after the smoke test
# Allow about 20% above the measured peak when setting memory requests

set -u
mkdir -p logs

CPUS=${CPUS:-16}
TIME_LIMIT=${TIME_LIMIT:-1300min}

declare -A MEM_MB=(
    [XGBClassifier]=20000
    [LogisticRegression]=16000
    [RandomForestClassifier]=48000
    [NeuralNetClassifier]=24000
)
MODELS=(XGBClassifier LogisticRegression RandomForestClassifier NeuralNetClassifier)

# Map IDs as 20 * prevalence index + 5 * model index + sampler index
for m in 0 1 2 3; do
    MODEL=${MODELS[$m]}
    IDS=""
    for f in 0 1 2 3 4; do
        START=$(( 20 * f + 5 * m ))
        IDS="${IDS:+$IDS,}${START}-$(( START + 4 ))"
    done
    echo "Submitting $MODEL: ids $IDS, ${MEM_MB[$MODEL]} MB, $CPUS cores"
    hq submit --name "exp_${MODEL}" --array "$IDS" \
        --stdout=logs/%{TASK_ID}.out --stderr=logs/%{TASK_ID}.err \
        --pin taskset --cpus="$CPUS" --resource "mem=${MEM_MB[$MODEL]}" \
        --time-limit="$TIME_LIMIT" \
        hq_task.sh || { echo "hq submit failed for $MODEL" >&2; exit 1; }
done

N_PRICING=$(python3.11 sensitivity.py --count 2>/dev/null || echo 30)
echo "Submitting pricing sensitivity: $N_PRICING tasks"
hq submit --name pricing --array "0-$(( N_PRICING - 1 ))" \
    --stdout=logs/pricing_%{TASK_ID}.out --stderr=logs/pricing_%{TASK_ID}.err \
    --pin taskset --cpus=4 --resource "mem=8000" --time-limit=120min \
    hq_task.sh sensitivity.py || { echo "hq submit failed for pricing" >&2; exit 1; }

hq job list
