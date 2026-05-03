#!/bin/bash
# SLURM array launcher for the filter waterfall.
# Per-month worker writes to TMP_DIR; dependent aggregator merges into
# the final CSV + markdown report and deletes the temp dir.

set -euo pipefail
cd /nfs/roberts/project/pi_btk22/jyz32/macronews

TMP_DIR="${TMP_DIR:-results/_tmp_waterfall_per_month}"
OUT_CSV="${OUT_CSV:-results/filter_waterfall.csv}"
OUT_MD="${OUT_MD:-docs/filter_audit/filter_waterfall.md}"
PARTITION="${PARTITION:-priority}"
ACCOUNT="${ACCOUNT:-prio_btk22}"
WALLTIME="${WALLTIME:-00:15:00}"
MAX_CONCURRENT=100
if [ "${1:-}" = "--max" ] && [ -n "${2:-}" ]; then
    MAX_CONCURRENT="$2"
fi
mkdir -p logs "$TMP_DIR"
export TMP_DIR

# All months in cleaned shards (file basename like 2014-10a_clean.jsonl → 2014-10)
mapfile -t MONTHS < <(
    ls /nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles/*_clean.jsonl \
      | sed 's|.*/||;s|[a-z]\?_clean\.jsonl$||' \
      | sort -u
)
N=${#MONTHS[@]}
echo "Submitting array job for $N months (max $MAX_CONCURRENT concurrent)."
echo "  temp:    $TMP_DIR"
echo "  out csv: $OUT_CSV"
echo "  out md:  $OUT_MD"

worker=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$WALLTIME" \
    --cpus-per-task=2 \
    --mem=8G \
    --job-name=wf_worker \
    --output=logs/wf_worker_%A_%a.out \
    --export=ALL,TMP_DIR \
    --array=0-$((N - 1))%${MAX_CONCURRENT} \
    scripts/_filter_waterfall_task.sh)

echo "Worker array: $worker"

agg=$(sbatch --parsable \
    --dependency=afterok:$worker \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time=00:05:00 \
    --cpus-per-task=1 \
    --mem=2G \
    --job-name=wf_agg \
    --output=logs/wf_agg_%j.out \
    --wrap="module load Python/3.12.3-GCCcore-13.3.0 >/dev/null 2>&1; .venv/bin/python scripts/per_month_waterfall_aggregate.py --in-dir ${TMP_DIR} --out-csv ${OUT_CSV} --out-md ${OUT_MD} && rm -rf ${TMP_DIR} && echo 'cleaned tmp dir'")

echo "Aggregator: $agg"
echo "Final CSV:    $OUT_CSV"
echo "Final report: $OUT_MD"
