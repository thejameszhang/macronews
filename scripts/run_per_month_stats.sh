#!/bin/bash
# Submit per-month tabular_body stats as a SLURM array (one task per month),
# then a dependent aggregate job.

set -euo pipefail
cd /nfs/roberts/project/pi_btk22/jyz32/macronews

SIDECAR_DIR="${SIDECAR_DIR:-results/tabular}"
STATS_DIR="${STATS_DIR:-results/tabular_stats}"
AGG_MD="${AGG_MD:-docs/filter_audit/tabular_body/per_month_stats.md}"
AGG_JSON="${AGG_JSON:-docs/filter_audit/tabular_body/per_month_stats.json}"
mkdir -p logs "$STATS_DIR"
export SIDECAR_DIR STATS_DIR

PARTITION="${PARTITION:-priority}"
ACCOUNT="${ACCOUNT:-prio_btk22}"
WALLTIME="${WALLTIME:-00:30:00}"
MAX_CONCURRENT=100
if [ "${1:-}" = "--max" ] && [ -n "${2:-}" ]; then
    MAX_CONCURRENT="$2"
fi

# Post-1996 only — project scope
N=$(ls "$SIDECAR_DIR"/*.jsonl | sed 's|.*/||;s|\.jsonl$||' | sort | awk '$0 >= "1996"' | wc -l)
echo "Submitting array job for $N post-1996 months (max $MAX_CONCURRENT concurrent)."
echo "  sidecars: $SIDECAR_DIR"
echo "  stats:    $STATS_DIR"

worker=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$WALLTIME" \
    --cpus-per-task=2 \
    --mem=8G \
    --job-name=stats \
    --output=logs/stats_%A_%a.out \
    --export=ALL,SIDECAR_DIR,STATS_DIR \
    --array=0-$((N - 1))%${MAX_CONCURRENT} \
    scripts/_per_month_stats_task.sh)

echo "Worker array: $worker"

agg=$(sbatch --parsable \
    --dependency=afterok:$worker \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time=00:10:00 \
    --cpus-per-task=1 \
    --mem=2G \
    --job-name=stats_agg \
    --output=logs/stats_agg_%j.out \
    --wrap="module load Python/3.12.3-GCCcore-13.3.0 >/dev/null 2>&1; .venv/bin/python scripts/per_month_stats_aggregate.py --in-dir ${STATS_DIR} --out ${AGG_MD} --out-json ${AGG_JSON}")

echo "Aggregator: $agg"
echo "Final report: $AGG_MD"
