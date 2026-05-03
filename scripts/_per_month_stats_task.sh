#!/bin/bash
# Worker for run_per_month_stats.sh — processes one month's stats.

set -euo pipefail
cd /nfs/roberts/project/pi_btk22/jyz32/macronews
module load Python/3.12.3-GCCcore-13.3.0 >/dev/null 2>&1 || true

SIDECAR_DIR="${SIDECAR_DIR:-results/tabular}"
STATS_DIR="${STATS_DIR:-results/tabular_stats}"

# Post-1996 only — project scope
mapfile -t MONTHS < <(ls "$SIDECAR_DIR"/*.jsonl | sed 's|.*/||;s|\.jsonl$||' | sort | awk '$0 >= "1996"')
MONTH="${MONTHS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="${STATS_DIR}/${MONTH}.json"

mkdir -p "$STATS_DIR"
echo "[task ${SLURM_ARRAY_TASK_ID}] month=${MONTH} sidecars=${SIDECAR_DIR} -> ${OUTPUT}"
.venv/bin/python scripts/per_month_stats_one.py --month "$MONTH" --out "$OUTPUT" --sidecar-dir "$SIDECAR_DIR"
