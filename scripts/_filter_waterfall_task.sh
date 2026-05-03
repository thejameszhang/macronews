#!/bin/bash
# Worker for run_filter_waterfall.sh — processes one month's waterfall counts.

set -euo pipefail
cd /nfs/roberts/project/pi_btk22/jyz32/macronews
module load Python/3.12.3-GCCcore-13.3.0 >/dev/null 2>&1 || true

TMP_DIR="${TMP_DIR:-results/_tmp_waterfall_per_month}"
mapfile -t MONTHS < <(
    ls /nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles/*_clean.jsonl \
      | sed 's|.*/||;s|[a-z]\?_clean\.jsonl$||' \
      | sort -u
)
MONTH="${MONTHS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="${TMP_DIR}/${MONTH}.json"

mkdir -p "$TMP_DIR"
echo "[task ${SLURM_ARRAY_TASK_ID}] month=${MONTH} -> ${OUTPUT}"
.venv/bin/python scripts/per_month_waterfall.py --month "$MONTH" --out "$OUTPUT"
