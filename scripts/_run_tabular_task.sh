#!/bin/bash
# Worker invoked by scripts/run_tabular.sh as a SLURM array task.
# Picks the SLURM_ARRAY_TASK_ID-th NML file from the sorted list and runs
# the tabular runner on it.

set -euo pipefail
cd /nfs/roberts/project/pi_btk22/jyz32/macronews
module load Python/3.12.3-GCCcore-13.3.0 >/dev/null 2>&1 || true

NML_DIR="/nfs/roberts/project/pi_btk22/rc2573/DJN"
OUT_DIR="${OUT_DIR:-results/tabular}"
MIN_MONTH="${MIN_MONTH:-1996-01}"
# Filter NML files by month, keep only >= MIN_MONTH
mapfile -t FILES < <(
    ls "$NML_DIR"/*.nml \
      | awk -v min="$MIN_MONTH" -v dir="$NML_DIR" '{
            n=$0; sub(".*/","",n); sub("\\.nml$","",n);
            if (n >= min) print $0
        }' \
      | sort
)
NML_FILE="${FILES[$SLURM_ARRAY_TASK_ID]}"
BASENAME=$(basename "$NML_FILE" .nml)
OUTPUT="${OUT_DIR}/${BASENAME}.jsonl"

echo "[task ${SLURM_ARRAY_TASK_ID}] $NML_FILE -> $OUTPUT"
.venv/bin/python src/tabular/runner.py --nml-file "$NML_FILE" --output "$OUTPUT"
