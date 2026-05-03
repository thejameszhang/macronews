#!/bin/bash
#
# SLURM array launcher for the tabular-body detector.
# One array task per monthly NML file in /nfs/roberts/.../rc2573/DJN/*.nml.
# Each task reads one NML file and writes one results/tabular/{YYYY-MM}.jsonl.
#
# Usage:
#   bash scripts/run_tabular.sh                   # 50 concurrent tasks
#   bash scripts/run_tabular.sh --max 100         # 100 concurrent tasks
#
# Env knobs:
#   PARTITION (default: day), ACCOUNT (default: pi_btk22), WALLTIME (default: 00:30:00)
#

set -euo pipefail
cd /nfs/roberts/project/pi_btk22/jyz32/macronews
OUT_DIR="${OUT_DIR:-results/tabular}"
MIN_MONTH="${MIN_MONTH:-1996-01}"
mkdir -p logs "$OUT_DIR"
export OUT_DIR MIN_MONTH

NML_DIR="/nfs/roberts/project/pi_btk22/rc2573/DJN"
PARTITION="${PARTITION:-day}"
ACCOUNT="${ACCOUNT:-pi_btk22}"
WALLTIME="${WALLTIME:-00:30:00}"
MAX_CONCURRENT=50
if [ "${1:-}" = "--max" ] && [ -n "${2:-}" ]; then
    MAX_CONCURRENT="$2"
fi

N=$(ls "$NML_DIR"/*.nml | sed 's|.*/||;s|\.nml$||' | awk -v min="$MIN_MONTH" '$0 >= min' | wc -l)
echo "Submitting array job for $N NML files (>= $MIN_MONTH), up to $MAX_CONCURRENT concurrent."

jobid=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$WALLTIME" \
    --cpus-per-task=2 \
    --mem=8G \
    --job-name=tabular \
    --output=logs/tabular_%A_%a.out \
    --export=ALL,OUT_DIR,MIN_MONTH \
    --array=0-$((N - 1))%${MAX_CONCURRENT} \
    scripts/_run_tabular_task.sh)

echo "Submitted job $jobid (array tasks 0-$((N - 1)))"
echo "Monitor:    squeue -j $jobid"
echo "Per-task logs: logs/tabular_${jobid}_<task>.out"
echo "Sidecars:   $OUT_DIR/{YYYY-MM}.jsonl"
