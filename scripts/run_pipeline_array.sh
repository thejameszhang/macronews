#!/bin/bash
#
# SLURM array launcher for the macronews mapper across multiple shards.
# One array task per *_clean.jsonl shard in INPUT_DIR. Each task runs
# src/pipeline.py with --mode prod and writes one JSONL + summary to OUT_DIR.
#
# Idempotent: at submit time, shards where OUT_DIR/{shard}.jsonl AND
# OUT_DIR/{shard}.summary.json both already exist are dropped from the
# manifest. Re-running the launcher picks up only what's missing.
#
# Usage:
#   OUT_DIR=results/prod/v1 bash scripts/run_pipeline_array.sh                  # all shards, 4 concurrent
#   OUT_DIR=results/prod/v1 bash scripts/run_pipeline_array.sh --max 8          # 8 concurrent
#   OUT_DIR=results/prod/v1 PATTERN='2014-*' \
#     bash scripts/run_pipeline_array.sh                                        # all of 2014
#   OUT_DIR=results/prod/v1 PATTERN='2014-0[1-6]?' \
#     bash scripts/run_pipeline_array.sh                                        # first half of 2014
#   OUT_DIR=results/prod/v1 PATTERN='201[4-6]-*' \
#     bash scripts/run_pipeline_array.sh                                        # 2014-2016
#   OUT_DIR=results/prod/v1 PATTERN='2014-05c' \
#     bash scripts/run_pipeline_array.sh                                        # single shard
#   OUT_DIR=results/prod/v1 LOG_DIR=logs/prod PATTERN='2014-*' \
#     bash scripts/run_pipeline_array.sh --max 8                                # prod: per-shard logs to logs/prod/
#   OUT_DIR=results/prod/v1 DRY_RUN=1 bash scripts/run_pipeline_array.sh        # print plan, don't submit
#
# Env knobs:
#   OUT_DIR (REQUIRED), INPUT_DIR, PATTERN, LOG_DIR,
#   MODEL_PATH, MAX_MODEL_LEN, PARTITION, ACCOUNT, GRES, WALLTIME,
#   DRY_RUN
#
# PATTERN is a bash glob matched against shard basenames (e.g. "2014-05c").
# Default "*" matches every shard in INPUT_DIR.
#
# LOG_DIR (default "logs") is where the manifest, sbatch .out, and .err
# files land. For prod runs, set LOG_DIR=logs/prod so per-shard timing
# logs are kept separate from dev/gamma noise — the worker logs each
# task's start/end/elapsed timing in its .out file.

set -euo pipefail
cd /nfs/roberts/project/pi_btk22/jyz32/macronews

INPUT_DIR="${INPUT_DIR:-/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles}"
: "${OUT_DIR:?OUT_DIR must be set, e.g. OUT_DIR=results/prod/v1 (no default; prevents accidental writes to the wrong run)}"
LOG_DIR="${LOG_DIR:-logs}"
PATTERN="${PATTERN:-*}"

mkdir -p "$LOG_DIR"

MODEL_PATH="${MODEL_PATH:-/nfs/roberts/scratch/pi_btk22/jyz32/gemma-4-26b-a4b-it}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
PARTITION="${PARTITION:-priority_gpu}"
ACCOUNT="${ACCOUNT:-prio_btk22}"
GRES="${GRES:-gpu:b200:1}"
WALLTIME="${WALLTIME:-24:00:00}"
# a1117u29n01 is ~3.4x slower than peer B200 nodes (observed on 2014 prod run:
# 6h35m / 6h48m on this node vs ~2h on others). Override with EXCLUDE_NODES=
# (empty) to allow it back in if Yale fixes it.
EXCLUDE_NODES="${EXCLUDE_NODES-a1117u29n01}"

MAX_CONCURRENT=4
if [ "${1:-}" = "--max" ] && [ -n "${2:-}" ]; then
    MAX_CONCURRENT="$2"
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: INPUT_DIR not found: $INPUT_DIR"
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH not found: $MODEL_PATH"
    exit 1
fi
mkdir -p "$OUT_DIR"

# Build manifest of shards to process. Each line is a SHARD basename
# (e.g. "2014-05c"), without the _clean.jsonl suffix.
MANIFEST=$(mktemp "$LOG_DIR/pipeline_array_manifest_XXXXXX.txt")
trap '[ -f "$MANIFEST" ] && rm -f "$MANIFEST"' EXIT

total=0
skipped_done=0
skipped_pattern=0
while IFS= read -r path; do
    base=$(basename "$path")
    shard="${base%_clean.jsonl}"
    total=$((total + 1))

    # PATTERN is a bash glob matched against the shard basename.
    case "$shard" in
        $PATTERN) ;;
        *)
            skipped_pattern=$((skipped_pattern + 1))
            continue
            ;;
    esac

    # Idempotency: both files must exist for the shard to be "complete".
    if [ -f "$OUT_DIR/${shard}.jsonl" ] && [ -f "$OUT_DIR/${shard}.summary.json" ]; then
        skipped_done=$((skipped_done + 1))
        continue
    fi

    echo "$shard" >> "$MANIFEST"
done < <(ls "$INPUT_DIR"/*_clean.jsonl 2>/dev/null | sort)

N=$(wc -l < "$MANIFEST")
echo "Shards in $INPUT_DIR:    $total"
echo "Pattern:                 '$PATTERN'"
echo "Skipped (pattern):       $skipped_pattern"
echo "Skipped (already done):  $skipped_done"
echo "To submit:               $N"

if [ "$N" -eq 0 ]; then
    echo "Nothing to do."
    rm -f "$MANIFEST"
    exit 0
fi

# Persist manifest under a stable name so workers can read it without
# racing against trap cleanup.
PERSIST_MANIFEST="$LOG_DIR/pipeline_array_manifest_$(date +%Y%m%d_%H%M%S).txt"
mv "$MANIFEST" "$PERSIST_MANIFEST"
trap - EXIT
echo "Manifest:                $PERSIST_MANIFEST"
echo "Log dir:                 $LOG_DIR"
echo "Concurrency:             $MAX_CONCURRENT"
echo "Output dir:              $OUT_DIR"
echo "Walltime per task:       $WALLTIME"
echo "Partition / account:     $PARTITION / $ACCOUNT"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "--- DRY_RUN=1, not submitting. Full manifest ($N shards): ---"
    cat "$PERSIST_MANIFEST"
    exit 0
fi

export INPUT_DIR OUT_DIR MODEL_PATH MAX_MODEL_LEN
export MANIFEST_FILE="$PERSIST_MANIFEST"

EXCLUDE_FLAG=""
if [ -n "$EXCLUDE_NODES" ]; then
    EXCLUDE_FLAG="--exclude=${EXCLUDE_NODES}"
fi

jobid=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --gres="$GRES" \
    --time="$WALLTIME" \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=128G \
    --job-name=pipeline_arr \
    --output="$LOG_DIR/pipeline_arr_%A_%a.out" \
    --error="$LOG_DIR/pipeline_arr_%A_%a.err" \
    --export=ALL,INPUT_DIR,OUT_DIR,MODEL_PATH,MAX_MODEL_LEN,MANIFEST_FILE \
    ${EXCLUDE_FLAG} \
    --array=0-$((N - 1))%${MAX_CONCURRENT} \
    scripts/_run_pipeline_task.sh)

echo "Submitted array job $jobid (tasks 0-$((N - 1)), %${MAX_CONCURRENT})"
echo "Monitor:    squeue -j $jobid"
echo "Per-task:   $LOG_DIR/pipeline_arr_${jobid}_<task>.out"
echo "Outputs:    $OUT_DIR/{shard}.jsonl + .summary.json"
