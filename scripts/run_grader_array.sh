#!/bin/bash
#
# SLURM array launcher for the macronews grader across multiple shards.
# One array task per shard listed in MAPPER_DIR. Each task joins
# MAPPER_DIR/{shard}.jsonl with INPUT_DIR/{shard}_clean.jsonl, calls
# QwQ-32B (or compatible) via src/mapping/grading/runner.py, and writes
# OUT_DIR/{shard}.jsonl + .summary.json.
#
# Idempotent: at submit time, shards where OUT_DIR/{shard}.jsonl AND
# OUT_DIR/{shard}.summary.json both already exist are dropped from the
# manifest. Shards whose mapper output is missing are skipped with a
# warning (you cannot grade what wasn't mapped).
#
# Usage:
#   OUT_DIR=results/mapper/prod/v1-grader \
#     MAPPER_DIR=results/mapper/prod/v1 \
#     bash scripts/run_grader_array.sh                                       # all shards in MAPPER_DIR, 4 concurrent
#   OUT_DIR=results/mapper/prod/v1-grader \
#     MAPPER_DIR=results/mapper/prod/v1 \
#     bash scripts/run_grader_array.sh --max 8                               # 8 concurrent
#   OUT_DIR=results/mapper/prod/v1-grader \
#     MAPPER_DIR=results/mapper/prod/v1 \
#     PATTERN='2014-0[1-6]?' \
#     bash scripts/run_grader_array.sh                                       # first half of 2014
#   OUT_DIR=results/mapper/prod/v1-grader \
#     MAPPER_DIR=results/mapper/prod/v1 \
#     DRY_RUN=1 bash scripts/run_grader_array.sh                             # print plan, don't submit
#
# Env knobs:
#   OUT_DIR (REQUIRED), MAPPER_DIR, INPUT_DIR, PATTERN, LOG_DIR,
#   MODEL_PATH, MAX_MODEL_LEN, MAX_TOKENS,
#   PARTITION, ACCOUNT, GRES, WALLTIME, EXCLUDE_NODES, DRY_RUN
#
# PATTERN is a bash glob matched against shard basenames (e.g. "2014-05c").
# Default "*" matches every shard found in MAPPER_DIR.
#
# LOG_DIR (default "logs") receives the manifest, sbatch .out, and .err
# files. For prod runs, consider LOG_DIR=logs/prod-grader so per-shard
# timing logs stay separate from dev/gamma noise.

set -euo pipefail
# Resolve repo root from script location so the launcher works regardless
# of which clone or which user runs it.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MAPPER_DIR="${MAPPER_DIR:-results/mapper/prod/v1}"
INPUT_DIR="${INPUT_DIR:-/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles}"
: "${OUT_DIR:?OUT_DIR must be set, e.g. OUT_DIR=results/mapper/prod/v1-grader (no default; prevents accidental writes to the wrong run)}"
LOG_DIR="${LOG_DIR:-logs}"
PATTERN="${PATTERN:-*}"

mkdir -p "$LOG_DIR"

MODEL_PATH="${MODEL_PATH:-/nfs/roberts/scratch/pi_btk22/jyz32/qwq-32b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
# FREE B200 tier is the DEFAULT (gpu_b200 / pi_btk22 / QOS normal). Do NOT default to
# priority_gpu/prio_btk22 — that BILLS the lab's paid allocation, and this is the launcher
# a full-corpus grader run uses. Pass those explicitly only when a run needs the queue.
PARTITION="${PARTITION:-gpu_b200}"
ACCOUNT="${ACCOUNT:-pi_btk22}"
QOS="${QOS:-normal}"
GRES="${GRES:-gpu:b200:1}"
WALLTIME="${WALLTIME:-12:00:00}"
# a1117u29n01 is ~3.4x slower than peer B200 nodes (observed on 2014 prod run:
# 6h35m / 6h48m on this node vs ~2h on others). Override with EXCLUDE_NODES=
# (empty) to allow it back in if Yale fixes it.
EXCLUDE_NODES="${EXCLUDE_NODES-a1117u29n01}"

MAX_CONCURRENT=4
if [ "${1:-}" = "--max" ] && [ -n "${2:-}" ]; then
    MAX_CONCURRENT="$2"
fi

if [ ! -d "$MAPPER_DIR" ]; then
    echo "ERROR: MAPPER_DIR not found: $MAPPER_DIR"
    exit 1
fi
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: INPUT_DIR not found: $INPUT_DIR"
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH not found: $MODEL_PATH"
    echo "       Submit the download first:"
    echo "         MODEL_REPO=Qwen/QwQ-32B LOCAL_DIR=$MODEL_PATH \\"
    echo "             sbatch scripts/download_llm.sh"
    exit 1
fi
mkdir -p "$OUT_DIR"

# Build manifest of shards to process. Each line is a SHARD basename
# (e.g. "2014-05c"), without the .jsonl suffix. We enumerate from
# MAPPER_DIR — only shards we actually mapped are gradable.
MANIFEST=$(mktemp "$LOG_DIR/grader_array_manifest_XXXXXX.txt")
trap '[ -f "$MANIFEST" ] && rm -f "$MANIFEST"' EXIT

total=0
skipped_done=0
skipped_pattern=0
skipped_no_source=0
while IFS= read -r path; do
    base=$(basename "$path")
    # Skip *.summary.json — we only want the mapper output JSONLs.
    case "$base" in
        *.summary.json) continue ;;
    esac
    shard="${base%.jsonl}"
    total=$((total + 1))

    # PATTERN is a bash glob matched against the shard basename.
    case "$shard" in
        $PATTERN) ;;
        *)
            skipped_pattern=$((skipped_pattern + 1))
            continue
            ;;
    esac

    # Source must exist — grader needs the cleaned shard for paragraphs.
    if [ ! -f "$INPUT_DIR/${shard}_clean.jsonl" ]; then
        echo "WARN: skipping $shard — source missing at $INPUT_DIR/${shard}_clean.jsonl"
        skipped_no_source=$((skipped_no_source + 1))
        continue
    fi

    # Idempotency: both files must exist for the shard to be "complete".
    if [ -f "$OUT_DIR/${shard}.jsonl" ] && [ -f "$OUT_DIR/${shard}.summary.json" ]; then
        skipped_done=$((skipped_done + 1))
        continue
    fi

    echo "$shard" >> "$MANIFEST"
done < <(ls "$MAPPER_DIR"/*.jsonl 2>/dev/null | sort)

N=$(wc -l < "$MANIFEST")
echo "Shards in $MAPPER_DIR:    $total"
echo "Pattern:                 '$PATTERN'"
echo "Skipped (pattern):       $skipped_pattern"
echo "Skipped (no source):     $skipped_no_source"
echo "Skipped (already done):  $skipped_done"
echo "To submit:               $N"

if [ "$N" -eq 0 ]; then
    echo "Nothing to do."
    rm -f "$MANIFEST"
    exit 0
fi

# Persist manifest under a stable name so workers can read it without
# racing against trap cleanup.
PERSIST_MANIFEST="$LOG_DIR/grader_array_manifest_$(date +%Y%m%d_%H%M%S).txt"
mv "$MANIFEST" "$PERSIST_MANIFEST"
trap - EXIT
echo "Manifest:                $PERSIST_MANIFEST"
echo "Log dir:                 $LOG_DIR"
echo "Concurrency:             $MAX_CONCURRENT"
echo "Mapper dir:              $MAPPER_DIR"
echo "Input dir:               $INPUT_DIR"
echo "Output dir:              $OUT_DIR"
echo "Walltime per task:       $WALLTIME"
echo "Partition / account:     $PARTITION / $ACCOUNT"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "--- DRY_RUN=1, not submitting. Full manifest ($N shards): ---"
    cat "$PERSIST_MANIFEST"
    exit 0
fi

export MAPPER_DIR INPUT_DIR OUT_DIR MODEL_PATH MAX_MODEL_LEN MAX_TOKENS
export MANIFEST_FILE="$PERSIST_MANIFEST"

EXCLUDE_FLAG=""
if [ -n "$EXCLUDE_NODES" ]; then
    EXCLUDE_FLAG="--exclude=${EXCLUDE_NODES}"
fi

jobid=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --qos="$QOS" \
    --gres="$GRES" \
    --time="$WALLTIME" \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=128G \
    --job-name=grader_arr \
    --output="$LOG_DIR/grader_arr_%A_%a.out" \
    --error="$LOG_DIR/grader_arr_%A_%a.err" \
    --export=ALL,MAPPER_DIR,INPUT_DIR,OUT_DIR,MODEL_PATH,MAX_MODEL_LEN,MAX_TOKENS,MANIFEST_FILE \
    ${EXCLUDE_FLAG} \
    --array=0-$((N - 1))%${MAX_CONCURRENT} \
    scripts/_run_grader_task.sh)

echo "Submitted array job $jobid (tasks 0-$((N - 1)), %${MAX_CONCURRENT})"
echo "Monitor:    squeue -j $jobid"
echo "Per-task:   $LOG_DIR/grader_arr_${jobid}_<task>.out"
echo "Outputs:    $OUT_DIR/{shard}.jsonl + .summary.json"
