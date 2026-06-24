#!/bin/bash
# Worker invoked by scripts/run_pipeline_array.sh as a SLURM array task.
# Reads SHARD basename from the SLURM_ARRAY_TASK_ID-th line of MANIFEST_FILE
# and runs src/pipeline.py on the corresponding *_clean.jsonl.

set -euo pipefail
# SLURM_SUBMIT_DIR is the cwd at sbatch time; the parent launcher cd's to
# the repo root before submitting, so this lands in the repo root.
cd "${SLURM_SUBMIT_DIR:?must be invoked via sbatch}"
module load Python/3.12.3-GCCcore-13.3.0 >/dev/null 2>&1 || true

: "${MANIFEST_FILE:?MANIFEST_FILE not set}"
: "${INPUT_DIR:?INPUT_DIR not set}"
: "${OUT_DIR:?OUT_DIR not set}"
: "${MODEL_PATH:?MODEL_PATH not set}"
: "${MAX_MODEL_LEN:?MAX_MODEL_LEN not set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

# Manifest is 0-indexed by line. sed -n is robust to large files.
SHARD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST_FILE")
if [ -z "$SHARD" ]; then
    echo "ERROR: empty shard at task index $SLURM_ARRAY_TASK_ID in $MANIFEST_FILE"
    exit 1
fi

INPUT_FILE="$INPUT_DIR/${SHARD}_clean.jsonl"
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: input not found: $INPUT_FILE"
    exit 1
fi

# Late re-check: if another task or a previous run completed this shard
# between launcher submit and worker start, skip rather than overwrite.
if [ -f "$OUT_DIR/${SHARD}.jsonl" ] && [ -f "$OUT_DIR/${SHARD}.summary.json" ]; then
    echo "[task ${SLURM_ARRAY_TASK_ID}] $SHARD already complete, skipping"
    exit 0
fi

echo "[task ${SLURM_ARRAY_TASK_ID}] $SHARD: $INPUT_FILE -> $OUT_DIR"
echo "[task ${SLURM_ARRAY_TASK_ID}] $SHARD start: $(date -Iseconds)"
START_EPOCH=$(date +%s)

# Print start/end/elapsed even if pipeline.py exits non-zero — `set -e`
# would otherwise skip the elapsed line on failure, hiding wall-clock
# data for failed shards (which we still want for capacity planning).
trap '_end=$(date +%s); echo "[task ${SLURM_ARRAY_TASK_ID}] $SHARD end:   $(date -Iseconds)"; echo "[task ${SLURM_ARRAY_TASK_ID}] $SHARD elapsed: $((_end - START_EPOCH))s ($(( (_end - START_EPOCH) / 60 ))m $(( (_end - START_EPOCH) % 60 ))s)"' EXIT

# Deterministic vLLM output: ~0.09% tag drift on DJNW 1000 articles
# vs ~29% without the flag. Requires compute capability >= 9.0.
export VLLM_BATCH_INVARIANT=1

# Venv python by ABSOLUTE PATH (not plain 'python'): module load re-prepends its
# bin/, so 'python' can resolve to the system interpreter. Same pattern as
# _run_grader_task.sh / run_kg.sh.
PYTHONPATH=src .venv/bin/python src/pipeline.py \
    --model "$MODEL_PATH" \
    --max-model-len "$MAX_MODEL_LEN" \
    --mode prod \
    --input-file "$INPUT_FILE" \
    --output-dir "$OUT_DIR"
