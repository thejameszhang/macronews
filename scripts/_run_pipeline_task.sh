#!/bin/bash
# Worker invoked by scripts/run_pipeline_array.sh as a SLURM array task.
# Reads SHARD basename from the SLURM_ARRAY_TASK_ID-th line of MANIFEST_FILE
# and runs src/pipeline.py on the corresponding *_clean.jsonl.

set -euo pipefail
cd /nfs/roberts/project/pi_btk22/jyz32/macronews
module load Python/3.12.3-GCCcore-13.3.0 >/dev/null 2>&1 || true
source .venv/bin/activate

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

# Deterministic vLLM output: ~0.09% tag drift on DJNW 1000 articles
# vs ~29% without the flag. Requires compute capability >= 9.0.
export VLLM_BATCH_INVARIANT=1

PYTHONPATH=src python src/pipeline.py \
    --model "$MODEL_PATH" \
    --max-model-len "$MAX_MODEL_LEN" \
    --mode prod \
    --input-file "$INPUT_FILE" \
    --output-dir "$OUT_DIR"
