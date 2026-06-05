#!/bin/bash
# Worker invoked by scripts/run_grader_array.sh as a SLURM array task.
# Reads SHARD basename from the SLURM_ARRAY_TASK_ID-th line of MANIFEST_FILE
# and runs src/mapping/grading/runner.py joining MAPPER_DIR/{shard}.jsonl with
# INPUT_DIR/{shard}_clean.jsonl, writing OUT_DIR/{shard}.jsonl.

set -euo pipefail
# SLURM_SUBMIT_DIR is the cwd at sbatch time; the parent launcher cd's to
# the repo root before submitting, so this lands in the repo root.
cd "${SLURM_SUBMIT_DIR:?must be invoked via sbatch}"
# Module load brings libpython3.12.so.1.0 onto LD_LIBRARY_PATH. We do NOT
# `source .venv/bin/activate` — on this cluster, module load re-prepends
# its bin/ to PATH, so plain `python` resolves to the system interpreter
# (which lacks pydantic/vllm). Call .venv/bin/python by absolute path.
module load Python/3.12.3-GCCcore-13.3.0 >/dev/null 2>&1 || true

: "${MANIFEST_FILE:?MANIFEST_FILE not set}"
: "${MAPPER_DIR:?MAPPER_DIR not set}"
: "${INPUT_DIR:?INPUT_DIR not set}"
: "${OUT_DIR:?OUT_DIR not set}"
: "${MODEL_PATH:?MODEL_PATH not set}"
: "${MAX_MODEL_LEN:?MAX_MODEL_LEN not set}"
: "${MAX_TOKENS:?MAX_TOKENS not set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

# Manifest is 0-indexed by line. sed -n is robust to large files.
SHARD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST_FILE")
if [ -z "$SHARD" ]; then
    echo "ERROR: empty shard at task index $SLURM_ARRAY_TASK_ID in $MANIFEST_FILE"
    exit 1
fi

MAPPER_OUTPUT="$MAPPER_DIR/${SHARD}.jsonl"
INPUT_FILE="$INPUT_DIR/${SHARD}_clean.jsonl"
OUTPUT_FILE="$OUT_DIR/${SHARD}.jsonl"

if [ ! -f "$MAPPER_OUTPUT" ]; then
    echo "ERROR: mapper output not found: $MAPPER_OUTPUT"
    exit 1
fi
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: source not found: $INPUT_FILE"
    exit 1
fi

# Late re-check: if another task or a previous run completed this shard
# between launcher submit and worker start, skip rather than overwrite.
if [ -f "$OUTPUT_FILE" ] && [ -f "$OUT_DIR/${SHARD}.summary.json" ]; then
    echo "[task ${SLURM_ARRAY_TASK_ID}] $SHARD already complete, skipping"
    exit 0
fi

echo "[task ${SLURM_ARRAY_TASK_ID}] $SHARD: $MAPPER_OUTPUT + $INPUT_FILE -> $OUTPUT_FILE"
echo "[task ${SLURM_ARRAY_TASK_ID}] $SHARD start: $(date -Iseconds)"
START_EPOCH=$(date +%s)

# Print start/end/elapsed even if runner.py exits non-zero — `set -e`
# would otherwise skip the elapsed line on failure, hiding wall-clock
# data for failed shards (which we still want for capacity planning).
trap '_end=$(date +%s); echo "[task ${SLURM_ARRAY_TASK_ID}] $SHARD end:   $(date -Iseconds)"; echo "[task ${SLURM_ARRAY_TASK_ID}] $SHARD elapsed: $((_end - START_EPOCH))s ($(( (_end - START_EPOCH) / 60 ))m $(( (_end - START_EPOCH) % 60 ))s)"' EXIT

# Deterministic vLLM output — same flag the mapper uses. Attention
# backend is pinned to TRITON_ATTN inside LLMGrader._init_llm
# (vLLM 0.19.0 + VLLM_BATCH_INVARIANT=1 rejects None on Qwen2-arch;
# FA2's PTX is sm_90-only so it crashes on B200).
export VLLM_BATCH_INVARIANT=1

PYTHONPATH=src .venv/bin/python src/mapping/grading/runner.py \
    --mapper-output "$MAPPER_OUTPUT" \
    --output "$OUTPUT_FILE" \
    --model "$MODEL_PATH" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-tokens "$MAX_TOKENS" \
    --dataset djnw \
    --input-file "$INPUT_FILE"
