#!/bin/bash
#
# SLURM launcher for the macronews grader on Yale Bouchet.
#
# Reads a mapper-output JSONL + the original source articles, calls
# QwQ-32B (or compatible) to verify each (article, mapping) pair, and
# writes a sidecar JSONL of grader verdicts.
#
# Gold usage:
#   bash scripts/run_grader.sh
#       (defaults to gold dataset, results/groups-v2/gold.jsonl input,
#        results/groups-v2-grader/gold.jsonl output)
#
# DJNW usage:
#   MODE=djnw \
#     MAPPER_OUTPUT=results/prod/groups-v2/2014-05c.jsonl \
#     INPUT_FILE=/nfs/.../articles/2014-05c_clean.jsonl \
#     OUTPUT_FILE=results/prod/groups-v2-grader/2014-05c.jsonl \
#     bash scripts/run_grader.sh
#
# Env knobs:
#   MODEL_PATH, MAX_MODEL_LEN, MAX_TOKENS,
#   PARTITION, ACCOUNT, GRES, WALLTIME

set -euo pipefail
cd /nfs/roberts/project/pi_btk22/jyz32/macronews
mkdir -p logs results

MODE="${MODE:-gold}"
MODEL_PATH="${MODEL_PATH:-/nfs/roberts/scratch/pi_btk22/jyz32/qwq-32b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
PARTITION="${PARTITION:-priority_gpu}"
ACCOUNT="${ACCOUNT:-prio_btk22}"
GRES="${GRES:-gpu:b200:1}"

if [ "$MODE" = "djnw" ]; then
    WALLTIME="${WALLTIME:-12:00:00}"
else
    WALLTIME="${WALLTIME:-04:00:00}"
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "       Submit the download first:"
    echo "         MODEL_REPO=Qwen/QwQ-32B LOCAL_DIR=$MODEL_PATH \\"
    echo "             sbatch scripts/download_llm.sh"
    exit 1
fi

if [ "$MODE" = "gold" ]; then
    MAPPER_OUTPUT="${MAPPER_OUTPUT:-results/groups-v2/gold.jsonl}"
    OUTPUT_FILE="${OUTPUT_FILE:-results/groups-v2-grader/gold.jsonl}"
    SAMPLE_DIR="${SAMPLE_DIR:-data/articles_sample}"
    DATASET_ARGS="--dataset gold --sample-dir ${SAMPLE_DIR}"
    JOB_DESC="gold: ${MAPPER_OUTPUT} -> ${OUTPUT_FILE}"
elif [ "$MODE" = "djnw" ]; then
    : "${MAPPER_OUTPUT:?MODE=djnw requires MAPPER_OUTPUT}"
    : "${OUTPUT_FILE:?MODE=djnw requires OUTPUT_FILE}"
    : "${INPUT_FILE:?MODE=djnw requires INPUT_FILE (matching *_clean.jsonl)}"
    if [ ! -f "$INPUT_FILE" ]; then
        echo "ERROR: INPUT_FILE not found: $INPUT_FILE"
        exit 1
    fi
    if [ ! -f "$MAPPER_OUTPUT" ]; then
        echo "ERROR: MAPPER_OUTPUT not found: $MAPPER_OUTPUT"
        exit 1
    fi
    DATASET_ARGS="--dataset djnw --input-file ${INPUT_FILE}"
    JOB_DESC="djnw shard: $(basename ${MAPPER_OUTPUT}) -> ${OUTPUT_FILE}"
else
    echo "ERROR: MODE must be 'gold' or 'djnw' (got: ${MODE})"
    exit 1
fi

if [ ! -f "$MAPPER_OUTPUT" ]; then
    echo "ERROR: MAPPER_OUTPUT not found: $MAPPER_OUTPUT"
    exit 1
fi

# a1117u29n01 is ~3.4x slower than peer B200 nodes (observed on 2014 prod run:
# 6h35m / 6h48m on this node vs ~2h on others). Override with EXCLUDE_NODES=
# (empty) to allow it back in if Yale fixes it.
EXCLUDE_NODES="${EXCLUDE_NODES-a1117u29n01}"
EXCLUDE_FLAG=""
if [ -n "$EXCLUDE_NODES" ]; then
    EXCLUDE_FLAG="--exclude=${EXCLUDE_NODES}"
fi

jobid=$(sbatch --parsable \
    --job-name="grader" \
    --output="logs/grader_%j.out" \
    --error="logs/grader_%j.err" \
    --time=${WALLTIME} \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=128G \
    --partition=${PARTITION} \
    --account=${ACCOUNT} \
    --gres=${GRES} \
    ${EXCLUDE_FLAG} \
    --wrap="
        module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null || true
        cd /nfs/roberts/project/pi_btk22/jyz32/macronews
        source .venv/bin/activate
        export VLLM_BATCH_INVARIANT=1
        # Attention backend is pinned to TRITON_ATTN inside LLMGrader._init_llm
        # (vLLM 0.19.0 + VLLM_BATCH_INVARIANT=1 rejects None on Qwen2-arch;
        # FA2's PTX is sm_90-only so it crashes on B200).
        PYTHONPATH=src python src/grading/runner.py \
            --mapper-output ${MAPPER_OUTPUT} \
            --output ${OUTPUT_FILE} \
            --model ${MODEL_PATH} \
            --max-model-len ${MAX_MODEL_LEN} \
            --max-tokens ${MAX_TOKENS} \
            ${DATASET_ARGS}
    ")

echo "Submitted grader run -> job ${jobid}"
echo "${JOB_DESC}"
echo "Walltime: ${WALLTIME}, partition: ${PARTITION}, gres: ${GRES}"
echo "Monitor: squeue -j ${jobid}"
echo "Output:  logs/grader_${jobid}.out"
