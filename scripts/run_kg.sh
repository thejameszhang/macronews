#!/bin/bash
#
# SLURM launcher for the macronews KG fact extractor on Yale Bouchet.
#
# Reads gold articles or DJNW *_clean.jsonl shards, calls Gemma 4 to extract
# entities + (s, r, o) facts per article, and writes a sidecar JSONL of
# KGArticleResult rows (+ summary.json).
#
# Gold usage:
#   bash scripts/run_kg.sh
#       (defaults to gold dataset -> results/kg/dev/gold.jsonl)
#
# DJNW single-shard usage:
#   MODE=djnw \
#     INPUT_FILE=/nfs/.../articles/2014-05c_clean.jsonl \
#     OUTPUT_FILE=results/kg/dev/2014-05c.jsonl \
#     bash scripts/run_kg.sh
#
# DJNW multi-shard with date range + deterministic sampling (mirrors mapper DEV):
#   MODE=djnw \
#     START_DATE=2022-03 END_DATE=2022-03 \
#     MAX_ARTICLES=1000 RANDOM_SEED=42 \
#     OUTPUT_FILE=results/kg/dev/march_2022_dev.jsonl \
#     bash scripts/run_kg.sh
#
# Env knobs:
#   MODEL_PATH, MAX_MODEL_LEN, MAX_TOKENS,
#   PARTITION, ACCOUNT, GRES, WALLTIME, EXCLUDE_NODES
# DJNW-only:
#   INPUT_FILE  (single-shard mode), OR
#   DATA_DIR, START_DATE, END_DATE, MAX_ARTICLES, RANDOM_SEED  (multi-shard mode)

set -euo pipefail
# Resolve repo root from script location so the launcher works regardless
# of which clone or which user runs it.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs results/kg/dev

MODE="${MODE:-gold}"
MODEL_PATH="${MODEL_PATH:-/nfs/roberts/scratch/pi_btk22/jyz32/gemma-4-26b-a4b-it}"
# Match mapper's default — keeps both pipelines on the same context window
# so the token-length filter at load time produces the same article pool.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
PARTITION="${PARTITION:-priority_gpu}"
ACCOUNT="${ACCOUNT:-prio_btk22}"
GRES="${GRES:-gpu:b200:1}"

if [ "$MODE" = "djnw" ]; then
    WALLTIME="${WALLTIME:-12:00:00}"
else
    WALLTIME="${WALLTIME:-02:00:00}"
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

if [ "$MODE" = "gold" ]; then
    OUTPUT_FILE="${OUTPUT_FILE:-results/kg/dev/gold.jsonl}"
    SAMPLE_DIR="${SAMPLE_DIR:-data/articles_sample}"
    DATASET_ARGS="--dataset gold --sample-dir ${SAMPLE_DIR}"
    JOB_DESC="gold -> ${OUTPUT_FILE}"
elif [ "$MODE" = "djnw" ]; then
    : "${OUTPUT_FILE:?MODE=djnw requires OUTPUT_FILE}"
    # Default DATA_DIR matches mapper's DEV default (v2 cleaning).
    DATA_DIR="${DATA_DIR:-/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles}"
    INPUT_FILE="${INPUT_FILE:-}"
    START_DATE="${START_DATE:-}"
    END_DATE="${END_DATE:-}"
    MAX_ARTICLES="${MAX_ARTICLES:-}"
    RANDOM_SEED="${RANDOM_SEED:-}"

    if [ -n "$INPUT_FILE" ]; then
        if [ ! -f "$INPUT_FILE" ]; then
            echo "ERROR: INPUT_FILE not found: $INPUT_FILE"
            exit 1
        fi
        DATASET_ARGS="--dataset djnw --input-file ${INPUT_FILE}"
        JOB_DESC="djnw single shard: $(basename ${INPUT_FILE}) -> ${OUTPUT_FILE}"
    else
        # Multi-shard / date-range / sampling mode.
        if [ ! -d "$DATA_DIR" ]; then
            echo "ERROR: DATA_DIR not found: $DATA_DIR"
            exit 1
        fi
        DATASET_ARGS="--dataset djnw --data-dir ${DATA_DIR}"
        [ -n "$START_DATE" ]   && DATASET_ARGS="${DATASET_ARGS} --start-date ${START_DATE}"
        [ -n "$END_DATE" ]     && DATASET_ARGS="${DATASET_ARGS} --end-date ${END_DATE}"
        [ -n "$MAX_ARTICLES" ] && DATASET_ARGS="${DATASET_ARGS} --max-articles ${MAX_ARTICLES}"
        [ -n "$RANDOM_SEED" ]  && DATASET_ARGS="${DATASET_ARGS} --random-seed ${RANDOM_SEED}"
        JOB_DESC="djnw range: ${START_DATE:-*}..${END_DATE:-*}"
        [ -n "$MAX_ARTICLES" ] && JOB_DESC="${JOB_DESC} (max ${MAX_ARTICLES}"
        [ -n "$RANDOM_SEED" ]  && JOB_DESC="${JOB_DESC}, seed ${RANDOM_SEED}"
        [ -n "$MAX_ARTICLES" ] && JOB_DESC="${JOB_DESC})"
        JOB_DESC="${JOB_DESC} -> ${OUTPUT_FILE}"
    fi
else
    echo "ERROR: MODE must be 'gold' or 'djnw' (got: ${MODE})"
    exit 1
fi

# a1117u29n01 is ~3.4x slower than peer B200 nodes; auto-exclude.
EXCLUDE_NODES="${EXCLUDE_NODES-a1117u29n01}"
EXCLUDE_FLAG=""
if [ -n "$EXCLUDE_NODES" ]; then
    EXCLUDE_FLAG="--exclude=${EXCLUDE_NODES}"
fi

jobid=$(sbatch --parsable \
    --job-name="kg_extract" \
    --output="logs/kg_%j.out" \
    --error="logs/kg_%j.err" \
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
        cd \$SLURM_SUBMIT_DIR
        export VLLM_BATCH_INVARIANT=1
        # Invoke the venv's python by absolute path. \`source .venv/bin/activate\`
        # alone is insufficient on this cluster: the Python module is auto-loaded
        # and re-prepends its bin/, so plain \`python\` resolves to the system
        # interpreter (which lacks pydantic/vllm). Module load is still needed
        # for libpython3.12.so.1.0 in LD_LIBRARY_PATH.
        # Gemma 4 uses vLLM's default attention backend (same as mapper);
        # no AttentionConfig override needed (TRITON_ATTN is Qwen2-only).
        PYTHONPATH=src .venv/bin/python src/kg/runner.py \
            --output ${OUTPUT_FILE} \
            --model ${MODEL_PATH} \
            --max-model-len ${MAX_MODEL_LEN} \
            --max-tokens ${MAX_TOKENS} \
            ${DATASET_ARGS}
    ")

echo "Submitted KG extraction run -> job ${jobid}"
echo "${JOB_DESC}"
echo "Walltime: ${WALLTIME}, partition: ${PARTITION}, gres: ${GRES}"
echo "Monitor: squeue -j ${jobid}"
echo "Output:  logs/kg_${jobid}.out"
