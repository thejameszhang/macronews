#!/bin/bash
#
# SLURM launcher for the macronews ArticleMapper on Yale Bouchet.
#
# Runs src/pipeline.py in either DEV (sampling/date-range) or PROD (single
# *_clean.jsonl shard) mode. Output is JSONL with a sidecar .summary.json.
#
# DEV usage:
#   bash scripts/run_pipeline.sh                              # gold dataset, default settings
#   MAX_ARTICLES=1000 START_DATE=2022-03 END_DATE=2022-03 \
#     RANDOM_SEED=42 DATASET=djnw bash scripts/run_pipeline.sh
#
# PROD usage:
#   MODE=prod \
#     INPUT_FILE=/path/to/2015-06a_clean.jsonl \
#     OUTPUT_DIR=results/prod/run-1 \
#     bash scripts/run_pipeline.sh
#
# Env knobs (all modes):
#   MODEL_PATH, MAX_MODEL_LEN, PARTITION, ACCOUNT, GRES, WALLTIME
# DEV-only:
#   DATASET, MAX_ARTICLES, START_DATE, END_DATE, RANDOM_SEED
# PROD-only:
#   INPUT_FILE, OUTPUT_DIR
# Extra args after the script are passed through to pipeline.py.

set -euo pipefail
cd /nfs/roberts/project/pi_btk22/jyz32/macronews
mkdir -p logs results

MODE="${MODE:-dev}"
MODEL_PATH="${MODEL_PATH:-/nfs/roberts/scratch/pi_btk22/jyz32/gemma-4-26b-a4b-it}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
PARTITION="${PARTITION:-priority_gpu}"
ACCOUNT="${ACCOUNT:-prio_btk22}"
GRES="${GRES:-gpu:b200:1}"
# a1117u29n01 is ~3.4x slower than peer B200 nodes (observed on 2014 prod run:
# 6h35m / 6h48m on this node vs ~2h on others). Override with EXCLUDE_NODES=
# (empty) to allow it back in if Yale fixes it.
EXCLUDE_NODES="${EXCLUDE_NODES-a1117u29n01}"
EXTRA_ARGS="${*}"

# DEV defaults (only used when MODE=dev)
DATASET="${DATASET:-djnw}"
MAX_ARTICLES="${MAX_ARTICLES:-100}"
START_DATE="${START_DATE-}"
END_DATE="${END_DATE-}"
RANDOM_SEED="${RANDOM_SEED:-}"

# PROD inputs
INPUT_FILE="${INPUT_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# Walltime: dev runs are minutes, prod runs can be hours on fat shards.
# Over-requesting costs nothing; under-requesting forfeits the run on timeout.
if [ "$MODE" = "prod" ]; then
    WALLTIME="${WALLTIME:-24:00:00}"
else
    WALLTIME="${WALLTIME:-08:00:00}"
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

if [ "$MODE" = "prod" ]; then
    if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_DIR" ]; then
        echo "ERROR: MODE=prod requires INPUT_FILE and OUTPUT_DIR"
        exit 1
    fi
    if [ ! -f "$INPUT_FILE" ]; then
        echo "ERROR: INPUT_FILE not found: $INPUT_FILE"
        exit 1
    fi
    PIPELINE_ARGS="--mode prod --input-file ${INPUT_FILE} --output-dir ${OUTPUT_DIR}"
    JOB_DESC="prod shard: $(basename ${INPUT_FILE}) -> ${OUTPUT_DIR}"
elif [ "$MODE" = "dev" ]; then
    PIPELINE_ARGS="--mode dev --dataset ${DATASET} --max-articles ${MAX_ARTICLES}"
    [ -n "$START_DATE" ]  && PIPELINE_ARGS="$PIPELINE_ARGS --start-date ${START_DATE}"
    [ -n "$END_DATE" ]    && PIPELINE_ARGS="$PIPELINE_ARGS --end-date ${END_DATE}"
    [ -n "$RANDOM_SEED" ] && PIPELINE_ARGS="$PIPELINE_ARGS --random-seed ${RANDOM_SEED}"
    JOB_DESC="dev: dataset=${DATASET}, max_articles=${MAX_ARTICLES}"
else
    echo "ERROR: MODE must be 'dev' or 'prod' (got: ${MODE})"
    exit 1
fi

EXCLUDE_FLAG=""
if [ -n "$EXCLUDE_NODES" ]; then
    EXCLUDE_FLAG="--exclude=${EXCLUDE_NODES}"
fi

jobid=$(sbatch --parsable \
    --job-name="pipeline" \
    --output="logs/pipeline_%j.out" \
    --error="logs/pipeline_%j.err" \
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
        # Deterministic vLLM output: ~0.09% tag drift on DJNW 1000 articles
        # vs ~29% without the flag. Requires compute capability >= 9.0.
        export VLLM_BATCH_INVARIANT=1
        PYTHONPATH=src python src/pipeline.py \
            --model ${MODEL_PATH} \
            --max-model-len ${MAX_MODEL_LEN} \
            ${PIPELINE_ARGS} \
            ${EXTRA_ARGS}
    ")

echo "Submitted pipeline run → job ${jobid}"
echo "${JOB_DESC}"
echo "Walltime: ${WALLTIME}, partition: ${PARTITION}, gres: ${GRES}"
echo "Monitor: squeue -j ${jobid}"
echo "Output:  logs/pipeline_${jobid}.out"
