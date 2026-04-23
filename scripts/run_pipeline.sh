#!/bin/bash
#
# SLURM launcher for the macronews ArticleMapper on Yale Bouchet.
#
# Runs src/pipeline.py: one LLM call per (article, asset) pair returning
# per-asset relevance, evidence paragraphs, signal ("strong"|"weak"), and
# relevance_score in [0,1]. Asset-class-specific rules are injected into
# the system prompt per class. Results are saved to results/<dataset>_summary.json.
#
# Common usage:
#   bash scripts/run_pipeline.sh                              # gold dataset, default settings
#   MAX_ARTICLES=1000 START_DATE=2022-03 END_DATE=2022-03 \
#     RANDOM_SEED=42 DATASET=djnw bash scripts/run_pipeline.sh
#
# Env knobs: MODEL_PATH, MAX_MODEL_LEN, DATASET, MAX_ARTICLES, START_DATE,
#            END_DATE, RANDOM_SEED, PARTITION, ACCOUNT, GRES.
# Extra args after the script are passed through to pipeline.py.

set -euo pipefail
cd /nfs/roberts/project/pi_btk22/jyz32/macronews
mkdir -p logs results

MODEL_PATH="${MODEL_PATH:-/nfs/roberts/scratch/pi_btk22/jyz32/gemma-4-26b-a4b-it}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
DATASET="${DATASET:-djnw}"
MAX_ARTICLES="${MAX_ARTICLES:-100}"
START_DATE="${START_DATE-}"
END_DATE="${END_DATE-}"
RANDOM_SEED="${RANDOM_SEED:-}"
PARTITION="${PARTITION:-priority_gpu}"
ACCOUNT="${ACCOUNT:-prio_btk22}"
GRES="${GRES:-gpu:h200:1}"
EXTRA_ARGS="${*}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

jobid=$(sbatch --parsable \
    --job-name="pipeline" \
    --output="logs/pipeline_%j.out" \
    --error="logs/pipeline_%j.err" \
    --time=08:00:00 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=128G \
    --partition=${PARTITION} \
    --account=${ACCOUNT} \
    --gres=${GRES} \
    --wrap="
        module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null || true
        cd /nfs/roberts/project/pi_btk22/jyz32/macronews
        source .venv/bin/activate
        # Deterministic vLLM output: same input + same seed -> near-bit-identical
        # outputs regardless of batch size/composition (measured 0.09% tag drift
        # on DJNW June 2015 1000 articles; without this flag, tag drift jumps
        # to ~29%). Requires compute capability >= 9.0 (H100/H200/B100/B200/
        # rtx_pro_6000_blackwell).
        export VLLM_BATCH_INVARIANT=1
        PYTHONPATH=src python src/pipeline.py \
            --model ${MODEL_PATH} \
            --max-model-len ${MAX_MODEL_LEN} \
            --dataset ${DATASET} \
            --max-articles ${MAX_ARTICLES} \
            ${START_DATE:+--start-date ${START_DATE}} \
            ${END_DATE:+--end-date ${END_DATE}} \
            ${RANDOM_SEED:+--random-seed ${RANDOM_SEED}} \
            ${EXTRA_ARGS}
    ")

echo "Submitted pipeline run → job ${jobid}"
echo "Dataset: ${DATASET}, max_articles: ${MAX_ARTICLES}, model: ${MODEL_PATH}"
echo "Monitor: squeue -j ${jobid}"
echo "Output:  logs/pipeline_${jobid}.out"
