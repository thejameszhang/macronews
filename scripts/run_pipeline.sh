#!/bin/bash
#
# SLURM launcher for the macronews tagging pipeline on Yale Bouchet.
#
# Four-stage pipeline on a dataset sample:
#   Stage 0: summarize + company-specific filter
#   Stage 1: article-level → (asset, signal, relevance_score, reasoning)
#   Stage 2: paragraph-level with macro_summary context → same shape
#   Stage 3: validate each (article, asset) pair + select text for embedding
#   Final:   validated union of Stage 1 and Stage 2 mappings
#
# Usage:
#   export MODEL_PATH=$HOME/models/gemma-4-26b-a4b-it   # optional override
#   bash scripts/run_pipeline.sh [extra args passed to pipeline.py]

set -euo pipefail
cd /nfs/roberts/project/pi_btk22/jyz32/macronews
mkdir -p logs results

MODEL_PATH="${MODEL_PATH:-/nfs/roberts/scratch/pi_btk22/jyz32/gemma-4-26b-a4b-it}"
DATASET="${DATASET:-djnw}"
MAX_ARTICLES="${MAX_ARTICLES:-100}"
START_DATE="${START_DATE:-2020-11}"
END_DATE="${END_DATE:-2020-11}"
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
        # Deterministic vLLM output: same input + same seed -> bit-identical
        # outputs regardless of batch size/composition. Requires compute
        # capability >= 9.0 (H100/H200/B100/B200/rtx_pro_6000_blackwell).
        export VLLM_BATCH_INVARIANT=1
        PYTHONPATH=src python src/experiments/pipeline.py \
            --model ${MODEL_PATH} \
            --max-model-len 65536 \
            --dataset ${DATASET} \
            --max-articles ${MAX_ARTICLES} \
            --start-date ${START_DATE} \
            --end-date ${END_DATE} \
            ${RANDOM_SEED:+--random-seed ${RANDOM_SEED}} \
            ${EXTRA_ARGS}
    ")

echo "Submitted pipeline run → job ${jobid}"
echo "Dataset: ${DATASET}, max_articles: ${MAX_ARTICLES}, model: ${MODEL_PATH}"
echo "Monitor: squeue -j ${jobid}"
echo "Output:  logs/pipeline_${jobid}.out"
