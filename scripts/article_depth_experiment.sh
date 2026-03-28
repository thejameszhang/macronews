#!/bin/bash
#
# SLURM launcher for the article-depth experiment.
#
# Two-stage pipeline on 10 sample articles:
#   Stage 1: article-level → themes, regions, assets, macro_summary
#   Stage 2: paragraph-level with macro_summary context → assets
#   Final:   union of Stage 1 and Stage 2 assets
#
# Usage:
#   scripts/article_depth_experiment.sh

set -euo pipefail
cd /gpfs/home/jyz32/macronews
mkdir -p logs

MODEL_PATH="$HOME/models/Qwen2.5-72B-Instruct-AWQ"
EXTRA_ARGS="${*}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Download with: huggingface-cli download Qwen/Qwen2.5-72B-Instruct-AWQ --local-dir $MODEL_PATH"
    exit 1
fi

jobid=$(sbatch --parsable \
    --job-name="article_depth" \
    --output="logs/article_depth_%j.out" \
    --error="logs/article_depth_%j.err" \
    --time=02:00:00 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --mem=64G \
    --partition=gpunormal,h100 \
    --gres=gpu:1 \
    --nodelist=c014,c015,c016,c017,c022 \
    --wrap="cd /gpfs/home/jyz32/macronews && source .venv/bin/activate && \
PYTHONPATH=src uv run python src/experiments/article_depth.py --model ${MODEL_PATH} --max-model-len 8192 ${EXTRA_ARGS}")

echo "Submitted article_depth experiment → job ${jobid}"
echo "Monitor with: squeue -u $USER"
echo "Output: logs/article_depth_${jobid}.out"
