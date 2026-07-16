#!/bin/bash
# Dispatch `macronews kg disambiguate` to a FREE-tier B200 (gpu_b200/pi_btk22).
# disambiguate.py auto-detects CUDA (GPU embed ~60-100x faster than CPU); this
# wrapper just hands it a GPU + RAM. NOT the billing priority tier.
#
#   INPUT=results/kg/temporal/2014-05.jsonl \
#   OUTPUT=results/kg/temporal/2014-05.disambig.jsonl \
#   CLUSTERS=results/kg/temporal/2014-05.clusters.json \
#   bash slurm/run_disambig.sh
#
# Optional env: THRESHOLD, SELF_REF_THRESHOLD, MODEL_PATH, GPU_PARTITION, GPU_ACCOUNT, GPU_GRES, WALLTIME, MEM,
#               EXCLUDE_NODES.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
: "${INPUT:?set INPUT (disambig input JSONL, e.g. a concatenated multi-shard KG file)}"
: "${OUTPUT:?set OUTPUT (disambiguated JSONL out)}"
CLUSTERS="${CLUSTERS:-}"
THRESHOLD="${THRESHOLD:-}"
WALLTIME="${WALLTIME:-01:00:00}"; MEM="${MEM:-64G}"
[ -f "$INPUT" ] || { echo "ERROR: INPUT not found: $INPUT"; exit 1; }
CLUSTERS_FLAG=""; [ -n "$CLUSTERS" ] && CLUSTERS_FLAG="--clusters ${CLUSTERS}"
THRESH_FLAG="";   [ -n "$THRESHOLD" ] && THRESH_FLAG="--threshold ${THRESHOLD}"
MODEL_FLAG="";    [ -n "${MODEL_PATH:-}" ] && MODEL_FLAG="--model ${MODEL_PATH}"
mkdir -p logs
jobid=$(sbatch --parsable \
    --job-name=disambig --time=${WALLTIME} \
    --partition=${GPU_PARTITION} --account=${GPU_ACCOUNT} --gres=${GPU_GRES} \
    --cpus-per-task=8 --mem=${MEM} ${EXCLUDE_FLAG} \
    --output=logs/disambig_%j.out --error=logs/disambig_%j.err \
    --wrap "module load Python/3.12.3-GCCcore-13.3.0 && \
        $PY -m macronews.cli kg disambiguate ${INPUT} \
            --output ${OUTPUT} ${CLUSTERS_FLAG} ${THRESH_FLAG} \
            ${SELF_REF_THRESHOLD:+--self-ref-threshold ${SELF_REF_THRESHOLD}} \
            ${MODEL_FLAG}")
echo "Submitted disambig job ${jobid} (partition ${GPU_PARTITION}, account ${GPU_ACCOUNT}, gres ${GPU_GRES})"
echo "Monitor: squeue -j ${jobid}  |  logs/disambig_${jobid}.out / .err (embed progress)"
