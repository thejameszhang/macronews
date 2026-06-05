#!/bin/bash
# SLURM launcher for the KG fact grader (QwQ-32B) on Yale Bouchet.
# Grades the facts in a KG extractor sidecar; writes a per-fact verdict sidecar.
#
# Usage (dev baseline, djnw March 2022):
#   KG_OUTPUT=results/kg/dev/march_2022_dev.v2.4.jsonl \
#   OUTPUT_FILE=results/kg/dev/march_2022_dev.v2.4.grader.jsonl \
#   DATASET_ARGS="--dataset djnw --data-dir /nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles --start-date 2022-03 --end-date 2022-03" \
#     bash scripts/run_kg_grader.sh
#
# Env knobs: MODEL_PATH, MAX_MODEL_LEN, MAX_TOKENS, PARTITION, ACCOUNT, GRES, WALLTIME
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs results

: "${KG_OUTPUT:?set KG_OUTPUT (extractor sidecar to grade)}"
: "${OUTPUT_FILE:?set OUTPUT_FILE (grader sidecar to write)}"
: "${DATASET_ARGS:?set DATASET_ARGS (e.g. --dataset gold --sample-dir data/articles_sample)}"
MODEL_PATH="${MODEL_PATH:-/nfs/roberts/scratch/pi_btk22/jyz32/qwq-32b}"
# 40960 (NOT the mapper-grader's 8192 default): the KG grader shows the FULL
# article, so a long article + system prompt + max_tokens needs the bigger
# context. This matches run_grader.sh's production MAX_MODEL_LEN.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
PARTITION="${PARTITION:-priority_gpu}"
ACCOUNT="${ACCOUNT:-prio_btk22}"
GRES="${GRES:-gpu:b200:1}"
WALLTIME="${WALLTIME:-04:00:00}"
EXCLUDE_NODES="${EXCLUDE_NODES-a1117u29n01}"
EXCLUDE_FLAG=""; [ -n "$EXCLUDE_NODES" ] && EXCLUDE_FLAG="--exclude=${EXCLUDE_NODES}"

if [ ! -d "$MODEL_PATH" ]; then echo "ERROR: model not found: $MODEL_PATH"; exit 1; fi
if [ ! -f "$KG_OUTPUT" ]; then echo "ERROR: KG_OUTPUT not found: $KG_OUTPUT"; exit 1; fi

# Mirrors scripts/run_grader.sh exactly: ${VAR} expands in THIS (outer) shell;
# only \$SLURM_SUBMIT_DIR is escaped (SLURM injects it into the job shell).
# --mem=128G because the djnw reload loads the whole month (~116k articles)
# into memory before the GPU is touched; default mem OOM-kills.
jobid=$(sbatch --parsable \
    --job-name="kg_grader" \
    --output="logs/kg_grader_%j.out" \
    --error="logs/kg_grader_%j.err" \
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
        # module load brings libpython onto LD_LIBRARY_PATH; call .venv/bin/python
        # by ABSOLUTE PATH (do NOT 'source activate' + plain 'python' — that
        # resolves to the module's interpreter, which lacks pydantic/vllm). Same
        # pattern as scripts/_run_grader_task.sh.
        # TRITON_ATTN is pinned inside LLMKGGrader._init_llm (FA2 is sm_90-only,
        # crashes on B200/sm_100; vLLM 0.19.0 + batch-invariant rejects None).
        PYTHONPATH=src .venv/bin/python src/kg/grading/runner.py \
            --kg-output ${KG_OUTPUT} \
            --output ${OUTPUT_FILE} \
            --model ${MODEL_PATH} \
            --max-model-len ${MAX_MODEL_LEN} \
            --max-tokens ${MAX_TOKENS} \
            ${DATASET_ARGS}
    ")
echo "Submitted KG grader -> job ${jobid}: ${KG_OUTPUT} -> ${OUTPUT_FILE}"
echo "Walltime: ${WALLTIME}, partition: ${PARTITION}, gres: ${GRES}"
echo "Monitor: squeue -j ${jobid}   Log: logs/kg_grader_${jobid}.out"
