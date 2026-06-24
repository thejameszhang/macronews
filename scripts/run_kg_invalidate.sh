#!/bin/bash
# SLURM launcher for the KG LLM Invalidation Agent (QwQ-32B) on Yale Bouchet.
# Embeds statements, selects candidate pairs via entity-share + temporal-overlap +
# embedding top-K, and judges chronological pairs for supersession; writes a
# non-lossy *.invalidated.jsonl (+ .summary.json).
#
# Usage (gamma 2014-05):
#   DISAMBIG=results/kg/gamma/2014-05.disambig.jsonl \
#   OUTPUT_FILE=results/kg/gamma/2014-05.invalidated.jsonl \
#     bash scripts/run_kg_invalidate.sh
#
# Env knobs: MODEL_PATH, EMB_OUT, MAX_MODEL_LEN, PARTITION, ACCOUNT, QOS,
#            GRES, WALLTIME.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs results

: "${DISAMBIG:?set DISAMBIG (disambiguated event sidecar to read)}"
# Two modes. Set INSPECT_CANDIDATES to a path => no-LLM gate: embed + run the
# candidate funnel and dump per-primary candidates, NO QwQ (read it before judging).
# Otherwise => judge mode: needs OUTPUT_FILE (the *.invalidated.jsonl to write).
INSPECT_CANDIDATES="${INSPECT_CANDIDATES:-}"
MODEL_PATH="${MODEL_PATH:-/nfs/roberts/scratch/pi_btk22/jyz32/qwq-32b}"
EMB_OUT="${EMB_OUT:-}"
# Option B: set ASSET_GROUPS to a link_groups entity_groups.json to scope invalidation
# to facts whose entities belong to an asset group. Omit = broad (cookbook) mode.
ASSET_GROUPS="${ASSET_GROUPS:-}"
# 8192 (NOT the grader's 40960): each prompt is just two short statements + the
# fixed guidelines — no full article — so a small context is plenty.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
# FREE B200 tier (gpu_b200 / pi_btk22 / QOS normal). NEVER priority_gpu/prio_btk22
# (that bills the lab's paid allocation). Override only with explicit permission.
PARTITION="${PARTITION:-gpu_b200}"
ACCOUNT="${ACCOUNT:-pi_btk22}"
QOS="${QOS:-normal}"
GRES="${GRES:-gpu:b200:1}"
WALLTIME="${WALLTIME:-08:00:00}"
EXCLUDE_NODES="${EXCLUDE_NODES-a1117u29n01}"   # slow node, auto-excluded
EXCLUDE_FLAG=""; [ -n "$EXCLUDE_NODES" ] && EXCLUDE_FLAG="--exclude=${EXCLUDE_NODES}"
EMB_OUT_FLAG=""; [ -n "$EMB_OUT" ] && EMB_OUT_FLAG="--emb-out ${EMB_OUT}"
ASSET_FLAG=""; [ -n "$ASSET_GROUPS" ] && ASSET_FLAG="--asset-groups ${ASSET_GROUPS}"

if [ ! -f "$DISAMBIG" ]; then echo "ERROR: DISAMBIG not found: $DISAMBIG"; exit 1; fi
if [ -n "$INSPECT_CANDIDATES" ]; then
    PY_MODE_ARGS="--inspect-candidates ${INSPECT_CANDIDATES}"   # no model needed
    JOB_DESC="${DISAMBIG} -> ${INSPECT_CANDIDATES} (inspect, no LLM)"
else
    : "${OUTPUT_FILE:?set OUTPUT_FILE (*.invalidated.jsonl) or INSPECT_CANDIDATES}"
    if [ ! -d "$MODEL_PATH" ]; then echo "ERROR: model not found: $MODEL_PATH"; exit 1; fi
    PY_MODE_ARGS="--out ${OUTPUT_FILE} --model ${MODEL_PATH}"
    JOB_DESC="${DISAMBIG} -> ${OUTPUT_FILE}"
fi

# ${VAR} expands in THIS (outer) shell; only \$SLURM_SUBMIT_DIR is escaped (SLURM
# injects it into the job shell). --mem=64G: we read the disambig JSONL + hold
# statement embeddings + the entity index in RAM, but do NOT reload the raw
# article corpus, so the grader's 128G is unnecessary.
jobid=$(sbatch --parsable \
    --job-name="kg_invalidate" \
    --output="logs/kg_invalidate_%j.out" \
    --error="logs/kg_invalidate_%j.err" \
    --time=${WALLTIME} \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=64G \
    --partition=${PARTITION} \
    --account=${ACCOUNT} \
    --qos=${QOS} \
    --gres=${GRES} \
    ${EXCLUDE_FLAG} \
    --wrap="
        module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null || true
        cd \$SLURM_SUBMIT_DIR
        export VLLM_BATCH_INVARIANT=1
        # module load brings libpython onto LD_LIBRARY_PATH; call .venv/bin/python
        # by ABSOLUTE PATH (do NOT 'source activate' + plain 'python'). TRITON_ATTN
        # is pinned inside InvalidationAgent._init_llm (FA2 is sm_90-only, crashes
        # on B200/sm_100).
        PYTHONPATH=src .venv/bin/python -m kg.invalidate_llm \
            --disambig ${DISAMBIG} \
            ${PY_MODE_ARGS} \
            ${ASSET_FLAG} \
            ${EMB_OUT_FLAG}
    ")
echo "Submitted KG invalidation agent -> job ${jobid}: ${JOB_DESC}"
echo "Walltime: ${WALLTIME}, partition: ${PARTITION} (FREE tier), gres: ${GRES}"
echo "Monitor: squeue -j ${jobid}   Log: logs/kg_invalidate_${jobid}.out"
