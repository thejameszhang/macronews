#!/bin/bash
# Build the macro knowledge graph from the Temporal Agent's extracted statements.
#
# This is the post-extraction assembly chain. Extraction itself (the Temporal
# Agent, a multi-shard GPU job) runs separately via scripts/run_kg.sh and writes
# $MONTH.extracted.jsonl; this script takes that plus the asset-group mapper output
# and runs the remaining five stages to produce the final graph and visualisation.
#
#   input : results/kg/prod/$MONTH.extracted.jsonl   (Temporal Agent output)
#           results/kg/prod/$MONTH.mapper.jsonl       (asset-group mapper output)
#   stages: type-gate -> disambiguate (+self-reference filter) -> invalidate (QwQ)
#           -> link asset groups -> visualize
#   output: $MONTH.facts.jsonl (the graph), entity_groups/group_members JSONs,
#           and $MONTH.graph.html (interactive viz + Fact Reader)
#
# One free-B200 job: only invalidation needs the GPU; the rest is CPU/embedding.
set -euo pipefail
module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null || true
cd /nfs/roberts/project/pi_btk22/jyz32/macronews
export VLLM_BATCH_INVARIANT=1

MONTH=2014-05
R=results/kg/prod
QWQ=/nfs/roberts/scratch/pi_btk22/jyz32/qwq-32b

echo "[1/5] type_gate $(date -Is)"
PYTHONPATH=src .venv/bin/python -m kg.type_gate \
    "$R/$MONTH.extracted.jsonl" \
    --clean "$R/$MONTH.typed.jsonl" \
    --rejected "$R/$MONTH.type_rejected.jsonl" --summary

echo "[2/5] disambiguate $(date -Is)"
PYTHONPATH=src .venv/bin/python -m src.kg.disambiguate \
    "$R/$MONTH.typed.jsonl" \
    --output "$R/$MONTH.disambiguated.jsonl" \
    --clusters "$R/$MONTH.entity_clusters.json"

echo "[3/5] invalidate (QwQ) $(date -Is)"
PYTHONPATH=src .venv/bin/python -m kg.invalidate_llm \
    --disambig "$R/$MONTH.disambiguated.jsonl" \
    --out "$R/$MONTH.facts.jsonl" \
    --model "$QWQ"

echo "[4/5] link_groups $(date -Is)"
PYTHONPATH=src .venv/bin/python -m kg.link_groups \
    "$R/$MONTH.facts.jsonl" \
    --mapper "$R/$MONTH.mapper.jsonl" \
    --output "$R/$MONTH.entity_groups.json" \
    --residual "$R/$MONTH.unconfirmed_links.jsonl" \
    --group-members "$R/$MONTH.group_members.json" \
    --no-llm

echo "[5/5] visualize $(date -Is)"
PYTHONPATH=src .venv/bin/python -m kg.visualize \
    "$R/$MONTH.facts.jsonl" \
    --output "$R/$MONTH.graph.html" \
    --entity-groups "$R/$MONTH.entity_groups.json" \
    --shards ${MONTH}a ${MONTH}b ${MONTH}c \
    --title "Macro KG (May 2014, final)"
echo "[done] $(date -Is)"
