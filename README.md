# MacroNews

NLP pipeline for predicting global macro futures returns from financial news.

Most text-as-data finance research targets equities, where company names anchor every article to a ticker. **Macro futures have no such anchors.** This project solves that mapping problem with LLMs, then feeds the asset-attributed text into downstream return regressions.

## How it works

```
  DJNW articles
       │
       ▼
  ┌──────────┐
  │  Filters │   drop tables, press releases, insider filings,
  └─────┬────┘   lifestyle pieces, procedural templates, etc.
        ▼
  ┌──────────┐
  │  Mapper  │   Gemma 4 26B-A4B: one call per (article, asset-group).
  └─────┬────┘   Outputs evidence paragraphs + relevance score per group.
        ▼
  asset-group-attributed text  →  downstream return regression
```

Asset universe: ~50 groups spanning global equities, FX, rates, commodities, and US equity sectors. Defined in [`src/config/group_universe.yaml`](src/config/group_universe.yaml).

## Running

```bash
# Activate the env (do NOT use `uv run` — see Setup)
source .venv/bin/activate

# Single DJNW shard, production mode
MODE=prod \
  INPUT_FILE=/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles/2014-05c_clean.jsonl \
  OUTPUT_DIR=results/prod/v1 \
  bash scripts/run_pipeline.sh

# Many shards in parallel via SLURM array
OUT_DIR=results/prod/v1 PATTERN='2014-*' \
  bash scripts/run_pipeline_array.sh
```

Outputs are JSONL (one record per article) with sidecar `.summary.json` aggregates.

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and a Hopper-or-newer GPU (B200 or H200).

```bash
# 1. Install project dependencies
uv sync

# 2. Install vLLM nightly + transformers 5.x for Gemma 4 support.
#    Adapted from the vLLM Gemma 4 recipe:
#      https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
source .venv/bin/activate
uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
uv pip install 'transformers==5.5.0'
```

> Do not use `uv run` — it re-syncs the env and breaks the fragile vLLM / transformers pin.

Then create a `.env` for credentials:

```bash
cp .env.example .env
# Edit .env to fill in HF_TOKEN (needed by scripts/download_llm.sh to pull model weights).
```

## Data

- **Articles**: Dow Jones Newswires (1996–2025), cleaned JSONL shards at `/nfs/roberts/.../v2/articles/`.
- **Returns**: `datasets/sync_daily.csv` (95 macro futures).
- **Asset universe**: [`universe.xlsx`](universe.xlsx) — canonical Tier 1 / Tier 2 / Groups reference (human-readable).
- **Group config**: [`src/config/group_universe.yaml`](src/config/group_universe.yaml) — the live configuration the mapper reads at runtime.

## Knowledge Graph

A parallel pipeline ([`src/kg/`](src/kg/)) extracts a **macroeconomic knowledge graph** from the same news corpus: an LLM turns each article into structured *(subject → relationship → object)* facts over a fixed vocabulary of macro entity and relationship types, name variants of an entity are merged across articles, and the facts are aggregated into a directed graph with per-edge provenance (which articles asserted each relationship). [`src/kg/visualize.py`](src/kg/visualize.py) renders the result as a single self-contained, interactive HTML page — GPU force-directed layout, search, type/edge filtering, click-to-focus a node, and source-article inspection. See [`scripts/run_kg.sh`](scripts/run_kg.sh) to run the extraction.
