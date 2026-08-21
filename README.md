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

Asset universe: 39 groups spanning global equities, FX, rates, commodities, and volatility. Defined in [`src/macronews/config/group_universe.yaml`](src/macronews/config/group_universe.yaml), which describes all 50 — the 11 US equity sector groups are read by the knowledge-graph lane and the grader but excluded from the mapper (see `utils/groups.MAPPER_EXCLUDED_ASSET_CLASSES`).

## Running

```bash
# Activate the env (do NOT use `uv run` — see Setup)
source .venv/bin/activate

# Single DJNW shard, production mode
MODE=prod \
  INPUT_FILE=/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles/2014-05c_clean.jsonl \
  OUTPUT_DIR=results/mapping/prod/2014-gated/mapper \
  bash slurm/run_pipeline.sh

# Many shards in parallel via SLURM array
OUT_DIR=results/mapping/prod/2014-gated/mapper PATTERN='2014-*' \
  bash slurm/run_pipeline_array.sh
```

Outputs are JSONL (one record per article) with sidecar `.summary.json` aggregates.

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and a Hopper-or-newer GPU (B200 or H200).

The mapper reads articles through [`djnw`](https://github.com/thejameszhang/djnw), a private
companion library (the DJNW corpus reader) that is **not on PyPI**. Clone it and point uv at
your clone first, or `uv sync` will fail to resolve it:

```bash
git clone git@github.com:thejameszhang/djnw.git ../djnw   # e.g. a sibling of this repo
```
```toml
# then add to pyproject.toml so uv can resolve it:
[tool.uv.sources]
djnw = { path = "../djnw", editable = true }
```

```bash
# 1. Install project dependencies (with djnw resolvable, per above)
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
# Edit .env to fill in HF_TOKEN (needed by slurm/download_llm.sh to pull model weights).
```

## Data

- **Articles**: Dow Jones Newswires (1996–2025), cleaned JSONL shards at `/nfs/roberts/.../v2/articles/`.
- **Returns**: `datasets/sync_daily.csv` (95 macro futures).
- **Asset universe**: [`universe.xlsx`](universe.xlsx) — canonical Tier 1 / Tier 2 / Groups reference (human-readable).
- **Group config**: [`src/macronews/config/group_universe.yaml`](src/macronews/config/group_universe.yaml) — the live configuration the mapper reads at runtime.

## Knowledge Graph

The macroeconomic knowledge graph pipeline that used to live here now lives in a separate `macro-kg` repo.

## Commodity News Factor

The embeddings, ridge fit, and return-prediction factor that consume these mappings now live in a
separate `commodity-news-factor` repo, which reads this repo's published mappings and grades as data
over shared storage. This repo's scope is producing the mappings and grades; the downstream
asset-pricing analysis (the "downstream return regression" above) lives there.

## Running on another cluster

Paths here are Bouchet-specific. The one hardcoded machine path in code is `_MODELS` in
[`src/macronews/config/paths.py`](src/macronews/config/paths.py) (`/nfs/roberts/scratch/pi_btk22/jyz32`
on Bouchet), where model weights are read from — repoint it for a new cluster. The article corpus is
passed at runtime (`INPUT_FILE` / `--input-file`), so pointing the mapper at a different dataset needs
no code change. Model weights are fetched with `slurm/download_llm.sh` into that scratch path.
