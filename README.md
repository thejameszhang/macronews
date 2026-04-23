# MacroNews

NLP pipeline for global macro futures return prediction. Maps news articles to relevant futures contracts (equity indices, FX, commodities, government bonds, short-term interest rates) using LLM-based classification, then feeds relevance-weighted text into ridge regression for return forecasting.

This is the first systematic text-as-data study targeting macro futures rather than equities. The key challenge is the **mapping problem**: unlike stocks, macro futures have no ticker mentions in news text, so relevance must be inferred through economic reasoning.

## Pipeline

**ArticleMapper** — one LLM call per (article, asset) pair. For each pair the model returns:
  - `relevant`: yes/no
  - `evidence_paragraphs`: indices of paragraphs where a rule-triggering force appears
  - `reasoning`: rule citation + quoted phrase from the article
  - `signal`: `"strong"` if the asset is named in the text, else `"weak"`
  - `relevance_score`: 0.0--1.0

The system prompt is `src/mapping/prompts/mapper.txt` with asset-class-specific disqualifiers and positive rules from `src/mapping/prompts/asset_class/{currency,rates,bonds,commodities,equity_sectors,equity_indices,volatility_indices}.txt` injected per batch.

Downstream ridge regression on relevance-weighted article aggregates is a separate stage outside this repo.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install base dependencies
uv sync

# 2. Install pip into the venv (uv venvs don't include it by default)
.venv/bin/python -m ensurepip

# 3. Override vLLM with 0.19.0 (PyPI only has 0.11.0, which lacks Gemma 4 support)
.venv/bin/python -m pip install vllm --force-reinstall --extra-index-url https://wheels.vllm.ai/nightly

# 4. Override transformers to 5.x (vLLM pins <5, but Gemma 4 config needs 5.x)
.venv/bin/python -m pip install --upgrade transformers

# 5. Verify
.venv/bin/python -c "from vllm.model_executor.models.registry import _VLLM_MODELS; assert 'Gemma4ForCausalLM' in _VLLM_MODELS, 'Missing Gemma4'"
```

> **Note:** `uv sync` alone is insufficient -- the pip overrides in steps 3-4 are required for Gemma 4 model support. Run scripts with `source .venv/bin/activate && python ...`, not `uv run`.

## Running the pipeline

```bash
# Set model path (defaults to gemma-4-31b-it if unset)
export MODEL_PATH=$HOME/models/gemma-4-26b-a4b-it

# Gold sample (20 labeled articles) — default
bash scripts/run_pipeline.sh

# DJNW March 2022, 1000-article seeded sample
DATASET=djnw MAX_ARTICLES=1000 \
  START_DATE=2022-03 END_DATE=2022-03 RANDOM_SEED=42 \
  bash scripts/run_pipeline.sh
```

Results land in `results/<dataset>_summary.json`. See `scripts/run_pipeline.sh` for all env knobs.

## Data

- **Articles**: DJNW cleaned JSONL at `/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles/YYYY-MM*_clean.jsonl`. Gold labeled samples for regression checks at `data/articles_sample/gold_*.json`. Sports news and WikiGaming loaders also wired up via `--dataset`.
- **Returns**: `datasets/sync_daily.csv` (1996--2025, 95 assets)
- **Asset universe**: `src/config/asset_universe.yaml` (95 active contracts)
