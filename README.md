# MacroNews

NLP pipeline for global macro futures return prediction. Maps news articles to relevant futures contracts (equity indices, FX, commodities, government bonds, short-term interest rates) using LLM-based classification, then feeds relevance-weighted text into ridge regression for return forecasting.

This is the first systematic text-as-data study targeting macro futures rather than equities. The key challenge is the **mapping problem**: unlike stocks, macro futures have no ticker mentions in news text, so relevance must be inferred through economic reasoning.

## Pipeline

1. **Summarize** -- LLM extracts macro context from each article, filters company-specific news
2. **Article Mapper** -- per-asset relevance classification with transmission channel reasoning
3. **Paragraph Mapper** -- paragraph-level relevance for finer-grained signal detection
4. **Validator** -- cross-references mapper reasoning against full article, assigns final signal strength and relevance score (0.0--1.0)
5. **Ridge Regression** -- downstream supervised model learns direction from relevance-weighted article aggregates

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

## Running Experiments

```bash
# Set model path (defaults to gemma-4-31b-it if unset)
export MODEL_PATH=$HOME/models/gemma-4-26b-a4b-it

# Submit to SLURM
bash scripts/article_depth_experiment.sh
```

## Data

- **Articles**: DJNW (primary), WSJ web archive (interim) in `data/`
- **Returns**: `src/datasets/sync_daily.csv` (1996--2025, 95 assets)
- **Asset universe**: `src/config/asset_universe.yaml` (95 active contracts)
