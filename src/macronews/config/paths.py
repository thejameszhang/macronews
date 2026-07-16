"""
Centralized path constants for the macronews project.

All modules should import paths from here instead of computing ROOT themselves.
"""

from pathlib import Path

# Project root: three levels up from src/macronews/config/paths.py
ROOT = Path(__file__).resolve().parents[3]

# Data directories (not committed — created by scripts at runtime)
DATA_DIR = ROOT / "data"
WSJ_DIR = DATA_DIR / "wsj"
DJNW_DIR = DATA_DIR / "djnw"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
ARTICLE_EMBEDDINGS_DIR = EMBEDDINGS_DIR / "wsj"
PRIMERS_DIR = DATA_DIR / "primers" / "web"
MAPPINGS_DIR = DATA_DIR / "mappings"
ARTICLES_SAMPLE_DIR = DATA_DIR / "articles_sample"

# Datasets (returns, etc.)
DATASETS_DIR = ROOT / "datasets"
RETURNS_CSV = DATASETS_DIR / "sync_daily.csv"

# Results (not committed)
RESULTS_DIR = ROOT / "results"

# Secrets (not committed)
SECRETS_DIR = ROOT / "secrets"

# Source config
CONFIG_DIR = ROOT / "src" / "macronews" / "config"
ASSET_UNIVERSE_YAML = CONFIG_DIR / "asset_universe.yaml"
GROUP_UNIVERSE_YAML = CONFIG_DIR / "group_universe.yaml"

# Prompts
PROMPTS_DIR = ROOT / "src" / "macronews" / "mapping" / "prompts"

# KG (Phase 1) prompts. Kept separate from PROMPTS_DIR (which is mapper-scoped)
# so iteration on KG taxonomies doesn't risk touching mapper prompt files.
KG_PROMPTS_DIR = ROOT / "src" / "macronews" / "kg" / "prompts"

# The weights production actually runs. Both live on scratch, which auto-purges by
# atime -- if a loader dies at AutoTokenizer complaining about sentencepiece, the
# directory is empty, not broken: re-download with slurm/download_llm.sh.
_MODELS = Path("/nfs/roberts/scratch/pi_btk22/jyz32")
MAPPER_MODEL = _MODELS / "gemma-4-26b-a4b-it"
GRADER_MODEL = _MODELS / "qwq-32b"

# NOT an LLM: the sentence-transformers embedder used by kg disambiguate. Named here
# so it stops being a module-local `DEFAULT_MODEL` that reads like the others.
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

# The djnw corpus.
DJNW_ARTICLES_DIR = Path("/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles")

# THE PROJECT IS POST-1996. The corpus on disk reaches back to 1979-06, but those years are
# out of scope, and the tabular-body detector never ran on them -- its sidecar covers exactly
# 1996-01..2025-08. Counting them burns 206 of the 1,260 shards on totals nobody quotes.
#
# This lived in the documentation's generator, which is gitignored: the definition of what
# "the corpus" means was in a file that ships nowhere.
FIRST_YEAR = 1996


def corpus_shards() -> list[Path]:
    """Every in-scope DJNW shard, sorted. The one definition of "the corpus"."""
    return sorted(p for p in DJNW_ARTICLES_DIR.glob("*_clean.jsonl")
                  if int(p.name[:4]) >= FIRST_YEAR)
