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

# The weights production actually runs. Both live on scratch, which auto-purges by
# atime -- if a loader dies at AutoTokenizer complaining about sentencepiece, the
# directory is empty, not broken: re-download with slurm/download_llm.sh.
_MODELS = Path("/nfs/roberts/scratch/pi_btk22/jyz32")
MAPPER_MODEL = _MODELS / "gemma-4-26b-a4b-it"
GRADER_MODEL = _MODELS / "qwq-32b"

# The djnw corpus.
from djnw.corpus import DJNW_ARTICLES_DIR, FIRST_YEAR, corpus_shards
