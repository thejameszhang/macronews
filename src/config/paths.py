"""
Centralized path constants for the macronews project.

All modules should import paths from here instead of computing ROOT themselves.
"""

from pathlib import Path

# Project root: two levels up from src/config/paths.py
ROOT = Path(__file__).resolve().parents[2]

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
CONFIG_DIR = ROOT / "src" / "config"
ASSET_UNIVERSE_YAML = CONFIG_DIR / "asset_universe.yaml"
GROUP_UNIVERSE_YAML = CONFIG_DIR / "group_universe.yaml"

# Prompts
PROMPTS_DIR = ROOT / "src" / "mapping" / "prompts"

# Default model
DEFAULT_MODEL = Path.home() / "models" / "gemma-4-31b-it"
