"""
Shared I/O helpers for loading data.
"""

from pathlib import Path

import numpy as np
import polars as pl

from config.paths import WSJ_DIR
from ingestion.schemas import ARTICLE_SCHEMA


def load_articles_year(year: int, source_dir: Path) -> pl.DataFrame:
    """Load all articles for a given year from any ARTICLE_SCHEMA-conforming source.

    Reads all parquet files under source_dir/YYYY/**/*, filters out null/empty
    headlines, and returns sorted by effective_date. Works with any ingestion
    source (WSJ, DJNW, Reuters) as long as output conforms to ARTICLE_SCHEMA.
    """
    year_dir = source_dir / str(year)
    if not year_dir.exists():
        return pl.DataFrame()
    files = sorted(year_dir.rglob("*.parquet"))
    if not files:
        return pl.DataFrame()
    df = pl.concat([pl.read_parquet(f) for f in files])
    df = df.filter(
        pl.col("headline").is_not_null()
        & (pl.col("headline").str.len_chars() > 0)
    )
    return df.sort("effective_date")


def load_wsj_year(year: int, wsj_dir: Path = WSJ_DIR) -> pl.DataFrame:
    """Load all WSJ headline articles for a given year.

    Convenience wrapper around load_articles_year for WSJ data.
    """
    return load_articles_year(year, wsj_dir)


def validate_article_schema(df: pl.DataFrame) -> list[str]:
    """Check that a DataFrame conforms to ARTICLE_SCHEMA.

    Returns a list of error strings. Empty list means valid.
    Useful for catching bugs early when writing a new ingestor.
    """
    errors = []

    # Check required columns exist
    for col, expected_dtype in ARTICLE_SCHEMA.items():
        if col not in df.columns:
            errors.append(f"missing column: {col}")
        elif df[col].dtype != expected_dtype:
            errors.append(
                f"column '{col}': expected {expected_dtype}, got {df[col].dtype}"
            )

    # Check for unexpected columns
    extra = set(df.columns) - set(ARTICLE_SCHEMA.keys())
    if extra:
        errors.append(f"unexpected columns: {sorted(extra)}")

    if errors:
        return errors

    # Check required non-null fields
    for col in ("article_id", "effective_date", "source", "headline"):
        null_count = df[col].null_count()
        if null_count > 0:
            errors.append(f"column '{col}' has {null_count} null values")

    # Check source values
    valid_sources = {"djnw", "reuters", "wsj"}
    if "source" in df.columns:
        sources = set(df["source"].unique().to_list())
        bad = sources - valid_sources
        if bad:
            errors.append(f"invalid source values: {sorted(bad)}")

    return errors


def load_embeddings(emb_dir: Path) -> tuple[np.ndarray, pl.DataFrame]:
    """Load all embedding shards and metadata from a directory.

    Expects paired files: YYYY.npy and YYYY_meta.parquet.
    Returns (embeddings array, metadata DataFrame).
    """
    year_npys = sorted(emb_dir.glob("*.npy"))
    year_metas = sorted(emb_dir.glob("*_meta.parquet"))

    if not year_npys:
        raise FileNotFoundError(f"No .npy embedding files found in {emb_dir}")

    emb_shards, meta_shards = [], []
    for npy, meta_path in zip(year_npys, year_metas):
        emb_shards.append(np.load(npy))
        meta_shards.append(pl.read_parquet(meta_path))

    return np.concatenate(emb_shards, axis=0), pl.concat(meta_shards)
