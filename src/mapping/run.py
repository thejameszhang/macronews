#!/usr/bin/env python3
"""
Batch mapping of news headlines to macro assets.
"""

import argparse
import logging
import time
from pathlib import Path

import polars as pl

from config.paths import MAPPINGS_DIR as OUT_DIR, DEFAULT_MODEL
from mapping.llm import LLMMapper
from mapping.schemas import MappingResult
from utils.io import load_wsj_year
from utils.logging import setup_slurm_logging

setup_slurm_logging()
logger = logging.getLogger(__name__)

def save_mappings(
    df: pl.DataFrame,
    results: list[MappingResult],
    out_path: Path,
) -> None:
    """Save mapping results as parquet."""
    records = []
    for i in range(len(results)):
        row = df.row(i, named=True)
        r = results[i]
        records.append({
            "article_id": row["article_id"],
            "effective_date": row["effective_date"],
            "headline": row["headline"],
            "relevant_assets": r.relevant_assets,
            "n_relevant": len(r.relevant_assets),
        })

    out_df = pl.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(out_path)
    logger.info(f"Saved {len(records):,} mappings to {out_path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years", type=int, nargs="+", default=[],
        help="Process full years and save to parquet",
    )
    parser.add_argument(
        "--model", type=str, default=str(DEFAULT_MODEL),
        help=f"Path to instruct model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=2048,
        help="Max context length for vLLM (default: 2048)",
    )
    args = parser.parse_args()

    if not args.years:
        parser.error("Specify --years YYYY [YYYY ...]")

    # Initialize Mapper
    mapper = LLMMapper(model_path=args.model, max_model_len=args.max_model_len)

    t0 = time.time()

    for year in args.years:
        logger.info(f"Processing year {year}...")
        df = load_wsj_year(year)
        if df.shape[0] == 0:
            logger.info(f"No articles found for {year}, skipping")
            continue

        logger.info(f"Loaded {df.shape[0]:,} articles")

        headlines = df["headline"].to_list()

        t_year = time.time()
        results = mapper.map(headlines)

        elapsed = time.time() - t_year
        logger.info(f"Year {year} mapping complete: {elapsed:.1f}s "
                    f"({len(results) / max(elapsed, 0.1):.1f} articles/sec)")

        # Save to model-specific subdirectory
        model_name = Path(args.model).name
        model_out_dir = OUT_DIR / model_name
        out_path = model_out_dir / f"{year}.parquet"
        
        save_mappings(df, results, out_path)

        n_empty = sum(1 for r in results if not r.relevant_assets)
        n_assets_list = [len(r.relevant_assets) for r in results]
        mean_assets = sum(n_assets_list) / max(len(results), 1)
        logger.info(f"Year {year} stats: {n_empty:,} empty ({n_empty / len(results):.1%}), "
                    f"mean {mean_assets:.1f} assets/article")

    total = time.time() - t0
    logger.info(f"Total processing time: {total:.1f}s")

if __name__ == "__main__":
    main()
