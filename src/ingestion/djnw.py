"""
DJNWIngestor — ingest Dow Jones Newswires articles into ARTICLE_SCHEMA.

Stub implementation. Fill in ingest_date() once the DJNW data format is known.
The abstract base class handles trading-day assignment, Parquet I/O, and
partitioned storage — this subclass only needs to parse the raw source format.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .base import AbstractIngestor


class DJNWIngestor(AbstractIngestor):
    """Ingestor for Dow Jones Newswires articles."""

    source_name = "djnw"

    def __init__(
        self,
        output_root: Path,
        trading_days: frozenset[date],
        raw_data_dir: Path | None = None,
    ) -> None:
        super().__init__(output_root, trading_days)
        self.raw_data_dir = raw_data_dir

    def ingest_date(self, target_date: date) -> list[dict]:
        """Parse all DJNW articles for a single calendar date.

        TODO: implement once DJNW data format is known.

        Expected steps:
        1. Locate raw file(s) for target_date in self.raw_data_dir
        2. Parse headline, body, timestamp, subject codes (DJCS), RICs
        3. Build body_excerpt = headline + " " + body[:1200]
        4. Compute effective_time (ET-aware datetime)
        5. Call assign_trading_day(effective_time, self.trading_days)
        6. Return list of dicts conforming to ARTICLE_SCHEMA
        """
        raise NotImplementedError(
            "DJNWIngestor.ingest_date() is a stub — "
            "implement once the DJNW data format is available."
        )
