"""
AbstractIngestor — common interface that all news source ingestors must implement.

Each concrete subclass is responsible for one data source (DJNW, Reuters, WSJ).
The contract: given a date range and the set of valid trading days, produce
ArticleRecord Parquet files partitioned by effective_date under data/articles/.
"""

from __future__ import annotations

import abc
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from .schemas import ARTICLE_SCHEMA

ET = ZoneInfo("America/New_York")

# 9:00 AM ET cutoff (minutes since midnight ET)
_CUTOFF_MINUTES = 9 * 60          # 540
_BUFFER_END_MINUTES = 9 * 60 + 30 # 570


def _et_minutes(dt: datetime) -> int:
    """Return minutes since midnight for an ET-localised datetime."""
    et = dt.astimezone(ET)
    return et.hour * 60 + et.minute


def is_buffer_zone(eff_time: datetime) -> bool:
    """True if eff_time falls in [9:00, 9:30) AM ET."""
    m = _et_minutes(eff_time)
    return _CUTOFF_MINUTES <= m < _BUFFER_END_MINUTES


def assign_trading_day(
    eff_time: datetime,
    valid_trading_days: frozenset[date],
) -> date | None:
    """
    Assign an article to trading day t such that eff_time < 9:00 AM ET on day t.

    If eff_time >= 9:00 AM ET on its calendar date, the article belongs to the
    next day's signal window. Then roll forward until a valid trading day is found.

    For dates beyond the range of sync_daily.csv (e.g. current year not yet in
    the CSV), we fall back to a simple Mon–Fri weekday check, which is correct
    for all major US equity/futures markets excluding US holidays. This lets the
    ingestor run on current-year data before sync_daily.csv is extended.
    """
    et = eff_time.astimezone(ET)
    d = et.date()
    if _et_minutes(et) >= _CUTOFF_MINUTES:
        d += timedelta(days=1)

    max_date = max(valid_trading_days) if valid_trading_days else None

    for _ in range(7):
        if d in valid_trading_days:
            return d
        # Beyond the CSV range: accept any Mon–Fri as a trading day
        if max_date is not None and d > max_date and d.weekday() < 5:
            return d
        d += timedelta(days=1)
    return None


def load_trading_days(sync_daily_path: Path) -> frozenset[date]:
    """
    Derive the set of valid US trading days from sync_daily.csv.

    A row exists in sync_daily.csv iff that date is a US business day on which
    at least one asset traded. No external calendar library needed.
    """
    df = pl.scan_csv(sync_daily_path).select("date").collect()
    return frozenset(
        date.fromisoformat(d) for d in df["date"].to_list()
    )


class AbstractIngestor(abc.ABC):
    """
    Base class for all news source ingestors.

    Subclasses implement `ingest_date()` which processes a single calendar date
    and returns a list of ArticleRecord dicts. The base class handles trading-day
    assignment, Parquet I/O, and partitioned storage.
    """

    source_name: str  # must be set by subclass: "djnw" | "reuters" | "wsj"

    def __init__(
        self,
        output_root: Path,
        trading_days: frozenset[date],
    ) -> None:
        self.output_root = output_root
        self.trading_days = trading_days

    @abc.abstractmethod
    def ingest_date(self, target_date: date) -> list[dict]:
        """
        Fetch and preprocess all articles for a single calendar date.

        Returns a list of dicts conforming to ARTICLE_SCHEMA. Implementations
        should apply source-specific filtering (language, word count, section
        exclusion) and populate all fields except effective_date — that is
        filled by the base class via assign_trading_day().

        The effective_time field must be an ET-aware datetime.
        """
        ...

    def run(self, start_date: date, end_date: date) -> int:
        """
        Ingest all calendar dates in [start_date, end_date] inclusive.

        Skips dates that already have a Parquet file on disk (idempotent).
        Prints one status line per processed date to stdout (visible in SLURM logs).
        Returns total number of ArticleRecords written.
        """
        import sys
        total = 0
        d = start_date
        while d <= end_date:
            out_path = self._parquet_path(d)
            if out_path.exists():
                d += timedelta(days=1)
                continue
            records = self.ingest_date(d)
            if records:
                self._write_parquet(records, out_path)
                total += len(records)
                print(f"  {d}  {len(records):4d} records  cumulative={total:,}", flush=True)
            else:
                print(f"  {d}  (no records)", flush=True)
            d += timedelta(days=1)
        return total

    def _parquet_path(self, calendar_date: date) -> Path:
        p = self.output_root / f"{calendar_date.year:04d}" / f"{calendar_date.month:02d}"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{calendar_date.day:02d}.parquet"

    def _write_parquet(self, records: list[dict], path: Path) -> None:
        df = pl.DataFrame(records, schema=ARTICLE_SCHEMA)
        df.write_parquet(path.resolve(), compression="zstd")
