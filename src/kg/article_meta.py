"""Shared cleaned-shard metadata: join each article's exact release time
(`display_date`) by `accession_number`. The KG sidecar carries only the
day-level docdate; the cleaned shards carry a fully-populated ms-precision
`display_date`. Used by the cosmos viz (within-day fact ordering in the Reader
tab) and by scripts/news_returns_overlay.py.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

CLEANED = "/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles"


def shard_path(stem: str) -> Path:
    """Cleaned-shard path for a stem like '2014-05a'."""
    return Path(CLEANED) / f"{stem}_clean.jsonl"


def parse_display_dt(s: str) -> datetime:
    """'20140502T174831.898Z' -> datetime (UTC, drop sub-second + Z)."""
    return datetime.strptime(s[:15], "%Y%m%dT%H%M%S")


def load_display_dates(shards: list[Path]) -> dict[str, str]:
    """Map ``accession_number -> display_date`` (raw basic-ISO string,
    lexically sortable) from cleaned shards. Missing files are skipped; rows
    without a display_date are skipped."""
    out: dict[str, str] = {}
    for s in shards:
        if not Path(s).exists():
            continue
        with open(s) as f:
            for line in f:
                if not line.strip():
                    continue
                a = json.loads(line)
                dd = a.get("display_date")
                if dd:
                    out[a["accession_number"]] = dd
    return out
