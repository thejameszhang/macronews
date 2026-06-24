"""Tests for src/kg/article_meta.py (cleaned-shard display_date join)."""
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kg.article_meta import CLEANED, load_display_dates, parse_display_dt, shard_path  # noqa: E402


def test_shard_path_matches_cleaned_layout():
    # Must reproduce the overlay's old path: CLEANED/<stem>_clean.jsonl.
    assert shard_path("2014-05a") == Path(CLEANED) / "2014-05a_clean.jsonl"


def test_parse_display_dt_drops_subsecond_and_z():
    assert parse_display_dt("20140502T174831.898Z") == datetime(2014, 5, 2, 17, 48, 31)


def test_load_display_dates_returns_accession_to_string_map(tmp_path):
    shard = tmp_path / "s_clean.jsonl"
    shard.write_text(
        json.dumps({"accession_number": "art1", "display_date": "20140502T174831.898Z"}) + "\n"
        + json.dumps({"accession_number": "art2", "display_date": "20140503T090000.000Z"}) + "\n")
    assert load_display_dates([shard]) == {
        "art1": "20140502T174831.898Z", "art2": "20140503T090000.000Z"}


def test_load_display_dates_skips_missing_files(tmp_path):
    shard = tmp_path / "present_clean.jsonl"
    shard.write_text(
        json.dumps({"accession_number": "art1", "display_date": "20140502T174831.898Z"}) + "\n")
    missing = tmp_path / "absent_clean.jsonl"
    assert load_display_dates([missing, shard]) == {"art1": "20140502T174831.898Z"}


def test_load_display_dates_skips_rows_without_display_date(tmp_path):
    shard = tmp_path / "s_clean.jsonl"
    shard.write_text(
        json.dumps({"accession_number": "art1"}) + "\n"
        + json.dumps({"accession_number": "art2", "display_date": "20140503T090000.000Z"}) + "\n")
    assert load_display_dates([shard]) == {"art2": "20140503T090000.000Z"}


def test_load_display_dates_merges_across_present_shards(tmp_path):
    """Two present shards with distinct keys are merged (not last-wins-overwrite)."""
    a = tmp_path / "a_clean.jsonl"
    a.write_text(json.dumps({"accession_number": "art1", "display_date": "20140502T174831.898Z"}) + "\n")
    b = tmp_path / "b_clean.jsonl"
    b.write_text(json.dumps({"accession_number": "art2", "display_date": "20140503T090000.000Z"}) + "\n")
    assert load_display_dates([a, b]) == {
        "art1": "20140502T174831.898Z", "art2": "20140503T090000.000Z"}
