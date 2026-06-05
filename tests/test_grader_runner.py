"""Tests for src/mapping/grading/runner.py — build_tasks + write_sidecar (no model)."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mapping.grading.runner import build_tasks, write_sidecar  # noqa: E402
from mapping.grading.schemas import GraderResult  # noqa: E402


def _source_article(aid="A1"):
    return {
        "id": aid,
        "headline": "Brent surges as OPEC trims production",
        "paragraphs": [
            "Brent crude rose to a six-month high.",
            "OPEC announced an unexpected production cut.",
            "Analysts expect prices to stay elevated.",
        ],
    }


def _group_universe():
    return {
        "Crude Oil": {
            "name": "Crude Oil",
            "asset_class": "commodity",
            "members": [
                {"name": "WTI Crude Oil", "ticker_symbol": "CL"},
                {"name": "Brent Crude Oil", "ticker_symbol": "BRN"},
            ],
        },
        "Natural Gas": {
            "name": "Natural Gas",
            "asset_class": "commodity",
            "members": [{"name": "Natural Gas", "ticker_symbol": "NG"}],
        },
    }


def test_build_tasks_basic_join():
    mapper_rows = [
        {
            "article_id": "A1",
            "headline": "Brent surges",
            "mappings": [
                {
                    "group": "Crude Oil",
                    "asset_class": "commodity",
                    "relevance_score": 0.9,
                    "evidence_paragraphs": [0, 1],
                }
            ],
        }
    ]
    src = {"A1": _source_article("A1")}
    inputs, meta, _skipped = build_tasks(mapper_rows, src, _group_universe())
    assert len(inputs) == 1
    assert len(meta) == 1
    assert inputs[0].article_id == "A1"
    assert inputs[0].group_name == "Crude Oil"
    assert inputs[0].asset_class == "commodity"
    assert inputs[0].member_names == ["WTI Crude Oil", "Brent Crude Oil"]
    assert inputs[0].mapper_evidence_paragraphs == [0, 1]
    assert inputs[0].paragraphs == _source_article("A1")["paragraphs"]
    assert meta[0]["mapper_score"] == 0.9


def test_build_tasks_skips_unknown_articles():
    mapper_rows = [
        {"article_id": "GHOST", "mappings": [
            {"group": "Crude Oil", "asset_class": "commodity",
             "relevance_score": 0.9, "evidence_paragraphs": []}
        ]},
        {"article_id": "A1", "headline": "ok", "mappings": [
            {"group": "Crude Oil", "asset_class": "commodity",
             "relevance_score": 0.5, "evidence_paragraphs": [0]}
        ]},
    ]
    src = {"A1": _source_article("A1")}
    inputs, meta, skipped = build_tasks(mapper_rows, src, _group_universe())
    assert len(inputs) == 1
    assert inputs[0].article_id == "A1"
    # GHOST should produce a skip row, not silent drop
    assert len(skipped) == 1
    assert skipped[0]["article_id"] == "GHOST"
    assert skipped[0]["skip_reason"] == "article_not_in_source"
    assert skipped[0]["group_name"] == "Crude Oil"


def test_build_tasks_skips_unknown_groups_records_skip():
    mapper_rows = [
        {"article_id": "A1", "headline": "ok", "mappings": [
            {"group": "DefunctGroup", "asset_class": "commodity",
             "relevance_score": 0.4, "evidence_paragraphs": []},
            {"group": "Natural Gas", "asset_class": "commodity",
             "relevance_score": 0.6, "evidence_paragraphs": [1]},
        ]},
    ]
    src = {"A1": _source_article("A1")}
    inputs, meta, skipped = build_tasks(mapper_rows, src, _group_universe())
    assert len(inputs) == 1
    assert inputs[0].group_name == "Natural Gas"
    assert len(skipped) == 1
    assert skipped[0]["group_name"] == "DefunctGroup"
    assert skipped[0]["skip_reason"] == "group_not_in_universe"
    assert skipped[0]["mapper_score"] == 0.4


def test_build_tasks_multiple_mappings_per_article():
    mapper_rows = [
        {"article_id": "A1", "headline": "ok", "mappings": [
            {"group": "Crude Oil", "asset_class": "commodity",
             "relevance_score": 0.8, "evidence_paragraphs": [0]},
            {"group": "Natural Gas", "asset_class": "commodity",
             "relevance_score": 0.4, "evidence_paragraphs": [2]},
        ]},
    ]
    src = {"A1": _source_article("A1")}
    inputs, meta, _skipped = build_tasks(mapper_rows, src, _group_universe())
    assert len(inputs) == 2
    assert {i.group_name for i in inputs} == {"Crude Oil", "Natural Gas"}


def test_write_sidecar_includes_skipped_rows(tmp_path):
    meta = [{
        "article_id": "A1", "group_name": "Crude Oil",
        "asset_class": "commodity", "mapper_score": 0.9,
        "mapper_evidence_paragraphs": [0],
    }]
    results = [GraderResult(evidence_paragraphs=[0], relevance_score=0.9, verdict="correct")]
    skipped = [
        {
            "article_id": "A1", "group_name": "Swedish Equities",
            "asset_class": "equity index", "mapper_score": 0.5,
            "mapper_evidence_paragraphs": [3],
            "skip_reason": "group_not_in_universe",
        },
        {
            "article_id": "GHOST", "group_name": "Crude Oil",
            "asset_class": "commodity", "mapper_score": 0.7,
            "mapper_evidence_paragraphs": [],
            "skip_reason": "article_not_in_source",
        },
    ]
    out = tmp_path / "sidecar.jsonl"
    summary = write_sidecar(out, meta, results, skipped=skipped)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 3  # 1 graded + 2 skipped
    graded = [r for r in rows if r["skip_reason"] is None]
    skipped_rows = [r for r in rows if r["skip_reason"] is not None]
    assert len(graded) == 1
    assert graded[0]["grader_verdict"] == "correct"
    assert len(skipped_rows) == 2
    assert {r["skip_reason"] for r in skipped_rows} == {
        "group_not_in_universe", "article_not_in_source"
    }
    for r in skipped_rows:
        assert r["grader_verdict"] == "skipped"
        assert r["grader_score"] is None
        assert r["grader_evidence_paragraphs"] == []
    assert summary["graded"] == 1
    assert summary["skipped"] == 2
    assert summary["skip_reasons"] == {
        "group_not_in_universe": 1,
        "article_not_in_source": 1,
    }
    assert summary["total_mappings"] == 3


def test_write_sidecar_roundtrip(tmp_path):
    meta = [
        {
            "article_id": "A1",
            "group_name": "Crude Oil",
            "asset_class": "commodity",
            "mapper_score": 0.9,
            "mapper_evidence_paragraphs": [0, 1],
        },
        {
            "article_id": "A1",
            "group_name": "Natural Gas",
            "asset_class": "commodity",
            "mapper_score": 0.4,
            "mapper_evidence_paragraphs": [],
        },
    ]
    results = [
        GraderResult(evidence_paragraphs=[0, 1], relevance_score=0.92, verdict="correct"),
        GraderResult(evidence_paragraphs=[], relevance_score=0.05, verdict="incorrect"),
    ]
    out = tmp_path / "sidecar.jsonl"
    summary = write_sidecar(out, meta, results)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["grader_verdict"] == "correct"
    assert rows[0]["grader_score"] == 0.92
    assert rows[0]["self_consistency_flag"] is False
    assert rows[1]["grader_verdict"] == "incorrect"
    assert summary == {
        "total_mappings": 2,
        "graded": 2,
        "correct": 1,
        "incorrect": 1,
        "self_inconsistent": 0,
        "skipped": 0,
        "skip_reasons": {},
        "by_asset_class": {
            "commodity": {
                "total": 2, "correct": 1, "incorrect": 1,
                "self_inconsistent": 0, "correct_pct": 50.0,
            },
        },
        "by_group": {
            "Crude Oil": {
                "total": 1, "correct": 1, "incorrect": 0,
                "self_inconsistent": 0, "correct_pct": 100.0,
            },
            "Natural Gas": {
                "total": 1, "correct": 0, "incorrect": 1,
                "self_inconsistent": 0, "correct_pct": 0.0,
            },
        },
    }
    summary_file = json.loads((out.with_suffix(".summary.json")).read_text())
    assert summary_file == summary


def test_write_sidecar_flags_inconsistency(tmp_path):
    # gold_07 Industrials shape: mapper=0.30, grader=0.30, verdict=incorrect.
    # Both LLMs agree on a low score (gap 0.0) but verdict says the mapper
    # made a real mistake — internally inconsistent under the new rule.
    meta = [{
        "article_id": "A1", "group_name": "Crude Oil",
        "asset_class": "commodity", "mapper_score": 0.30,
        "mapper_evidence_paragraphs": [],
    }]
    results = [GraderResult(
        evidence_paragraphs=[], relevance_score=0.30, verdict="incorrect"
    )]
    out = tmp_path / "sidecar.jsonl"
    summary = write_sidecar(out, meta, results)
    row = json.loads(out.read_text())
    assert row["self_consistency_flag"] is True
    assert summary["self_inconsistent"] == 1


def test_write_sidecar_correct_with_large_gap_flags(tmp_path):
    # Verdict=correct but scores differ by > 0.40 — flagged.
    meta = [{
        "article_id": "A1", "group_name": "Crude Oil",
        "asset_class": "commodity", "mapper_score": 0.85,
        "mapper_evidence_paragraphs": [0],
    }]
    results = [GraderResult(
        evidence_paragraphs=[0], relevance_score=0.20, verdict="correct"
    )]
    out = tmp_path / "sidecar.jsonl"
    summary = write_sidecar(out, meta, results)
    row = json.loads(out.read_text())
    assert row["self_consistency_flag"] is True
    assert summary["self_inconsistent"] == 1


def test_write_sidecar_by_class_and_by_group_breakdown(tmp_path):
    # Two asset classes (commodity, rates) split across three groups, with
    # one self-inconsistent row to verify the overlay accounting per bucket.
    meta = [
        {"article_id": "A1", "group_name": "Crude Oil", "asset_class": "commodity",
         "mapper_score": 0.9, "mapper_evidence_paragraphs": [0]},
        {"article_id": "A1", "group_name": "Crude Oil", "asset_class": "commodity",
         "mapper_score": 0.4, "mapper_evidence_paragraphs": []},
        {"article_id": "A2", "group_name": "Natural Gas", "asset_class": "commodity",
         "mapper_score": 0.7, "mapper_evidence_paragraphs": [2]},
        {"article_id": "A2", "group_name": "US Rates", "asset_class": "rates",
         # self-inconsistent: verdict=correct but |gap|=0.65
         "mapper_score": 0.85, "mapper_evidence_paragraphs": [3]},
    ]
    results = [
        GraderResult(evidence_paragraphs=[0], relevance_score=0.85, verdict="correct"),
        GraderResult(evidence_paragraphs=[], relevance_score=0.05, verdict="incorrect"),
        GraderResult(evidence_paragraphs=[2], relevance_score=0.6, verdict="correct"),
        GraderResult(evidence_paragraphs=[3], relevance_score=0.20, verdict="correct"),
    ]
    out = tmp_path / "sidecar.jsonl"
    summary = write_sidecar(out, meta, results)
    assert summary["by_asset_class"] == {
        "commodity": {"total": 3, "correct": 2, "incorrect": 1,
                      "self_inconsistent": 0, "correct_pct": 66.67},
        "rates": {"total": 1, "correct": 1, "incorrect": 0,
                  "self_inconsistent": 1, "correct_pct": 100.0},
    }
    assert summary["by_group"] == {
        "Crude Oil": {"total": 2, "correct": 1, "incorrect": 1,
                      "self_inconsistent": 0, "correct_pct": 50.0},
        "Natural Gas": {"total": 1, "correct": 1, "incorrect": 0,
                        "self_inconsistent": 0, "correct_pct": 100.0},
        "US Rates": {"total": 1, "correct": 1, "incorrect": 0,
                     "self_inconsistent": 1, "correct_pct": 100.0},
    }
    # Skipped rows should NOT show up in by_class/by_group
    assert summary["graded"] == 4
    assert summary["correct"] == 3
    assert summary["incorrect"] == 1
    assert summary["self_inconsistent"] == 1
