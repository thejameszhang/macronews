"""Tests for src/mapping/grading/schemas.py."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mapping.grading.schemas import GraderResult, is_self_inconsistent  # noqa: E402


def test_default_construction():
    r = GraderResult()
    assert r.evidence_paragraphs == []
    assert r.relevance_score == 0.0
    # Default verdict is `correct` — under default-to-correct semantics,
    # absent positive evidence the tag is wrong, give the mapper the
    # benefit of the doubt.
    assert r.verdict == "correct"


def test_full_construction_roundtrip():
    r = GraderResult(
        evidence_paragraphs=[0, 3, 7],
        relevance_score=0.85,
        verdict="correct",
    )
    raw = r.model_dump_json()
    parsed = json.loads(raw)
    # Field order in the JSON must be evidence -> score -> verdict (CoT).
    keys = list(parsed.keys())
    assert keys == ["evidence_paragraphs", "relevance_score", "verdict"]
    r2 = GraderResult.model_validate_json(raw)
    assert r2 == r


def test_verdict_enum_only_correct_or_incorrect():
    GraderResult(verdict="correct")
    GraderResult(verdict="incorrect")
    # `borderline` was dropped from the schema; uncertainty lives
    # entirely in `relevance_score` now.
    with pytest.raises(Exception):
        GraderResult(verdict="borderline")
    with pytest.raises(Exception):
        GraderResult(verdict="maybe")


def test_score_accepts_zero_and_one():
    GraderResult(relevance_score=0.0)
    GraderResult(relevance_score=1.0)


# ---------------------------------------------------------------------------
# Self-consistency: based on score gap between mapper and grader.
# correct + |gap| > 0.40 -> flag
# incorrect + |gap| < 0.15 -> flag
# ---------------------------------------------------------------------------

def test_self_inconsistency_correct_with_large_gap_flags():
    # Grader said correct but disagreed with mapper by > 0.40
    assert is_self_inconsistent(0.85, 0.20, "correct") is True


def test_self_inconsistency_correct_with_small_gap_ok():
    # Strong agreement
    assert is_self_inconsistent(0.85, 0.80, "correct") is False
    # Weak-but-real agreement (low-low)
    assert is_self_inconsistent(0.30, 0.30, "correct") is False
    # Mid-range agreement
    assert is_self_inconsistent(0.45, 0.50, "correct") is False


def test_self_inconsistency_correct_with_borderline_gap_ok():
    # Gap exactly at threshold — should NOT flag
    assert is_self_inconsistent(0.70, 0.30, "correct") is False  # gap = 0.40


def test_self_inconsistency_incorrect_with_small_gap_flags():
    # Grader said incorrect but scores barely differ
    assert is_self_inconsistent(0.40, 0.35, "incorrect") is True
    assert is_self_inconsistent(0.30, 0.30, "incorrect") is True


def test_self_inconsistency_incorrect_with_large_gap_ok():
    # Mapper over-confident — grader correctly disagrees
    assert is_self_inconsistent(0.85, 0.10, "incorrect") is False
    # Mapper under-confident — grader correctly disagrees
    assert is_self_inconsistent(0.10, 0.85, "incorrect") is False


def test_self_inconsistency_incorrect_at_threshold_ok():
    # Gap exactly at threshold — should NOT flag
    assert is_self_inconsistent(0.40, 0.25, "incorrect") is False  # gap = 0.15
