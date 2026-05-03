"""Tests for src/tabular/schemas.py."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tabular.schemas import TabularResult  # noqa: E402


def test_default_construction():
    r = TabularResult(accession_number="20020906003029")
    assert r.accession_number == "20020906003029"
    assert r.p_tokens == 0
    assert r.pre_tabular_tokens == 0
    assert r.pre_narrative_tokens == 0


def test_full_construction_roundtrip():
    r = TabularResult(
        accession_number="20020906003029",
        p_tokens=0,
        pre_tabular_tokens=772,
        pre_narrative_tokens=0,
    )
    raw = r.model_dump_json()
    r2 = TabularResult.model_validate_json(raw)
    assert r2 == r


def test_pre_total_tokens_sums_only_pre_buckets():
    # p_tokens deliberately excluded — pre_total is "tokens inside <pre>"
    r = TabularResult(
        accession_number="x",
        p_tokens=100,
        pre_tabular_tokens=300,
        pre_narrative_tokens=200,
    )
    assert r.pre_total_tokens == 500


def test_pre_aligned_pct_with_mixed_content():
    r = TabularResult(
        accession_number="x",
        pre_tabular_tokens=300,
        pre_narrative_tokens=100,
    )
    assert r.pre_aligned_pct == 0.75


def test_pre_aligned_pct_zero_when_no_pre_content():
    r = TabularResult(accession_number="x")
    assert r.pre_aligned_pct == 0.0


def test_pre_aligned_pct_one_when_all_aligned():
    r = TabularResult(
        accession_number="x",
        pre_tabular_tokens=500,
        pre_narrative_tokens=0,
    )
    assert r.pre_aligned_pct == 1.0


def test_pre_aligned_pct_zero_when_only_p_tokens():
    # Article with only <p> body content — denominator (pre_total) is zero.
    r = TabularResult(accession_number="x", p_tokens=200)
    assert r.pre_aligned_pct == 0.0


def test_negative_tokens_rejected():
    with pytest.raises(Exception):
        TabularResult(accession_number="x", p_tokens=-1)
    with pytest.raises(Exception):
        TabularResult(accession_number="x", pre_tabular_tokens=-1)
    with pytest.raises(Exception):
        TabularResult(accession_number="x", pre_narrative_tokens=-1)


def test_field_order_in_jsonl():
    # Stable field order helps when grepping/diffing sidecars.
    r = TabularResult(
        accession_number="abc",
        p_tokens=1,
        pre_tabular_tokens=2,
        pre_narrative_tokens=3,
    )
    raw = r.model_dump_json()
    import json
    keys = list(json.loads(raw).keys())
    assert keys == [
        "accession_number",
        "p_tokens",
        "pre_tabular_tokens",
        "pre_narrative_tokens",
        "is_tabular_body",
    ]


# ---------------------------------------------------------------------------
# is_tabular_body: derived boolean encoding the filter decision.
# Rule: pre_tabular_tokens > 0 AND >= p_tokens AND >= pre_narrative_tokens
# ("the tabular bucket is the largest, and it's nonzero")
# ---------------------------------------------------------------------------

def test_is_tabular_body_true_for_pure_table():
    r = TabularResult(
        accession_number="x",
        pre_tabular_tokens=500,
        pre_narrative_tokens=0,
    )
    assert r.is_tabular_body is True


def test_is_tabular_body_true_with_small_p_when_tabular_dominates():
    # Relaxed from old strict p == 0 rule: a small <p> intro/dateline
    # doesn't kill the filter as long as the tabular bucket dominates.
    r = TabularResult(
        accession_number="x",
        p_tokens=2,
        pre_tabular_tokens=500,
        pre_narrative_tokens=0,
    )
    assert r.is_tabular_body is True


def test_is_tabular_body_false_when_p_dominates():
    # Real news with a small summary table — <p> content wins, KEEP.
    r = TabularResult(
        accession_number="x",
        p_tokens=600,
        pre_tabular_tokens=200,
        pre_narrative_tokens=50,
    )
    assert r.is_tabular_body is False


def test_is_tabular_body_false_when_no_pre_content():
    # Article with only <p> body or no <pre> at all: not "tabular".
    r = TabularResult(accession_number="x", p_tokens=200)
    assert r.is_tabular_body is False
    r2 = TabularResult(accession_number="x")  # everything zero
    assert r2.is_tabular_body is False


def test_is_tabular_body_false_when_pre_narrative_dominates():
    # Below the tab >= nar majority (49 vs 51).
    r = TabularResult(
        accession_number="x",
        pre_tabular_tokens=49,
        pre_narrative_tokens=51,
    )
    assert r.is_tabular_body is False


def test_is_tabular_body_true_at_pre_majority_boundary():
    # Exactly equal pre_tab and pre_nar — boundary tie goes to tabular.
    r = TabularResult(
        accession_number="x",
        pre_tabular_tokens=50,
        pre_narrative_tokens=50,
    )
    assert r.is_tabular_body is True


def test_is_tabular_body_true_at_p_eq_pre_tab_boundary():
    # Boundary where p_tokens == pre_tabular_tokens: inclusive (>=).
    r = TabularResult(
        accession_number="x",
        p_tokens=50,
        pre_tabular_tokens=50,
        pre_narrative_tokens=0,
    )
    assert r.is_tabular_body is True


def test_is_tabular_body_serializes_in_jsonl():
    import json
    r = TabularResult(
        accession_number="x",
        pre_tabular_tokens=100,
        pre_narrative_tokens=0,
    )
    parsed = json.loads(r.model_dump_json())
    assert parsed["is_tabular_body"] is True


def test_is_tabular_body_roundtrip():
    # Round-trip preserves the boolean (recomputed at deserialization).
    r = TabularResult(
        accession_number="x",
        pre_tabular_tokens=80,
        pre_narrative_tokens=20,
    )
    raw = r.model_dump_json()
    r2 = TabularResult.model_validate_json(raw)
    assert r2.is_tabular_body == r.is_tabular_body
    assert r2.is_tabular_body is True
