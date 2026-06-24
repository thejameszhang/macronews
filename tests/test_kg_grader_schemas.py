"""Tests for src/kg/grading/schemas.py (statement-level verdict)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kg.grading.schemas import KGStatementVerdict, TripletVerdict  # noqa: E402


def test_statement_verdict_field_order_is_cot():
    # CoT order: read -> macro-relevance -> statement truth -> directionality ->
    # per-triplet faithfulness last.
    assert list(KGStatementVerdict.model_fields.keys()) == [
        "evidence_paragraphs", "macro_relevant", "supported",
        "asserts_direction", "triplets",
    ]


def test_triplet_verdict_fields():
    assert list(TripletVerdict.model_fields.keys()) == [
        "faithful", "relation_suggestion",
        "subject_type_suggestion", "object_type_suggestion",
    ]


def test_defaults():
    v = KGStatementVerdict()
    assert v.evidence_paragraphs == []
    assert v.macro_relevant is True
    assert v.supported is True
    assert v.asserts_direction is False
    assert v.triplets == []
    t = TripletVerdict()
    assert t.faithful is True
    assert t.relation_suggestion == ""
    assert t.subject_type_suggestion == ""
    assert t.object_type_suggestion == ""


def test_nested_validation_from_json():
    v = KGStatementVerdict.model_validate_json(
        '{"evidence_paragraphs":[0,2],"macro_relevant":true,"supported":true,'
        '"asserts_direction":true,"triplets":[{"faithful":false,'
        '"relation_suggestion":"CAUSES_FALL_IN","subject_type_suggestion":"",'
        '"object_type_suggestion":"INTEREST_RATE"}]}'
    )
    assert v.asserts_direction is True
    assert len(v.triplets) == 1
    assert v.triplets[0].faithful is False
    assert v.triplets[0].relation_suggestion == "CAUSES_FALL_IN"


def test_dropped_fields_not_reintroduced():
    fields = set(KGStatementVerdict.model_fields)
    for gone in ("correct", "modality_suggestion", "score", "critique",
                 "relation_ok", "non_trivial", "ideal_relation"):
        assert gone not in fields
    # the per-triplet verdict must not re-grow the old per-fact fields either
    triplet_fields = set(TripletVerdict.model_fields)
    for gone in ("modality_suggestion", "correct"):
        assert gone not in triplet_fields
