"""Tests for src/kg/grading/schemas.py."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kg.grading.schemas import KGFactVerdict  # noqa: E402


def test_field_order_is_cot():
    # Schema-blind CoT order: ground first, macro-relevance, a per-slot "better
    # label" suggestion (blank = fine), then the headline `correct` verdict last.
    assert list(KGFactVerdict.model_fields.keys()) == [
        "evidence_paragraphs", "macro_relevant",
        "subject_type_suggestion", "relation_suggestion", "object_type_suggestion",
        "correct",
    ]


def test_types_and_validation():
    v = KGFactVerdict.model_validate_json(
        '{"evidence_paragraphs":[0,2],"macro_relevant":true,'
        '"subject_type_suggestion":"","relation_suggestion":"CAUSES_FALL_IN",'
        '"object_type_suggestion":"COMMODITY","correct":false}'
    )
    assert v.evidence_paragraphs == [0, 2]
    assert v.relation_suggestion == "CAUSES_FALL_IN"
    assert v.object_type_suggestion == "COMMODITY"
    assert v.correct is False


def test_suggestion_fields_default_empty():
    v = KGFactVerdict()
    assert v.subject_type_suggestion == ""
    assert v.relation_suggestion == ""
    assert v.object_type_suggestion == ""


def test_dropped_fields_not_reintroduced():
    # These belonged to the old coupled, omnibus verdict; keep them gone.
    fields = set(KGFactVerdict.model_fields)
    for gone in ("score", "critique", "supported", "non_trivial",
                 "subject_type_ok", "relation_ok", "ideal_relation"):
        assert gone not in fields
