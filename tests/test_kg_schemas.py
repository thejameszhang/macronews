"""Tests for src/kg/schemas.py (v2 — FINDKG-style)."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kg.schemas import (  # noqa: E402
    ENTITY_TYPES_TUPLE,
    RELATION_TYPES_TUPLE,
    KGArticleResult,
    KGFact,
)


def test_entity_count_is_18():
    assert len(ENTITY_TYPES_TUPLE) == 18


def test_relation_count_is_18():
    assert len(RELATION_TYPES_TUPLE) == 18


def test_entity_codes_are_screaming_snake():
    """v2: codes are short SCREAMING_SNAKE_CASE (CENTRAL_BANK, GPE), not
    PascalCase. Catches accidental regressions to v1 naming."""
    for code in ENTITY_TYPES_TUPLE:
        assert code == code.upper(), f"{code} is not all-uppercase"
        assert " " not in code, f"{code} contains a space"


def test_specific_entity_codes_present():
    """Spot-check the codes the spec calls out by name."""
    for code in ("PERSON", "CENTRAL_BANK", "GPE", "SOVEREIGN",
                 "CURRENCY", "COMMODITY", "INTEREST_RATE", "INDEX",
                 "GOV_BOND", "FIN_INSTRUMENT", "ECON_INDICATOR", "EVENT"):
        assert code in ENTITY_TYPES_TUPLE


def test_v1_entity_names_removed():
    """v1's PascalCase + finer-grain types must be gone."""
    for old in ("Person", "CentralBank", "Sovereign", "AggregateActor",
                "Product", "TreatyLegislation",
                "PolicyEvent", "GeopoliticalEvent", "MarketEvent", "NaturalEvent"):
        assert old not in ENTITY_TYPES_TUPLE, f"v1 name {old!r} still present"


def test_specific_relations_present():
    """The 18 relations we settled on."""
    for rel in ("OWNS", "HOLDS_POSITION", "IS_MEMBER_OF", "OPERATES_IN",
                "ANNOUNCES", "REPORTS", "PRODUCES", "ACQUIRES", "INVESTS_IN",
                "CONTROLS", "REGULATES", "IMPOSES",
                "RAISES", "DECREASES",
                "CAUSES_RISE_IN", "CAUSES_FALL_IN", "IMPACT",
                "RELATED_TO"):
        assert rel in RELATION_TYPES_TUPLE


def test_v1_relations_removed():
    """v1 had CAUSES, ISSUED_BY, AFFILIATED_WITH, PARTICIPATES_IN — gone.
    Scheme B renamed POSITIVE/NEGATIVE_IMPACT_ON -> CAUSES_RISE_IN/CAUSES_FALL_IN."""
    for old in ("CAUSES", "ISSUED_BY", "AFFILIATED_WITH", "PARTICIPATES_IN",
                "LOWERS", "POSITIVE_IMPACT_ON", "NEGATIVE_IMPACT_ON"):
        assert old not in RELATION_TYPES_TUPLE, f"removed relation {old!r} still present"


def test_kg_fact_field_order_is_cot():
    f = KGFact(
        evidence_paragraphs=[0, 3],
        subject="Federal Reserve",
        subject_type="CENTRAL_BANK",
        relation="RAISES",
        object="U.S. Federal Funds Rate",
        object_type="INTEREST_RATE",
    )
    raw = f.model_dump_json()
    keys = list(json.loads(raw).keys())
    # evidence first as the chain-of-thought anchor.
    assert keys == [
        "evidence_paragraphs",
        "subject", "subject_type",
        "relation",
        "object", "object_type",
    ]


def test_kg_fact_rejects_unknown_entity_type():
    with pytest.raises(Exception):
        KGFact(
            evidence_paragraphs=[0],
            subject="x", subject_type="WALLABY",
            relation="OWNS",
            object="y", object_type="PERSON",
        )


def test_kg_fact_rejects_unknown_relation():
    with pytest.raises(Exception):
        KGFact(
            evidence_paragraphs=[0],
            subject="x", subject_type="PERSON",
            relation="WIBBLES",
            object="y", object_type="PERSON",
        )


def test_kg_article_result_roundtrip():
    r = KGArticleResult(
        facts=[
            KGFact(
                evidence_paragraphs=[3],
                subject="Federal Reserve", subject_type="CENTRAL_BANK",
                relation="RAISES",
                object="U.S. Federal Funds Rate", object_type="INTEREST_RATE",
            ),
        ],
    )
    raw = r.model_dump_json()
    parsed = json.loads(raw)
    # Article-level: only `facts` (no entities).
    assert list(parsed.keys()) == ["facts"]
    r2 = KGArticleResult.model_validate_json(raw)
    assert r2 == r


def test_kg_article_result_defaults_to_empty():
    r = KGArticleResult()
    assert r.facts == []


def test_kg_entity_class_is_gone():
    """KGEntity was dropped — entities are implicit via fact endpoints."""
    import kg.schemas as s
    assert not hasattr(s, "KGEntity"), "KGEntity should no longer be exported"


def test_unlinked_facts_helper_is_gone():
    """No more orphan-fact diagnostic — no entities to orphan against."""
    import kg.schemas as s
    assert not hasattr(s, "unlinked_facts"), \
        "unlinked_facts should no longer be exported"
