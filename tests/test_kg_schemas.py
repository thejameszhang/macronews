"""Tests for src/kg/schemas.py (v2 — FINDKG-style)."""

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


def test_entity_count_is_19():
    assert len(ENTITY_TYPES_TUPLE) == 19


def test_relation_count_is_15():
    assert len(RELATION_TYPES_TUPLE) == 15


def test_leaves_unchanged_present():
    assert "LEAVES_UNCHANGED" in RELATION_TYPES_TUPLE


def test_entity_codes_are_screaming_snake():
    """v2: codes are short SCREAMING_SNAKE_CASE (CENTRAL_BANK, GPE), not
    PascalCase. Catches accidental regressions to v1 naming."""
    for code in ENTITY_TYPES_TUPLE:
        assert code == code.upper(), f"{code} is not all-uppercase"
        assert " " not in code, f"{code} contains a space"


def test_specific_entity_codes_present():
    """Spot-check the codes the spec calls out by name."""
    for code in ("PERSON", "CENTRAL_BANK", "GPE", "SOVEREIGN",
                 "CURRENCY", "COMMODITY", "INTEREST_RATE", "EQUITY_INDEX",
                 "US_GICS_SECTOR", "GOV_BOND", "FIN_INSTRUMENT",
                 "ECON_INDICATOR", "ASSET_METRIC", "EVENT"):
        assert code in ENTITY_TYPES_TUPLE


def test_v1_entity_names_removed():
    """v1's PascalCase + finer-grain types must be gone."""
    for old in ("Person", "CentralBank", "Sovereign", "AggregateActor",
                "Product", "TreatyLegislation",
                "PolicyEvent", "GeopoliticalEvent", "MarketEvent", "NaturalEvent"):
        assert old not in ENTITY_TYPES_TUPLE, f"v1 name {old!r} still present"


def test_specific_relations_present():
    """The 15 relations we settled on."""
    for rel in ("IS_MEMBER_OF", "OPERATES_IN",
                "ANNOUNCES", "REPORTS", "FORECASTS", "PRODUCES",
                "CONTROLS", "REGULATES", "IMPOSES",
                "RAISES", "DECREASES", "LEAVES_UNCHANGED",
                "CAUSES_RISE_IN", "CAUSES_FALL_IN", "IMPACT"):
        assert rel in RELATION_TYPES_TUPLE


def test_v1_relations_removed():
    """v1 had CAUSES, ISSUED_BY, AFFILIATED_WITH, PARTICIPATES_IN — gone.
    Scheme B renamed POSITIVE/NEGATIVE_IMPACT_ON -> CAUSES_RISE_IN/CAUSES_FALL_IN.
    2026-06 pruned firm-micro (OWNS, HOLDS_POSITION, ACQUIRES, INVESTS_IN) + the
    contentless RELATED_TO catch-all."""
    for old in ("CAUSES", "ISSUED_BY", "AFFILIATED_WITH", "PARTICIPATES_IN",
                "LOWERS", "POSITIVE_IMPACT_ON", "NEGATIVE_IMPACT_ON",
                "OWNS", "HOLDS_POSITION", "ACQUIRES", "INVESTS_IN", "RELATED_TO"):
        assert old not in RELATION_TYPES_TUPLE, f"removed relation {old!r} still present"


def test_kg_fact_rejects_unknown_entity_type():
    with pytest.raises(Exception):
        KGFact(
            evidence_paragraphs=[0],
            subject="x", subject_type="WALLABY",
            relation="CONTROLS",
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


def test_structural_fact_is_valid():
    KGFact(subject="A", subject_type="COMPANY", relation="CONTROLS",
           object="B", object_type="COMPANY")  # constructs without error
