"""Tests for src/kg/runner.py _postprocess_facts (v2)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kg.runner import _postprocess_facts  # noqa: E402
from kg.schemas import KGFact  # noqa: E402


def _f(s, s_t, r, o, o_t, ev):
    return KGFact(
        evidence_paragraphs=ev,
        subject=s, subject_type=s_t,
        relation=r,
        object=o, object_type=o_t,
    )


def test_dedup_merges_triples_unions_evidence():
    facts = [
        _f("Fed", "CENTRAL_BANK", "POSITIVE_IMPACT_ON", "Inflation", "CONCEPT", [18]),
        _f("Fed", "CENTRAL_BANK", "POSITIVE_IMPACT_ON", "Inflation", "CONCEPT", [19]),
        _f("Fed", "CENTRAL_BANK", "POSITIVE_IMPACT_ON", "Inflation", "CONCEPT", [22]),
    ]
    out = _postprocess_facts(facts)
    assert len(out) == 1
    assert out[0].evidence_paragraphs == [18, 19, 22]


def test_dedup_preserves_distinct_triples():
    facts = [
        _f("Fed", "CENTRAL_BANK", "POSITIVE_IMPACT_ON", "Inflation", "CONCEPT", [1]),
        _f("Fed", "CENTRAL_BANK", "POSITIVE_IMPACT_ON", "U.S. Dollar", "CURRENCY", [2]),
        _f("ECB", "CENTRAL_BANK", "CONTROLS", "Eurozone Inflation", "CONCEPT", [3]),
    ]
    out = _postprocess_facts(facts)
    assert len(out) == 3


def test_dedup_unions_overlapping_evidence_without_dupes():
    facts = [
        _f("Trump", "PERSON", "POSITIVE_IMPACT_ON", "Fed", "CENTRAL_BANK", [10, 18]),
        _f("Trump", "PERSON", "POSITIVE_IMPACT_ON", "Fed", "CENTRAL_BANK", [18, 19]),
    ]
    out = _postprocess_facts(facts)
    assert len(out) == 1
    assert out[0].evidence_paragraphs == [10, 18, 19]


def test_strip_self_referential_facts():
    """'Nvidia ANNOUNCES Nvidia' — self-ref, drop."""
    facts = [
        _f("Nvidia", "COMPANY", "ANNOUNCES", "Nvidia", "COMPANY", [1]),     # self-ref
        _f("Fed", "CENTRAL_BANK", "CONTROLS", "Inflation", "CONCEPT", [4]),
        _f("Gold", "COMMODITY", "RELATED_TO", "Gold", "COMMODITY", [9]),     # self-ref
    ]
    out = _postprocess_facts(facts)
    assert len(out) == 1
    assert out[0].subject == "Fed"


def test_strip_leak_with_type_name_as_object():
    """v1 saw leaks like 'Federal Reserve ANNOUNCES InterestRate' where
    the model emitted the type tag as the entity name. v2 codes are
    SCREAMING_SNAKE; the rule still catches them."""
    facts = [
        _f("Federal Reserve", "CENTRAL_BANK", "ANNOUNCES",
           "INTEREST_RATE", "INTEREST_RATE", [8]),                    # leak
        _f("Federal Reserve", "CENTRAL_BANK", "CONTROLS",
           "U.S. Federal Funds Rate", "INTEREST_RATE", [4]),
    ]
    out = _postprocess_facts(facts)
    assert len(out) == 1
    assert out[0].object == "U.S. Federal Funds Rate"


def test_strip_leak_with_type_name_as_subject():
    facts = [
        _f("COMMODITY", "COMMODITY", "POSITIVE_IMPACT_ON",
           "South Africa", "SOVEREIGN", [10]),                       # leak
        _f("Crude Oil", "COMMODITY", "POSITIVE_IMPACT_ON",
           "U.S. CPI Inflation", "ECON_INDICATOR", [6]),
    ]
    out = _postprocess_facts(facts)
    assert len(out) == 1
    assert out[0].subject == "Crude Oil"


def test_leak_strip_covers_all_entity_codes():
    """Every code in the taxonomy is rejected if it appears as an
    entity name."""
    from kg.schemas import ENTITY_TYPES_TUPLE
    for code in ENTITY_TYPES_TUPLE:
        # subject_type and object_type chosen arbitrarily; only the
        # subject/object string matters for the leak check.
        facts = [_f(code, "PERSON", "POSITIVE_IMPACT_ON",
                    "Gold", "COMMODITY", [1])]
        assert _postprocess_facts(facts) == [], \
            f"leak slipped through: {code}"


def test_postprocess_preserves_first_occurrence_order():
    facts = [
        _f("A", "PERSON", "OWNS", "B", "COMPANY", [1]),
        _f("C", "PERSON", "POSITIVE_IMPACT_ON", "D", "INDEX", [2]),
        _f("A", "PERSON", "OWNS", "B", "COMPANY", [3]),  # dup of first
    ]
    out = _postprocess_facts(facts)
    assert len(out) == 2
    assert out[0].subject == "A"
    assert out[0].evidence_paragraphs == [1, 3]
    assert out[1].subject == "C"


def test_postprocess_handles_empty_list():
    assert _postprocess_facts([]) == []


def test_drop_orphan_entities_helper_is_gone():
    """_drop_orphan_entities was deleted — no entities to drop against."""
    import kg.runner as r
    assert not hasattr(r, "_drop_orphan_entities"), \
        "_drop_orphan_entities should no longer be exported"
