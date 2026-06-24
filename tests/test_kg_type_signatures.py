import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg.type_signatures import type_violation, RELATION_SIGNATURES  # noqa: E402
from kg.schemas import RELATION_TYPES_TUPLE, ENTITY_TYPES_TUPLE  # noqa: E402


def test_every_relation_has_a_signature():
    assert set(RELATION_SIGNATURES) == set(RELATION_TYPES_TUPLE)


def test_directional_gates_entity_object():
    # an actor/place in the object slot of a directional relation = violation
    assert type_violation("COMMODITY", "CAUSES_RISE_IN", "GPE") is not None
    assert type_violation("COMMODITY", "CAUSES_RISE_IN", "ASSET_METRIC") is None


def test_structural_keeps_entity_object():
    assert type_violation("SOVEREIGN", "IS_MEMBER_OF", "ORG") is None
    assert type_violation("PERSON", "CONTROLS", "COMPANY") is None
    assert type_violation("COMPANY", "OPERATES_IN", "GPE") is None
    assert type_violation("CENTRAL_BANK", "REGULATES", "COMPANY") is None   # AC7 names REGULATES too


def test_raises_interest_rate_is_legal():
    assert type_violation("CENTRAL_BANK", "RAISES", "INTEREST_RATE") is None


def test_subject_is_validated_too():
    # a bare metric cannot be the agent that RAISES something
    assert type_violation("ASSET_METRIC", "RAISES", "INTEREST_RATE") is not None


def test_unknown_relation_fails_open():
    assert type_violation("X", "NOT_A_RELATION", "Y") is None


def test_all_signature_types_are_valid_entity_types():
    valid = set(ENTITY_TYPES_TUPLE)
    for rel, sig in RELATION_SIGNATURES.items():
        for slot, types in sig.items():
            for t in types:
                assert t in valid, f"{rel}/{slot}: unknown type {t!r}"


def test_none_type_is_a_violation():
    # a missing type tag fails CLOSED (a violation), unlike an unknown relation
    assert type_violation(None, "RAISES", "INTEREST_RATE") is not None
