import numpy as np
import pytest

from macronews.kg.self_reference import is_self_loop, is_same_asset, SELF_REF_COSINE_THRESHOLD, filter_event
from macronews.kg.type_signatures import DIRECTIONAL_RELATIONS
from macronews.kg.schemas import RELATION_TYPES_TUPLE


def test_directional_relations_value_and_subset():
    assert DIRECTIONAL_RELATIONS == frozenset(
        {"CAUSES_RISE_IN", "CAUSES_FALL_IN", "IMPACT", "RAISES", "DECREASES", "LEAVES_UNCHANGED"})
    assert DIRECTIONAL_RELATIONS <= set(RELATION_TYPES_TUPLE)


def test_self_loop_folds_case_and_whitespace():
    assert is_self_loop("US Dollar / Swiss Franc", "US Dollar / Swiss Franc")
    assert is_self_loop("Brent  Crude", "brent crude")          # whitespace + case
    assert not is_self_loop("Brent Crude Oil", "Brent Crude Price")
    assert not is_self_loop("", "")                              # empty is not a self-loop
    assert not is_self_loop(None, None)


def test_same_asset_cosine():
    a = np.array([1.0, 0.0]); b = np.array([1.0, 0.0]); c = np.array([0.0, 1.0])
    assert is_same_asset(a, b, 0.85) is True       # cosine 1.0
    assert is_same_asset(a, c, 0.85) is False      # cosine 0.0


def test_threshold_default():
    assert SELF_REF_COSINE_THRESHOLD == 0.85


def _ev(triplets):
    return {"id": "e1", "article_id": "a1", "statement": "S", "triplets": triplets}


def _t(subj, obj, rel, st="COMMODITY", ot="ASSET_METRIC", value=None):
    return {"subject": subj, "subject_type": st, "relation": rel,
            "object": obj, "object_type": ot, "value": value}


_EMB = {"X": np.array([1.0, 0.0]), "Y": np.array([0.0, 1.0]), "Z": np.array([1.0, 0.0])}


def test_filter_event_non_lossy_and_keyset():
    ev = _ev([
        _t("X", "X", "DECREASES", st="COMMODITY", ot="COMMODITY"),  # Tier-1 self-loop
        _t("X", "Z", "DECREASES", value="1"),                        # Tier-2 same-asset (cos 1.0)
        _t("X", "Y", "DECREASES"),                                   # cross-asset (cos 0.0) -> kept
    ])
    kept, rejected = filter_event(ev, _EMB, 0.85)
    assert len(kept) == 1 and kept[0]["object"] == "Y"
    assert len(kept) + len(rejected) == 3                            # non-lossy
    assert {r["self_ref_reason"] for r in rejected} == {"self_loop", "same_asset_cosine"}
    assert set(rejected[0]) == {"article_id", "event_id", "statement", "subject", "subject_type",
                                "relation", "object", "object_type", "value", "self_ref_reason", "cosine"}
    sl = next(r for r in rejected if r["self_ref_reason"] == "self_loop")
    sa = next(r for r in rejected if r["self_ref_reason"] == "same_asset_cosine")
    assert sl["cosine"] is None                                      # Tier-1 -> null
    assert sa["cosine"] == 1.0                                       # Tier-2 -> float


def test_tier2_directional_only():
    # same-asset (cos 1.0) but a STRUCTURAL relation -> Tier-2 must NOT drop it
    ev = _ev([_t("X", "Z", "PRODUCES", ot="COMMODITY")])
    kept, rejected = filter_event(ev, _EMB, 0.85)
    assert len(kept) == 1 and rejected == []


def test_tier1_all_relations():
    ev = _ev([_t("X", "X", "IS_MEMBER_OF", st="ORG", ot="ORG")])
    kept, rejected = filter_event(ev, _EMB, 0.85)
    assert kept == [] and rejected[0]["self_ref_reason"] == "self_loop"


def test_missing_embedding_raises():
    ev = _ev([_t("X", "MISSING", "DECREASES")])                      # directional, "MISSING" not in emb
    with pytest.raises(RuntimeError):
        filter_event(ev, _EMB, 0.85)


def test_does_not_mutate_input():
    ev = _ev([_t("X", "X", "DECREASES", st="COMMODITY", ot="COMMODITY")])
    before = len(ev["triplets"])
    filter_event(ev, _EMB, 0.85)
    assert len(ev["triplets"]) == before                            # input untouched
