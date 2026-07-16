from datetime import datetime

import pytest
from pydantic import ValidationError

from macronews.kg.temporal_schemas import (
    StatementType, TemporalType, RawStatement, RawStatementList,
    TemporalValidityRange, RawTriplet, RawTripletList, TemporalEvent,
)


def test_enum_values():
    assert {s.value for s in StatementType} == {"FACT", "OPINION", "PREDICTION"}
    assert {t.value for t in TemporalType} == {"STATIC", "DYNAMIC", "ATEMPORAL"}


def test_raw_statement_carries_evidence_paragraphs():
    s = RawStatement(evidence_paragraphs=[0, 2], statement="x",
                     statement_type="FACT", temporal_type="STATIC")
    assert s.evidence_paragraphs == [0, 2]


def test_raw_triplet_constrains_vocab():
    t = RawTriplet(subject="Banco Central do Brasil", subject_type="CENTRAL_BANK",
                   relation="RAISES", object="Selic Rate",
                   object_type="INTEREST_RATE", value="to 11%")
    assert t.relation == "RAISES" and t.value == "to 11%"
    with pytest.raises(ValidationError):
        RawTriplet(subject="x", subject_type="NOPE", relation="RAISES",
                   object="y", object_type="INTEREST_RATE")
    with pytest.raises(ValidationError):
        RawTriplet(subject="x", subject_type="CENTRAL_BANK", relation="NOPE",
                   object="y", object_type="INTEREST_RATE")


def test_temporal_event_mutable_no_auto_expired():
    ev = TemporalEvent(article_id="a1", statement="s", statement_type="FACT",
                       temporal_type="DYNAMIC", created_at=datetime(2014, 5, 1),
                       invalid_at=datetime(2014, 6, 1))
    # No set_expired_at validator: constructing with invalid_at must NOT auto-set expired_at.
    assert ev.expired_at is None
    # Mutable: the invalidation pass sets these post-hoc.
    ev.expired_at = ev.created_at
    assert ev.expired_at == datetime(2014, 5, 1)


def test_list_wrappers_default_empty():
    assert RawStatementList().statements == []
    assert RawTripletList().triplets == []


from macronews.config.paths import KG_PROMPTS_DIR  # noqa: E402


def test_pass_prompts_exist_and_mirror_cookbook_wording():
    s = (KG_PROMPTS_DIR / "statement_extraction.txt").read_text()
    assert "Structure statements to clearly show subject-predicate-object" in s
    assert "PRESERVE modal" in s                      # our RAISES-bug addition
    assert "Atemporal" in s and "Prediction" in s
    assert "{{ relevant_asset_groups_block }}" in s   # mapper priming input
    assert "evidence_paragraphs" in s

    t = (KG_PROMPTS_DIR / "temporal_extraction.txt").read_text()
    assert "valid_at" in t and "invalid_at" in t
    assert "publication date" in t
    assert "prediction" in t.lower()

    r = (KG_PROMPTS_DIR / "triplet_extraction.txt").read_text()
    assert "First, NER" in r and "Second, Triplet extraction" in r
    assert "{{ entity_types_block }}" in r and "{{ relation_types_block }}" in r
    assert "Exclude all temporal expressions" in r
