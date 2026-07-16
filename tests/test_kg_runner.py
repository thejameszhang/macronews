import json
from datetime import datetime
from macronews.kg.runner import write_sidecar, _clean_triplets, gate_zero_mappings
from macronews.kg.temporal_schemas import TemporalEvent, RawTriplet


def test_gate_zero_mappings_drops_no_mapping_articles():
    """The (unconditional) zero-mapping gate keeps only articles the mapper tagged
    with >=1 group >0.5 (an empty list = no-mapping = dropped)."""
    articles = [{"id": "a1"}, {"id": "a2"}, {"id": "a3"}]
    mapper_rows = {"a1": ["Crude Oil"], "a2": [], "a3": ["Euro", "US Rates"]}
    kept = gate_zero_mappings(articles, mapper_rows)
    assert [a["id"] for a in kept] == ["a1", "a3"]   # a2 (no >0.5 tag) dropped


def test_gate_zero_mappings_missing_row_is_dropped():
    # an article absent from mapper_rows is treated as no-mapping (defensive).
    assert gate_zero_mappings([{"id": "x"}], {}) == []


def test_clean_triplets_drops_typecode_and_selfref():
    trips = [
        RawTriplet(subject="CENTRAL_BANK", subject_type="CENTRAL_BANK", relation="RAISES",
                   object="Selic Rate", object_type="INTEREST_RATE"),   # type-code leak
        RawTriplet(subject="Gold", subject_type="COMMODITY", relation="IMPACT",
                   object="Gold", object_type="COMMODITY"),             # self-ref
        RawTriplet(subject="Fed", subject_type="CENTRAL_BANK", relation="RAISES",
                   object="US Federal Funds Rate", object_type="INTEREST_RATE"),
    ]
    kept = _clean_triplets(trips)
    assert len(kept) == 1 and kept[0].subject == "Fed"


def test_write_sidecar_event_rows(tmp_path):
    art = {"id": "20140527001", "date": "2014-05-27", "headline": "h",
           "paragraphs": ["p0", "p1"]}
    ev = TemporalEvent(article_id="20140527001", statement="The Fed held rates.",
                       statement_type="FACT", temporal_type="STATIC",
                       created_at=datetime(2014, 5, 27), evidence_paragraphs=[0],
                       triplets=[RawTriplet(subject="Fed", subject_type="CENTRAL_BANK",
                                 relation="LEAVES_UNCHANGED", object="US Federal Funds Rate",
                                 object_type="INTEREST_RATE")])
    out = tmp_path / "x.jsonl"
    write_sidecar(out, [art], {"20140527001": [ev]})
    row = json.loads(out.read_text().splitlines()[0])
    assert row["article_id"] == "20140527001" and row["date"] == "2014-05-27"
    assert row["headline"] == "h"
    assert len(row["events"]) == 1
    assert row["events"][0]["statement_type"] == "FACT"
    assert row["events"][0]["triplets"][0]["relation"] == "LEAVES_UNCHANGED"
