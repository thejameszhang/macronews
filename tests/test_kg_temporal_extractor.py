import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kg.temporal_extractor import (  # noqa: E402
    render_statement_user_msg, render_temporal_user_msg,
    render_triplet_user_msg, parse_statements, parse_validity,
    parse_triplets, assemble_events,
)
from kg.temporal_schemas import RawStatement  # noqa: E402


def test_statement_msg_includes_article_and_priming():
    msg = render_statement_user_msg(
        "Fed holds", ["The Fed held rates.", "Markets rose."],
        mapper_context="[RELEVANT ASSET GROUPS] ...", publication_date="2014-05-27")
    assert "[RELEVANT ASSET GROUPS]" in msg
    assert "[0] The Fed held rates." in msg and "[1] Markets rose." in msg
    assert "2014-05-27" in msg


def test_temporal_and_triplet_msgs_are_statement_only():
    s = RawStatement(evidence_paragraphs=[0], statement="The Fed held rates.",
                     statement_type="FACT", temporal_type="STATIC")
    tmsg = render_temporal_user_msg(s, publication_date="2014-05-27")
    assert "The Fed held rates." in tmsg and "2014-05-27" in tmsg
    assert "Markets rose." not in tmsg          # no article leakage
    rmsg = render_triplet_user_msg(s)
    assert "The Fed held rates." in rmsg
    assert "2014-05-27" not in rmsg             # pass 3 gets no metadata


def test_parse_statements_salvages_on_bad_json():
    good = '{"statements":[{"evidence_paragraphs":[0],"statement":"x",' \
           '"statement_type":"FACT","temporal_type":"STATIC"}]}'
    assert len(parse_statements(good, 0).statements) == 1
    assert parse_statements("garbage", 0).statements == []   # WARN, empty


def test_assemble_events_threads_fields():
    s = RawStatement(evidence_paragraphs=[2], statement="x",
                     statement_type="PREDICTION", temporal_type="DYNAMIC")
    from kg.temporal_schemas import TemporalValidityRange, RawTriplet
    vr = TemporalValidityRange(valid_at=datetime(2014, 5, 27), invalid_at=None)
    trips = [RawTriplet(subject="Fed", subject_type="CENTRAL_BANK", relation="RAISES",
                        object="US Federal Funds Rate", object_type="INTEREST_RATE")]
    ev = assemble_events("20140527001", "2014-05-27", [s], [vr], [trips])[0]
    assert ev.article_id == "20140527001"
    assert ev.created_at == datetime.fromisoformat("2014-05-27")
    assert ev.statement_type == "PREDICTION" and ev.temporal_type == "DYNAMIC"
    assert ev.valid_at == datetime(2014, 5, 27) and ev.expired_at is None
    assert ev.evidence_paragraphs == [2] and ev.triplets[0].relation == "RAISES"
