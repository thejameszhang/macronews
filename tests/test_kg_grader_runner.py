import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

from macronews.loaders import load_articles
from macronews.kg.grading.runner import build_statement_tasks, write_sidecar
from macronews.kg.grading.schemas import KGStatementVerdict, TripletVerdict

SPORTS_DIR = REPO / "data" / "sports_news_1994_2000"


def test_sports_loader_returns_standard_schema():
    arts = load_articles(dataset="sports", sample_dir=SPORTS_DIR, max_articles=3)
    assert len(arts) == 3
    for a in arts:
        assert isinstance(a["id"], str)
        assert isinstance(a["paragraphs"], list) and a["paragraphs"]


def test_sports_loader_token_filter_fast_path_and_guard(tmp_path):
    # max_tokens lets the sports loader skip over-long articles; short text stays
    # under the fast-path char limit (no tokenizer), long text needs tokenizer_path.
    from macronews.loaders import load_sports_articles
    (tmp_path / "a.json").write_text(json.dumps(
        {"title": "t", "text": "Short recap. Team won the game."}))
    arts = load_sports_articles(tmp_path, max_tokens=50, tokenizer_path=None)
    assert len(arts) == 1
    (tmp_path / "b.json").write_text(json.dumps({"title": "t", "text": "word " * 200}))
    with pytest.raises(ValueError):  # long article needs a tokenizer; none given
        load_sports_articles(tmp_path, max_tokens=50, tokenizer_path=None)


def _src(aid, paras):
    return {"id": aid, "headline": "h", "paragraphs": paras}


def _event(eid, statement, triplets, stype="FACT", ttype="DYNAMIC", ev=(0,)):
    return {"id": eid, "statement": statement, "statement_type": stype,
            "temporal_type": ttype, "triplets": triplets,
            "evidence_paragraphs": list(ev)}


def _trip(s, r, o, sv="CONCEPT", ov="CONCEPT", value=None):
    return {"subject": s, "subject_type": sv, "relation": r,
            "object": o, "object_type": ov, "value": value}


def test_build_statement_tasks_one_input_per_event():
    rows = [{"article_id": "a1", "date": "2014-05-27", "headline": "h",
             "events": [
                 _event("e1", "Fed raised rates.", [_trip("Fed", "RAISES", "FFR")]),
                 _event("e2", "Yield fell.",
                        [_trip("note", "CAUSES_FALL_IN", "yield", value="to 2.4%")]),
             ]}]
    src = {"a1": _src("a1", ["The Fed raised rates.", "Yield fell.", "Oil rose."])}
    inputs, meta, skipped = build_statement_tasks(rows, src)
    assert len(inputs) == 2 and len(meta) == 2 and not skipped
    assert inputs[0].statement == "Fed raised rates."
    assert inputs[0].statement_type == "FACT"
    assert inputs[0].paragraphs == ["The Fed raised rates.", "Yield fell.", "Oil rose."]
    assert meta[0]["event_id"] == "e1" and meta[0]["temporal_type"] == "DYNAMIC"
    assert meta[1]["triplets"][0]["relation"] == "CAUSES_FALL_IN"


def test_build_statement_tasks_sorted_by_article_id():
    rows = [
        {"article_id": "a2", "events": [_event("e2", "s2", [_trip("x", "IMPACT", "y")])]},
        {"article_id": "a1", "events": [_event("e1", "s1", [_trip("p", "IMPACT", "q")])]},
    ]
    src = {"a1": _src("a1", ["t"]), "a2": _src("a2", ["t"])}
    inputs, meta, _ = build_statement_tasks(rows, src)
    assert [m["article_id"] for m in meta] == ["a1", "a2"]


def test_build_statement_tasks_skips_missing_article():
    rows = [{"article_id": "ghost",
             "events": [_event("e1", "s", [_trip("x", "IMPACT", "y")])]}]
    inputs, meta, skipped = build_statement_tasks(rows, {})
    assert not inputs and len(skipped) == 1
    assert skipped[0]["skip_reason"] == "article_not_in_source"
    assert skipped[0]["event_id"] == "e1"


def test_build_statement_tasks_ignores_eventless_rows():
    rows = [{"article_id": "a1", "events": []}]
    inputs, meta, skipped = build_statement_tasks(rows, {"a1": _src("a1", ["t"])})
    assert not inputs and not meta and not skipped


def _meta(eid, stype="FACT", ttype="DYNAMIC", triplets=None):
    return {"article_id": "a", "date": "", "event_id": eid,
            "statement": "s", "statement_type": stype, "temporal_type": ttype,
            "triplets": triplets or [_trip("s", "IMPACT", "o")],
            "evidence_paragraphs": [0]}


def test_write_sidecar_one_row_per_statement(tmp_path):
    meta = [_meta("e1", triplets=[_trip("note", "REPORTS", "yield", value="2.4%")])]
    verdicts = [KGStatementVerdict(
        macro_relevant=True, supported=True, asserts_direction=True,
        triplets=[TripletVerdict(faithful=False, relation_suggestion="CAUSES_FALL_IN")])]
    out = tmp_path / "g.jsonl"
    summary = write_sidecar(out, meta, verdicts, skipped=[])
    r = json.loads(out.read_text().splitlines()[0])
    assert r["event_id"] == "e1"
    assert r["supported"] is True and r["asserts_direction"] is True
    assert r["triplet_verdicts"][0]["faithful"] is False
    assert r["triplet_verdicts"][0]["relation_suggestion"] == "CAUSES_FALL_IN"
    assert r["triplet_count_mismatch"] is False
    assert summary["statements_graded"] == 1 and summary["triplets_judged"] == 1
    # the lone relation_suggestion is tallied; the two blank slots stay 0
    assert summary["suggestion_counts"] == {
        "relation_suggestion": 1, "subject_type_suggestion": 0,
        "object_type_suggestion": 0}


def test_faithful_rate_directional_is_model_agnostic(tmp_path):
    # A REPORTS triplet on an asserts_direction statement enters the directional
    # denominator; faithful=False lowers faithful_rate_directional.
    meta = [_meta("e1", triplets=[_trip("note", "REPORTS", "yield")]),   # directional stmt
            _meta("e2", triplets=[_trip("x", "RELATED_TO", "y")])]       # non-directional
    verdicts = [
        KGStatementVerdict(asserts_direction=True,
                           triplets=[TripletVerdict(faithful=False)]),
        KGStatementVerdict(asserts_direction=False,
                           triplets=[TripletVerdict(faithful=True)]),
    ]
    summary = write_sidecar(tmp_path / "g.jsonl", meta, verdicts, skipped=[])
    # directional slice = e1 only (1 triplet, faithful False) -> 0.0
    assert summary["faithful_rate_directional"] == 0.0
    assert summary["asserts_direction_rate"] == 0.5
    # overall faithful = 1 of 2
    assert summary["faithful_rate"] == 0.5
    # REPORTS appears in the by-relation breakdown
    assert "REPORTS" in summary["faithful_by_relation"]


def test_write_sidecar_zero_triplet_statement(tmp_path):
    # A statement the triplet pass left undecomposed (a unary level/move fact —
    # "S&P 500 rose 0.6%"). 0 triplets in; the judge may still emit a phantom
    # triplet verdict. It must be dropped, NOT counted as a mismatch (0 in -> 0
    # expected), and the statement-level verdict still counts.
    # build meta inline: _meta's `triplets or [...]` would coerce [] to a default
    meta = [{"article_id": "a", "date": "", "event_id": "e1", "statement": "s",
             "statement_type": "FACT", "temporal_type": "STATIC",
             "triplets": [], "evidence_paragraphs": [0]}]
    verdicts = [KGStatementVerdict(
        macro_relevant=True, supported=True, asserts_direction=True,
        triplets=[TripletVerdict(faithful=True)])]   # phantom verdict
    summary = write_sidecar(tmp_path / "g.jsonl", meta, verdicts, skipped=[])
    r = json.loads((tmp_path / "g.jsonl").read_text().splitlines()[0])
    assert r["triplet_verdicts"] == []
    assert r["triplet_count_mismatch"] is False
    assert r["macro_relevant"] is True and r["asserts_direction"] is True
    assert summary["zero_triplet_statements"] == 1
    assert summary["triplet_count_mismatch"] == 0     # NOT flagged as a mismatch
    assert summary["triplets_judged"] == 0            # contributes 0 triplets
    assert summary["statements_graded"] == 1
    # statement-level rates still see this statement
    assert summary["macro_relevant_rate"] == 1.0
    assert summary["asserts_direction_rate"] == 1.0


def test_write_sidecar_triplet_count_mismatch(tmp_path):
    # verdict has 2 triplet verdicts but the input statement had 1 triplet.
    meta = [_meta("e1", triplets=[_trip("a", "IMPACT", "b")])]
    verdicts = [KGStatementVerdict(
        triplets=[TripletVerdict(faithful=True), TripletVerdict(faithful=False)])]
    summary = write_sidecar(tmp_path / "g.jsonl", meta, verdicts, skipped=[])
    r = json.loads((tmp_path / "g.jsonl").read_text().splitlines()[0])
    assert r["triplet_count_mismatch"] is True
    assert r["triplet_verdicts"] == []            # no fabricated verdicts
    assert summary["triplet_count_mismatch"] == 1
    assert summary["triplets_judged"] == 0        # excluded from denominators


def test_write_sidecar_skipped_rows(tmp_path):
    skipped = [{"article_id": "a1", "event_id": "e9", "statement": "s",
                "skip_reason": "article_not_in_source"}]
    summary = write_sidecar(tmp_path / "g.jsonl", [], [], skipped=skipped)
    r = json.loads((tmp_path / "g.jsonl").read_text().splitlines()[0])
    assert r["grader_verdict_present"] is False
    assert summary["statements_graded"] == 0 and summary["statements_skipped"] == 1


def test_write_sidecar_length_mismatch_raises(tmp_path):
    with pytest.raises(ValueError):
        write_sidecar(tmp_path / "g.jsonl", [{"event_id": "e"}], [], skipped=[])
