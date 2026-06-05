import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pipeline import load_articles  # noqa: E402
from kg.grading.runner import build_fact_tasks, write_sidecar  # noqa: E402
from kg.grading.schemas import KGFactVerdict  # noqa: E402

SPORTS_DIR = REPO / "data" / "sports_news_1994_2000"


def test_sports_loader_returns_standard_schema():
    arts = load_articles(dataset="sports", sample_dir=SPORTS_DIR, max_articles=3)
    assert len(arts) == 3
    for a in arts:
        assert isinstance(a["id"], str)
        assert isinstance(a["paragraphs"], list) and a["paragraphs"]


def test_sports_loader_token_filter_fast_path_and_guard(tmp_path):
    # max_tokens lets the sports loader skip over-long articles, the same way
    # the djnw loader does, so a long sports article can't blow the model
    # context. Short text stays under the fast-path char limit and loads with
    # no tokenizer; long text trips the slow path and needs tokenizer_path.
    from loaders import load_sports_articles
    (tmp_path / "a.json").write_text(json.dumps(
        {"title": "t", "text": "Short recap. Team won the game."}))
    arts = load_sports_articles(tmp_path, max_tokens=50, tokenizer_path=None)
    assert len(arts) == 1  # short article: fast path, no tokenizer needed

    (tmp_path / "b.json").write_text(json.dumps({"title": "t", "text": "word " * 200}))
    with pytest.raises(ValueError):  # long article needs a tokenizer; none given
        load_sports_articles(tmp_path, max_tokens=50, tokenizer_path=None)


def _src(aid, paras):
    return {"id": aid, "headline": "h", "paragraphs": paras}


def test_build_fact_tasks_joins_and_emits_one_input_per_fact():
    kg_rows = [{
        "article_id": "a1", "headline": "h",
        "facts": [
            {"evidence_paragraphs": [0], "subject": "Fed", "subject_type": "CENTRAL_BANK",
             "relation": "RAISES", "object": "FFR", "object_type": "INTEREST_RATE"},
            {"evidence_paragraphs": [1], "subject": "FFR", "subject_type": "INTEREST_RATE",
             "relation": "CAUSES_FALL_IN", "object": "S&P 500", "object_type": "INDEX"},
        ],
        "paragraphs": {"0": "The Fed raised rates.", "1": "Stocks fell."},
    }]
    src_by_id = {"a1": _src("a1", ["The Fed raised rates.", "Stocks fell.", "Oil rose."])}
    inputs, meta, skipped = build_fact_tasks(kg_rows, src_by_id)
    assert len(inputs) == 2 and len(meta) == 2 and not skipped
    assert inputs[0].paragraphs == ["The Fed raised rates.", "Stocks fell.", "Oil rose."]
    assert meta[0]["article_id"] == "a1" and meta[0]["date"] == ""
    assert meta[1]["relation"] == "CAUSES_FALL_IN"


def test_build_fact_tasks_skips_missing_article():
    kg_rows = [{"article_id": "ghost", "facts": [
        {"evidence_paragraphs": [0], "subject": "x", "subject_type": "CONCEPT",
         "relation": "IMPACT", "object": "y", "object_type": "CONCEPT"}], "paragraphs": {}}]
    inputs, meta, skipped = build_fact_tasks(kg_rows, {})
    assert not inputs and len(skipped) == 1
    assert skipped[0]["skip_reason"] == "article_not_in_source"


def test_index_alignment_guard_flags_mismatch():
    # Two facts on a misaligned article -> the WHOLE article is dropped (both facts).
    kg_rows = [{"article_id": "a1", "facts": [
        {"evidence_paragraphs": [0], "subject": "x", "subject_type": "CONCEPT",
         "relation": "IMPACT", "object": "y", "object_type": "CONCEPT"},
        {"evidence_paragraphs": [0], "subject": "p", "subject_type": "CONCEPT",
         "relation": "IMPACT", "object": "q", "object_type": "CONCEPT"}],
        "paragraphs": {"0": "STALE TEXT"}}]
    src_by_id = {"a1": _src("a1", ["fresh text", "b"])}
    inputs, meta, skipped = build_fact_tasks(kg_rows, src_by_id)
    assert not inputs
    assert len(skipped) == 2
    assert all(s["skip_reason"] == "paragraph_misalignment" for s in skipped)


def test_articles_with_zero_facts_are_ignored():
    kg_rows = [{"article_id": "a1", "facts": [], "paragraphs": {}}]
    inputs, meta, skipped = build_fact_tasks(kg_rows, {"a1": _src("a1", ["t"])})
    assert not inputs and not meta and not skipped


def test_kg_fact_verdict_has_only_the_six_schema_blind_fields():
    # The redesigned verdict is schema-blind and lean: evidence, macro-relevance,
    # one free-text "better label" suggestion per slot (blank = the label is fine),
    # and a single `correct` bool. No critique, no *_ok bools, no conformance codes.
    assert set(KGFactVerdict.model_fields) == {
        "evidence_paragraphs", "macro_relevant",
        "subject_type_suggestion", "relation_suggestion", "object_type_suggestion",
        "correct"}


def test_write_sidecar_one_row_per_fact_with_tally(tmp_path):
    meta = [{"article_id": "a1", "date": "2022-03-01", "subject": "x",
             "subject_type": "CONCEPT", "relation": "IMPACT", "object": "y",
             "object_type": "CONCEPT", "evidence_paragraphs": [0]}]
    verdicts = [KGFactVerdict(
        evidence_paragraphs=[0], macro_relevant=True,
        subject_type_suggestion="", relation_suggestion="CAUSES_RISE_IN",
        object_type_suggestion="COMMODITY", correct=False)]
    out = tmp_path / "g.jsonl"
    summary = write_sidecar(out, meta, verdicts, skipped=[])
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert r["correct"] is False and r["macro_relevant"] is True
    assert r["relation_suggestion"] == "CAUSES_RISE_IN"   # non-blank suggestion lands
    assert r["object_type_suggestion"] == "COMMODITY"
    assert r["subject_type_suggestion"] == ""             # blank suggestion = label fine
    # every dropped field is gone
    for gone in ("critique", "supported", "relation_ok", "ideal_relation",
                 "non_trivial", "subject_type_ok", "object_type_fix"):
        assert gone not in r
    # the two fail-able bools are tallied; False counts as a fail
    assert summary["graded"] == 1
    assert summary["axis_fail_counts"] == {"macro_relevant": 0, "correct": 1}
    # non-blank suggestions are tallied per slot
    assert summary["suggestion_counts"] == {
        "subject_type": 0, "relation": 1, "object_type": 1}
    assert out.with_suffix(".summary.json").exists()


def test_grader_system_prompt_is_schema_blind():
    # The grader must NOT be handed the entity/relation taxonomy — otherwise an
    # extractor schema edit moves the ruler and it is not a true A/B. The system
    # prompt is grader.txt verbatim, with no type/relation substitution.
    from kg.grading.llm import LLMKGGrader, GRADER_PROMPT_PATH
    g = LLMKGGrader(model_path="/nonexistent-no-vllm-init")
    assert g.system_prompt == GRADER_PROMPT_PATH.read_text()
    assert "{{ENTITY_TYPES}}" not in g.system_prompt
    assert "{{RELATION_TYPES}}" not in g.system_prompt


def test_summary_incorrect_by_relation_and_type(tmp_path):
    # Per-relation / per-type incorrect (correct=False) rates in the summary, so we
    # can see which relations/types the grader most often fails.
    def m(rel, st="EVENT", ot="COMMODITY"):
        return {"article_id": "a", "date": "", "subject": "s", "subject_type": st,
                "relation": rel, "object": "o", "object_type": ot,
                "evidence_paragraphs": []}
    meta = [m("CAUSES_RISE_IN"), m("CAUSES_RISE_IN"), m("CAUSES_RISE_IN"), m("RAISES")]
    correct = [False, False, True, True]
    verdicts = [KGFactVerdict(correct=c) for c in correct]
    summary = write_sidecar(tmp_path / "g.jsonl", meta, verdicts, skipped=[])
    ibr = summary["incorrect_by_relation"]
    assert ibr["CAUSES_RISE_IN"] == {"n": 2, "of": 3, "rate": 0.67}
    assert ibr["RAISES"] == {"n": 0, "of": 1, "rate": 0.0}
    # subject EVENT: 2 of 4 wrong; object COMMODITY: 2 of 4 wrong
    assert summary["incorrect_by_subject_type"]["EVENT"] == {"n": 2, "of": 4, "rate": 0.5}
    assert summary["incorrect_by_object_type"]["COMMODITY"]["n"] == 2


def test_write_sidecar_skipped_rows(tmp_path):
    out = tmp_path / "g.jsonl"
    skipped = [{"article_id": "a1", "skip_reason": "article_not_in_source",
                "subject": "x", "subject_type": "CONCEPT", "relation": "IMPACT",
                "object": "y", "object_type": "CONCEPT", "evidence_paragraphs": []}]
    summary = write_sidecar(out, [], [], skipped=skipped)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["grader_verdict_present"] is False
    assert rows[0]["skip_reason"] == "article_not_in_source"
    assert summary["graded"] == 0 and summary["skipped"] == 1


def test_write_sidecar_length_mismatch_raises(tmp_path):
    with pytest.raises(ValueError):
        write_sidecar(tmp_path / "g.jsonl", [{"article_id": "a"}], [], skipped=[])
