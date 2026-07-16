"""Runner priming: attach_mapper_context join + 0-flag + missing-row error."""
import pytest
from macronews.kg.runner import attach_mapper_context, load_mapper_rows


def _art(aid, paras=("p0",)):
    return {"id": aid, "headline": "H", "paragraphs": list(paras)}


def test_flagged_article_gets_group_mapper_context():
    arts = [_art("a1")]
    mapper = {"a1": ["Crude Oil"]}            # article_id -> flagged group names (already filtered)
    out = attach_mapper_context(arts, mapper)
    assert "[RELEVANT ASSET GROUPS]" in out[0]["mapper_context"]
    assert "WTI Crude Oil" in out[0]["mapper_context"]   # bare short name (no quotes/arrows)


def test_zero_flag_article_gets_no_groups_mapper_context_not_skipped():
    arts = [_art("a1")]
    mapper = {"a1": []}                        # covered, but 0 groups
    out = attach_mapper_context(arts, mapper)
    assert len(out) == 1                       # NOT skipped
    assert "flagged no asset groups" in out[0]["mapper_context"]


def test_missing_mapper_row_raises():
    arts = [_art("a1"), _art("ghost")]
    mapper = {"a1": []}                        # 'ghost' absent entirely
    with pytest.raises(ValueError, match="no matching mapper row"):
        attach_mapper_context(arts, mapper)


def test_load_mapper_rows_keeps_only_scores_above_threshold(tmp_path):
    """Only tags with relevance_score > 0.5 prime; 0.5 and below are dropped."""
    import json
    f = tmp_path / "m.jsonl"
    f.write_text(
        json.dumps({"article_id": "a1", "groups": ["Crude Oil", "US Equities", "Gold"],
                    "mappings": [
                        {"group": "Crude Oil", "relevance_score": 0.8},
                        {"group": "US Equities", "relevance_score": 0.5},   # dropped (not > 0.5)
                        {"group": "Gold", "relevance_score": 0.3},          # dropped
                    ]}) + "\n"
        + json.dumps({"article_id": "a2", "groups": [], "mappings": []}) + "\n"
    )
    rows = load_mapper_rows(f)
    assert rows == {"a1": ["Crude Oil"], "a2": []}
