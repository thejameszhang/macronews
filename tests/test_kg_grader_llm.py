import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kg.grading.llm import KGGraderInput, LLMKGGrader  # noqa: E402


def _item():
    return KGGraderInput(
        article_id="a1", headline="Fed hikes",
        paragraphs=["The Fed raised rates.", "Stocks fell.", "Oil rose."],
        subject="Federal Reserve", subject_type="CENTRAL_BANK",
        relation="RAISES", object="U.S. Federal Funds Rate",
        object_type="INTEREST_RATE", evidence_paragraphs=[0],
    )


def test_user_message_renders_indexed_paragraphs_and_fact():
    msg = LLMKGGrader.build_user_message(_item())
    assert "[0] The Fed raised rates." in msg
    assert "[2] Oil rose." in msg
    assert "Federal Reserve" in msg and "CENTRAL_BANK" in msg
    assert "RAISES" in msg
    assert "U.S. Federal Funds Rate" in msg and "INTEREST_RATE" in msg
    assert "0" in msg  # the cited evidence index appears in the FACT block


def test_system_prompt_is_schema_blind():
    # The grader is NOT handed the entity/relation taxonomy (decoupled from the
    # extractor): no placeholders, and no injected type/relation codes.
    g = LLMKGGrader(model_path="/unused")  # no GPU touched in __init__
    assert "{{ENTITY_TYPES}}" not in g.system_prompt
    assert "{{RELATION_TYPES}}" not in g.system_prompt
    assert "CENTRAL_BANK" not in g.system_prompt       # taxonomy NOT injected
    assert "CAUSES_FALL_IN" not in g.system_prompt


def test_grade_batch_empty_returns_empty():
    # Empty input short-circuits before any vLLM import / GPU init.
    g = LLMKGGrader(model_path="/unused")
    assert g.grade_batch([]) == []


def test_user_message_uses_none_when_no_evidence():
    item = _item()
    item.evidence_paragraphs = []
    msg = LLMKGGrader.build_user_message(item)
    assert "(none)" in msg
