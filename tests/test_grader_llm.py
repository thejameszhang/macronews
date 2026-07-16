"""Tests for src/mapping/grading/llm.py — prompt construction only (no model)."""

from macronews.mapping.grading.llm import GraderInput, LLMGrader


def make_input(**overrides):
    base = dict(
        article_id="X1",
        headline="Test headline",
        paragraphs=["First paragraph.", "Second paragraph.", "Third paragraph."],
        group_name="Crude Oil",
        asset_class="commodity",
        member_names=["WTI Crude Oil", "Brent Crude Oil"],
        mapper_score=0.85,
        mapper_evidence_paragraphs=[0, 2],
    )
    base.update(overrides)
    return GraderInput(**base)


def test_user_message_renders_mapper_score():
    msg = LLMGrader.build_user_message(make_input(mapper_score=0.42))
    assert "Mapper relevance score: 0.42" in msg


def test_user_message_has_required_blocks():
    msg = LLMGrader.build_user_message(make_input())
    assert "[HEADLINE] Test headline" in msg
    assert "[ARTICLE]\n" in msg
    assert "[/ARTICLE]" in msg
    assert "[MAPPER_CLAIM]" in msg
    assert "[/MAPPER_CLAIM]" in msg
    assert "[MAPPER_EVIDENCE]" in msg


def test_user_message_indexes_paragraphs():
    msg = LLMGrader.build_user_message(make_input())
    assert "[0] First paragraph." in msg
    assert "[1] Second paragraph." in msg
    assert "[2] Third paragraph." in msg


def test_user_message_includes_group_metadata():
    msg = LLMGrader.build_user_message(make_input())
    assert "Asset group: Crude Oil" in msg
    assert "Asset class: commodity" in msg
    assert "Members: WTI Crude Oil, Brent Crude Oil" in msg


def test_user_message_renders_evidence_indices():
    msg = LLMGrader.build_user_message(make_input(mapper_evidence_paragraphs=[0, 2]))
    assert "0, 2" in msg


def test_user_message_handles_empty_evidence():
    msg = LLMGrader.build_user_message(make_input(mapper_evidence_paragraphs=[]))
    # Empty evidence is rendered with an explanatory placeholder, not blank
    assert "(none" in msg


def test_user_message_handles_singleton_group():
    msg = LLMGrader.build_user_message(
        make_input(group_name="Natural Gas", member_names=["Natural Gas"])
    )
    assert "Members: Natural Gas" in msg
