from macronews.kg.grading.llm import KGGraderInput, LLMKGGrader


def _item():
    return KGGraderInput(
        article_id="a1", headline="Treasury yields fall",
        paragraphs=["The 10-year yield fell to 2.46%.", "Stocks rose.", "Oil rose."],
        statement="The 10-year Treasury yield fell to 2.46%.",
        statement_type="FACT",
        triplets=[
            {"subject": "10-Year US Treasury Note", "subject_type": "INTEREST_RATE",
             "relation": "CAUSES_FALL_IN", "object": "yield",
             "object_type": "INTEREST_RATE", "value": "to 2.46%"},
        ],
        evidence_paragraphs=[0],
    )


def test_user_message_static_first_layout():
    msg = LLMKGGrader.build_user_message(_item())
    # full article appears BEFORE any statement-specific content (prefix caching)
    assert msg.index("[ARTICLE]") < msg.index("[STATEMENT]")
    assert msg.index("[/ARTICLE]") < msg.index("[STATEMENT]")
    assert "[0] The 10-year yield fell to 2.46%." in msg
    assert "[2] Oil rose." in msg


def test_user_message_renders_statement_type_and_triplet_value():
    msg = LLMKGGrader.build_user_message(_item())
    assert "(FACT)" in msg
    assert "The 10-year Treasury yield fell to 2.46%." in msg
    assert "CAUSES_FALL_IN" in msg
    assert "10-Year US Treasury Note" in msg and "INTEREST_RATE" in msg
    assert "to 2.46%" in msg          # the triplet value is rendered
    assert "1." in msg                # triplets are numbered


def test_user_message_evidence_line():
    msg = LLMKGGrader.build_user_message(_item())
    assert "Extractor-cited evidence paragraphs: 0" in msg   # the evidence line, not "[0]"
    item = _item(); item.evidence_paragraphs = []
    assert "(none)" in LLMKGGrader.build_user_message(item)


def test_triplet_without_value_omits_equals():
    item = _item()
    item.triplets = [{"subject": "Fed", "subject_type": "CENTRAL_BANK",
                      "relation": "LEAVES_UNCHANGED", "object": "rate",
                      "object_type": "INTEREST_RATE", "value": None}]
    msg = LLMKGGrader.build_user_message(item)
    assert "LEAVES_UNCHANGED" in msg
    assert " = " not in msg.split("[TRIPLETS]")[1]   # no value -> no "= ..."


def test_system_prompt_is_schema_blind():
    g = LLMKGGrader(model_path="/unused")  # no GPU touched in __init__
    assert "{{ENTITY_TYPES}}" not in g.system_prompt
    assert "CENTRAL_BANK" not in g.system_prompt
    assert "CAUSES_FALL_IN" not in g.system_prompt


def test_grade_batch_empty_returns_empty():
    g = LLMKGGrader(model_path="/unused")
    assert g.grade_batch([]) == []
