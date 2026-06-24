import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PROMPT = REPO / "src" / "kg" / "grading" / "prompts" / "grader.txt"


def test_prompt_is_schema_blind_no_placeholders():
    text = PROMPT.read_text()
    assert "{{ENTITY_TYPES}}" not in text
    assert "{{RELATION_TYPES}}" not in text
    # no taxonomy injected
    assert "CENTRAL_BANK" not in text
    assert "CAUSES_FALL_IN" not in text


def test_output_json_lists_fields_in_cot_order():
    # Search the QUOTED field names so we only hit the JSON example at the bottom
    # (the rubric prose mentions "triplets"/"faithful" unquoted, out of order).
    text = PROMPT.read_text()
    order = ["evidence_paragraphs", "macro_relevant", "supported",
             "asserts_direction", "triplets", "faithful", "relation_suggestion"]
    positions = [text.find(f'"{f}"') for f in order]
    assert all(p >= 0 for p in positions), "every field must appear in the JSON example"
    assert positions == sorted(positions), "JSON example must list fields in CoT order"


def test_prompt_states_the_drops_direction_rule():
    text = PROMPT.read_text().lower()
    assert "drops a direction" in text or "directionless" in text


def test_prompt_drops_old_fields():
    text = PROMPT.read_text()
    assert "modality_suggestion" not in text
    assert '"correct"' not in text and "correct:" not in text
