import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PROMPT = REPO / "src" / "kg" / "grading" / "prompts" / "grader.txt"


def test_prompt_is_schema_blind_no_placeholders():
    # The grader is NOT handed the taxonomy: no type/relation substitution markers.
    text = PROMPT.read_text()
    assert "{{ENTITY_TYPES}}" not in text
    assert "{{RELATION_TYPES}}" not in text


def test_output_block_lists_fields_in_cot_order():
    # Only the OUTPUT block fixes the JSON field order the model must emit;
    # the rubric above may discuss fields in any order.
    text = PROMPT.read_text()
    out = text[text.index("OUTPUT"):]
    order = [
        "evidence_paragraphs", "macro_relevant",
        "subject_type_suggestion", "relation_suggestion", "object_type_suggestion",
        "correct",
    ]
    positions = [out.find(f) for f in order]
    assert all(p >= 0 for p in positions), "every field must be in the OUTPUT block"
    assert positions == sorted(positions), "OUTPUT must list fields in CoT order"


def test_prompt_does_not_copy_extractor_rules():
    # Independence guard: no verbatim copy of the extractor's inclusion rule.
    extractor = (REPO / "src" / "kg" / "prompts" / "extractor.txt").read_text()
    grader = PROMPT.read_text()
    needle = "acts OUTWARD on the economy"
    assert needle in extractor and needle not in grader
