"""Tests for the KG extractor prompt files and rendering (v2)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.paths import KG_PROMPTS_DIR  # noqa: E402
from kg.schemas import ENTITY_TYPES_TUPLE, RELATION_TYPES_TUPLE  # noqa: E402


def test_extractor_template_loads():
    text = (KG_PROMPTS_DIR / "extractor.txt").read_text()
    assert "{{ENTITY_TYPES}}" in text
    assert "{{RELATION_TYPES}}" in text


def test_entity_types_file_lists_every_code():
    """Every code in the schema tuple must appear in the entity_types file
    as a `CODE:` line — that's how the prompt teaches the model the codes."""
    text = (KG_PROMPTS_DIR / "entity_types.txt").read_text()
    for code in ENTITY_TYPES_TUPLE:
        assert f"{code}:" in text, f"entity code {code!r} missing as `CODE:` line"


def test_relation_types_file_lists_every_relation():
    text = (KG_PROMPTS_DIR / "relation_types.txt").read_text()
    for rel in RELATION_TYPES_TUPLE:
        assert rel in text, f"relation {rel!r} missing from relation_types file"


def test_rendered_prompt_substitutes_placeholders():
    from kg.llm import render_system_prompt
    rendered = render_system_prompt()
    assert "{{ENTITY_TYPES}}" not in rendered
    assert "{{RELATION_TYPES}}" not in rendered


def test_rendered_prompt_contains_every_entity_code():
    from kg.llm import render_system_prompt
    rendered = render_system_prompt()
    for code in ENTITY_TYPES_TUPLE:
        assert code in rendered, f"entity code {code!r} missing from rendered prompt"


def test_rendered_prompt_contains_every_relation():
    from kg.llm import render_system_prompt
    rendered = render_system_prompt()
    for rel in RELATION_TYPES_TUPLE:
        assert rel in rendered, f"relation {rel!r} missing from rendered prompt"


def test_rendered_prompt_is_static_for_prefix_cache():
    """vLLM prefix caching requires byte-identical system prompts."""
    from kg.llm import render_system_prompt
    assert render_system_prompt() == render_system_prompt()


def test_rendered_prompt_has_no_per_article_content():
    from kg.llm import render_system_prompt
    rendered = render_system_prompt()
    assert "{{" not in rendered
    assert "}}" not in rendered
    assert "today is" not in rendered.lower()
    assert "article id" not in rendered.lower()


def test_v1_framing_is_gone():
    """v2 dropped: WSJ macroeconomist persona, DO NOT EXTRACT block,
    DIRECTIONALITY block, HARD RULES block, EXAMPLE OUTPUT SHAPE label."""
    from kg.llm import render_system_prompt
    rendered = render_system_prompt()
    for v1_marker in (
        "Wall Street Journal",
        "WSJ macroeconomist",
        "DO NOT EXTRACT",
        "DIRECTIONALITY",
        "HARD RULES FOR CAUSAL RELATIONS",
        "EXAMPLE OUTPUT SHAPE",
        "jurisdiction qualifier",
        "Tautological facts",
        "global-macro frame",
    ):
        assert v1_marker not in rendered, \
            f"v1 framing {v1_marker!r} still present in rendered prompt"


def test_prompt_uses_inline_type_format():
    """v2 spec: each fact has subject_type and object_type as inline fields.
    The prompt must instruct on this format."""
    from kg.llm import render_system_prompt
    rendered = render_system_prompt()
    assert "(subject:type, relation, object:type)" in rendered


def test_prompt_has_one_macro_example():
    """v2 keeps a single FINDKG-style worked example to anchor format."""
    from kg.llm import render_system_prompt
    rendered = render_system_prompt()
    # The Fed example uses these specific anchors.
    assert "Federal Reserve" in rendered
    assert "CENTRAL_BANK" in rendered
    assert "RAISES" in rendered
    assert "CAUSES_RISE_IN" in rendered
    assert "CAUSES_FALL_IN" in rendered


def test_prompt_total_size_is_lean():
    """v1 was ~210 lines across the three files. v2 target: well under 100
    lines combined (after template substitution). Soft cap to catch
    accidental bloat from future edits."""
    from kg.llm import render_system_prompt
    rendered = render_system_prompt()
    line_count = rendered.count("\n")
    assert line_count < 100, \
        f"rendered prompt has {line_count} lines (soft cap 100)"
