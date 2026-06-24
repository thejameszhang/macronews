"""Tests for the KG prompt taxonomy files (entity_types.txt / relation_types.txt)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from config.paths import KG_PROMPTS_DIR  # noqa: E402
from kg.schemas import ENTITY_TYPES_TUPLE, RELATION_TYPES_TUPLE  # noqa: E402


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
