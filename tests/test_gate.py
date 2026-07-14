"""Keyword pre-filter gate: match semantics, and the real universe compiles.

The saving/recall figures in documentation/mapper were measured with the
alphanumeric-boundary predicate these tests pin. Changing the predicate
invalidates those numbers, so the boundary cases are asserted explicitly.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pytest  # noqa: E402
from mapping.gate import compile_gate, gate_text  # noqa: E402
from utils.groups import load_group_universe  # noqa: E402


def _pat(keywords: list[str]):
    return compile_gate({"g": {"keywords": keywords}})["g"]


def test_matches_whole_word_case_insensitively():
    p = _pat(["oil", "crude"])
    assert p.search("Oil prices rose")
    assert p.search("the CRUDE market")


def test_does_not_match_inside_a_longer_word():
    """The boundary is alphanumeric, so 'oil' must not fire on 'turmoil'."""
    p = _pat(["oil"])
    assert not p.search("political turmoil in the region")
    assert not p.search("she oiled the hinge")


def test_punctuation_is_a_boundary():
    p = _pat(["oil"])
    assert p.search("Brent (oil) settled lower")
    assert p.search("oil, gold and copper")


def test_multi_word_keyword():
    p = _pat(["federal reserve"])
    assert p.search("The Federal Reserve said Tuesday")
    assert not p.search("a federal agency reserve fund")


def test_regex_metacharacters_are_escaped():
    """'s&p 500' is a real keyword; a keyword must never act as a pattern."""
    p = _pat(["s&p 500"])
    assert p.search("the S&P 500 closed up")
    p_dot = _pat(["u.s."])
    assert p_dot.search("u.s. growth")
    assert not p_dot.search("uxsx growth")


def test_gate_text_includes_the_headline():
    """A keyword that appears only in the headline must still fire."""
    text = gate_text("Gold hits record", ["The metal advanced on haven demand."])
    assert _pat(["gold"]).search(text)


def test_gate_text_includes_the_body():
    text = gate_text("Metals roundup", ["Copper fell while gold advanced."])
    assert _pat(["gold"]).search(text)


def test_empty_keywords_is_a_hard_error():
    """Fail loud: a group with no keywords would silently gate to nothing."""
    with pytest.raises(ValueError, match="keywords"):
        compile_gate({"g": {"keywords": []}})
    with pytest.raises(ValueError, match="keywords"):
        compile_gate({"g": {}})


def test_real_universe_compiles_for_every_group():
    universe = load_group_universe()
    gate = compile_gate(universe)
    assert set(gate) == set(universe)
    assert len(gate) == 50


def test_real_universe_gates_a_realistic_article():
    """An oil article reaches Crude Oil and Energy; it must not reach Cocoa."""
    gate = compile_gate()
    text = gate_text(
        "Oil Prices Climb on Supply Concerns",
        ["Crude futures rose after the cartel signalled deeper output cuts."],
    )
    fired = {gk for gk, p in gate.items() if p.search(text)}
    assert "crude_oil" in fired
    assert "energy_sector" in fired
    assert "cocoa" not in fired
    assert len(fired) < 50, "gate must skip at least one group or it saves nothing"
