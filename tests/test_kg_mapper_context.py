"""render_mapper_context: flagged-groups form + 0-flag form, deterministic order."""
import pytest
from macronews.kg.llm import render_mapper_context


def test_mapper_context_lists_flagged_groups_with_short_names():
    p = render_mapper_context(["US Equities", "Crude Oil"])
    norm = " ".join(p.split())
    assert "[RELEVANT ASSET GROUPS]" in p
    assert "A prior step flagged this article as relevant" in norm
    # group header + Option A constituent format (official name — use canonical)
    assert "- Asset Group: US Equities | Asset Class: equity index" in p
    assert "S&P 500 Index — use S&P 500" in p
    # same-name constituent shows the bare name (no "— use" clause)
    assert "    Brent Crude Oil\n" in p + "\n"
    # no arrow symbols anywhere
    assert "→" not in p
    # breadth license + veto present
    assert "general macroeconomic facts" in norm
    assert "not actually relevant" in norm


def test_mapper_context_is_deterministic_group_key_order():
    # Crude Oil (crude_oil) sorts before US Equities (us_equities) by group_key
    p = render_mapper_context(["US Equities", "Crude Oil"])
    assert p.index("Crude Oil |") < p.index("US Equities |")


def test_mapper_context_zero_flags_is_no_groups_note():
    p = render_mapper_context([])
    norm = " ".join(p.split())
    assert "[RELEVANT ASSET GROUPS]" in p
    assert "flagged no asset groups" in norm
    assert "general macroeconomic facts" in norm
    # no constituent arrows in the empty form
    assert "→" not in p


def test_mapper_context_unknown_group_name_raises():
    with pytest.raises(KeyError):
        render_mapper_context(["Not A Group"])
