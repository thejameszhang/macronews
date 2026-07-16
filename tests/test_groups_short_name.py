"""short_name field: present on every member + globally unique."""
import pytest
from macronews.utils.groups import load_group_universe


def test_every_member_has_nonempty_short_name():
    gu = load_group_universe()
    for gk, gv in gu.items():
        for m in gv["members"]:
            assert m.get("short_name"), f"{gk}/{m['name']} missing short_name"


def test_short_names_globally_unique():
    gu = load_group_universe()
    seen = {}
    for gk, gv in gu.items():
        for m in gv["members"]:
            sn = m["short_name"]
            assert sn not in seen, f"duplicate short_name {sn!r}: {seen.get(sn)} & {gk}"
            seen[sn] = gk


def test_loader_raises_on_missing_short_name(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "g1:\n  name: G1\n  asset_class: commodity\n"
        "  members:\n    - name: A\n      ticker_symbol: 'A'\n"  # no short_name
    )
    with pytest.raises(ValueError, match="short_name"):
        load_group_universe(bad)


def test_loader_raises_on_duplicate_short_name(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "g1:\n  name: G1\n  asset_class: commodity\n  members:\n"
        "    - name: A\n      ticker_symbol: 'A'\n      short_name: Dup\n"
        "g2:\n  name: G2\n  asset_class: commodity\n  members:\n"
        "    - name: B\n      ticker_symbol: 'B'\n      short_name: Dup\n"
    )
    with pytest.raises(ValueError, match="short_name"):
        load_group_universe(bad)


def test_build_group_lookup_maps_short_and_group_names():
    from macronews.utils.groups import build_group_lookup
    lut = build_group_lookup()
    assert lut["S&P 500"] == "us_equities"          # member short_name
    assert lut["US Equities"] == "us_equities"       # group name
    assert lut["Brent Crude Oil"] == "crude_oil"
    assert lut["Crude Oil"] == "crude_oil"


def test_build_group_lookup_raises_on_conflicting_collision(tmp_path):
    from macronews.utils.groups import build_group_lookup, load_group_universe
    bad = tmp_path / "bad.yaml"
    # member short_name "Gold" collides with a DIFFERENT group's name "Gold"
    bad.write_text(
        "metals:\n  name: Metals\n  asset_class: commodity\n  members:\n"
        "    - name: Gold Bar\n      ticker_symbol: 'GC'\n      short_name: Gold\n"
        "gold:\n  name: Gold\n  asset_class: commodity\n  members:\n"
        "    - name: Gold Coin\n      ticker_symbol: 'GX'\n      short_name: Gold Coin\n"
    )
    with pytest.raises(ValueError, match="collision"):
        build_group_lookup(load_group_universe(bad))


def test_constituents_with_short_names():
    from macronews.utils.groups import constituents_with_short_names
    pairs = constituents_with_short_names("crude_oil")
    assert pairs == [("WTI Crude Oil", "WTI Crude Oil"),
                     ("Brent Crude Oil", "Brent Crude Oil")]
    # Verify a group where short_name differs from full name
    assert ("S&P 500 Index", "S&P 500") in constituents_with_short_names("us_equities")


def test_constituents_unknown_group_raises():
    from macronews.utils.groups import constituents_with_short_names
    with pytest.raises(KeyError):
        constituents_with_short_names("not_a_group")
