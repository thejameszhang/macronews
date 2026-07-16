"""
Group universe loader.

Loads `src/macronews/config/group_universe.yaml`, validates structure, and exposes
helpers for the mapper pipeline. Each group has a name, asset_class, and
a list of members. Each member is a dict {name, ticker_symbol, short_name}, where
`name` matches an entry in `asset_universe.yaml` exactly.
"""

from pathlib import Path

import yaml

from macronews.config.paths import GROUP_UNIVERSE_YAML


def load_group_universe(yaml_path: Path = GROUP_UNIVERSE_YAML) -> dict:
    """Load `group_universe.yaml` and return the full dict, keyed by
    group_key. Raises ValueError if any group is missing required fields
    or any member is not a {name, ticker_symbol, short_name} dict."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    all_short: dict[str, str] = {}
    for gk, gv in data.items():
        if not isinstance(gv, dict):
            raise ValueError(f"group {gk!r}: expected dict, got {type(gv).__name__}")
        for required in ("name", "asset_class", "members"):
            if required not in gv:
                raise ValueError(f"group {gk!r}: missing key {required!r}")
        for m in gv["members"]:
            if (not isinstance(m, dict) or "name" not in m
                    or "ticker_symbol" not in m or "short_name" not in m):
                raise ValueError(
                    f"group {gk!r}: member must be a dict with 'name', "
                    f"'ticker_symbol', and 'short_name', got {m!r}"
                )
            sn = m["short_name"]
            if sn in all_short:
                raise ValueError(
                    f"duplicate short_name {sn!r} in {all_short[sn]!r} and {gk!r}"
                )
            all_short[sn] = gk
    return data


def build_group_lookup(universe: dict | None = None) -> dict[str, str]:
    """Map every member `short_name` AND every group `name` to its group_key.

    Used to recover the asset group from an entity string the extractor
    emitted. Raises if two distinct group_keys claim the same key string
    (e.g. a member short_name accidentally equal to another group's name).
    """
    if universe is None:
        universe = load_group_universe()
    lut: dict[str, str] = {}

    def _add(key: str, gk: str) -> None:
        if key in lut and lut[key] != gk:
            raise ValueError(
                f"lookup collision: {key!r} maps to both {lut[key]!r} and {gk!r}"
            )
        lut[key] = gk

    for gk, gv in universe.items():
        _add(gv["name"], gk)
        for m in gv["members"]:
            _add(m["short_name"], gk)
    return lut


def constituents_with_short_names(
    group_key: str, universe: dict | None = None
) -> list[tuple[str, str]]:
    """(full name, short_name) for each member of a group, in YAML order."""
    if universe is None:
        universe = load_group_universe()
    return [(m["name"], m["short_name"]) for m in universe[group_key]["members"]]


def group_keys(universe: dict | None = None) -> list[str]:
    """Sorted list of group keys (deterministic iteration order)."""
    if universe is None:
        universe = load_group_universe()
    return sorted(universe.keys())


def member_index(universe: dict | None = None) -> dict[str, str]:
    """Reverse lookup: asset name (matching `asset_universe.yaml`) -> group_key.
    Raises if any asset name appears in two groups."""
    if universe is None:
        universe = load_group_universe()
    index: dict[str, str] = {}
    for gk, gv in universe.items():
        for m in gv["members"]:
            n = m["name"]
            if n in index:
                raise ValueError(
                    f"asset {n!r} appears in groups {index[n]!r} and {gk!r}"
                )
            index[n] = gk
    return index


def ticker_index(universe: dict | None = None) -> dict[str, str]:
    """Asset name -> ticker_symbol lookup."""
    if universe is None:
        universe = load_group_universe()
    return {
        m["name"]: m["ticker_symbol"]
        for gv in universe.values()
        for m in gv["members"]
    }
