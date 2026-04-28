"""
Group universe loader.

Loads `src/config/group_universe.yaml`, validates structure, and exposes
helpers for the mapper pipeline. Each group has a name, asset_class, and
a list of members. Each member is a dict {name, ticker_symbol}, where
`name` matches an entry in `asset_universe.yaml` exactly.
"""

from pathlib import Path

import yaml

from config.paths import GROUP_UNIVERSE_YAML


def load_group_universe(yaml_path: Path = GROUP_UNIVERSE_YAML) -> dict:
    """Load `group_universe.yaml` and return the full dict, keyed by
    group_key. Raises ValueError if any group is missing required fields
    or any member is not a {name, ticker_symbol} dict."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    for gk, gv in data.items():
        if not isinstance(gv, dict):
            raise ValueError(f"group {gk!r}: expected dict, got {type(gv).__name__}")
        for required in ("name", "asset_class", "members"):
            if required not in gv:
                raise ValueError(f"group {gk!r}: missing key {required!r}")
        for m in gv["members"]:
            if not isinstance(m, dict) or "name" not in m or "ticker_symbol" not in m:
                raise ValueError(
                    f"group {gk!r}: member must be a dict with 'name' and "
                    f"'ticker_symbol', got {m!r}"
                )
    return data


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
