"""Verify group_universe.yaml is structurally consistent with
asset_universe.yaml and the mapper's prompt registry.

Checks:
1. Both YAMLs parse.
2. Every member name in groups exists in asset_universe.yaml.
3. Every asset in asset_universe.yaml is a member of exactly one group.
4. No member appears in two groups.
5. Every group's asset_class is in the recognized set.
6. Every member has a non-empty ticker_symbol.

Exit code 0 if all checks pass, 1 on first batch of failures.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from utils.config import load_asset_universe  # noqa: E402
from utils.groups import load_group_universe, member_index  # noqa: E402

# Asset classes the mapper knows about. Keep in sync with
# ASSET_CLASS_PROMPT_FILES in src/mapping/llm.py.
KNOWN_GROUP_CLASSES = {
    "currency",
    "rates",
    "commodity",
    "US equity sector",
    "equity index",
    "volatility",
}


def main() -> int:
    universe = load_asset_universe()
    groups = load_group_universe()
    universe_names = {
        v["name"] for v in universe.values() if isinstance(v, dict) and "name" in v
    }
    member_names = {m["name"] for gv in groups.values() for m in gv["members"]}

    errors: list[str] = []

    for n in sorted(member_names - universe_names):
        errors.append(f"Group member not in asset_universe.yaml: {n!r}")
    for n in sorted(universe_names - member_names):
        errors.append(f"Asset in universe has no group: {n!r}")

    try:
        member_index(groups)
    except ValueError as e:
        errors.append(f"Member duplicated across groups: {e}")

    for gk, gv in groups.items():
        if gv["asset_class"] not in KNOWN_GROUP_CLASSES:
            errors.append(
                f"Group {gk!r}: unknown asset_class {gv['asset_class']!r} "
                f"(not in KNOWN_GROUP_CLASSES)"
            )
        for m in gv["members"]:
            if not m.get("ticker_symbol"):
                errors.append(
                    f"Group {gk!r}: member {m['name']!r} missing ticker_symbol"
                )

    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1

    multi = sum(1 for g in groups.values() if len(g["members"]) > 1)
    print(
        f"OK: {len(universe_names)} assets, {len(groups)} groups "
        f"({multi} multi-member, {len(groups) - multi} singletons)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
