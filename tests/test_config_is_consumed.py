"""No key in group_universe.yaml is orphaned -- every one has a consumer in src/.

This catches a key that nothing reads at all. It does NOT catch a key that the wrong
lane reads: the mapper's keyword gate sat unwired for a month while the KG's
link_groups.py was reading `keywords` the whole time, so a check like this was green
throughout. The gate's wiring is pinned behaviourally by test_gate_wiring.py.
"""
from pathlib import Path

from macronews.utils.groups import load_group_universe

SRC = Path(__file__).resolve().parents[1] / "src" / "macronews"


def test_no_group_universe_key_is_orphaned():
    keys = {k for group in load_group_universe().values() for k in group}
    source = "\n".join(p.read_text() for p in SRC.rglob("*.py"))
    unread = {k for k in keys if f'"{k}"' not in source and f"'{k}'" not in source}

    assert not unread, (
        f"keys in group_universe.yaml that nothing in src/ reads: {sorted(unread)}"
    )
