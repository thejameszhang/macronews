"""KG LLM helpers: mapper-context renderer for per-article primed extraction."""

from __future__ import annotations

import logging
import textwrap

from macronews.utils.groups import (
    build_group_lookup, constituents_with_short_names, load_group_universe,
)

logger = logging.getLogger(__name__)

# Cache the universe + lookup ONCE at import. render_mapper_context runs per article;
# load_group_universe() opens the YAML on every call, so calling it per article
# would be tens of thousands of NFS reads per run.
_GROUP_UNIVERSE = load_group_universe()
_GROUP_LOOKUP = build_group_lookup(_GROUP_UNIVERSE)

_CONTEXT_HEADER = "[RELEVANT ASSET GROUPS]"
_BREADTH = (
    "extract the general macroeconomic facts in it — events, policies, "
    "indicators, and other forces at play."
)


def render_mapper_context(flagged_group_names: list[str]) -> str:
    """Per-article mapper-context block (goes at the top of the user message).

    flagged_group_names: the group *names* the mapper flagged for this
    article (may be empty → the no-groups form). Raises KeyError on an
    unknown group name (alignment bug — fail loud).
    """
    if not flagged_group_names:
        return textwrap.fill(
            f"{_CONTEXT_HEADER} A prior step flagged no asset groups for this "
            f"article; {_BREADTH}",
            width=80,
        )
    keys = sorted({_GROUP_LOOKUP[name] for name in flagged_group_names})  # group_key order
    lines = [
        textwrap.fill(
            f"{_CONTEXT_HEADER} A prior step flagged this article as relevant to "
            f"the following asset groups and their constituent assets:",
            width=80,
        )
    ]
    for gk in keys:
        gv = _GROUP_UNIVERSE[gk]
        lines.append(
            f"- Asset Group: {gv['name']} | Asset Class: {gv['asset_class']}"
        )
        for full, short in constituents_with_short_names(gk, _GROUP_UNIVERSE):
            if full == short:
                lines.append(f"    {full}")
            else:
                lines.append(f"    {full} — use {short}")
    lines.append(textwrap.fill(
        "Use these as a guide for some of your extracted facts. If you judge "
        "that one of these groups is not actually relevant to this article, you "
        "need not include it in any fact. Beyond these groups, also " + _BREADTH,
        width=80,
    ))
    return "\n".join(lines)
