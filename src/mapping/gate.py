"""
Keyword pre-filter gate for the article -> asset-group mapper.

Skips the LLM call for an (article, group) pair when the article contains none
of that group's keywords. The gate is recall-optimized: a false positive costs
one LLM call (the model still rejects the pair), while a false negative loses
the tag for good.

The gate is a production cost optimization and runs on djnw only; see
pipeline.gate_default. Keywords live in `src/config/group_universe.yaml` and are
also read by `src/kg/link_groups.py`, where a false positive is NOT free -- it
mints a spurious entity->asset-group edge with no model in the loop to reject
it. Broaden a keyword for gate recall and you narrow KG link precision.
"""

import re

from utils.groups import load_group_universe


def _boundary(body: str) -> re.Pattern:
    """Alphanumeric-boundary, case-insensitive match.

    The gate's measured figures are tied to this exact construction: on 2014-05c
    it skips 88% of calls. Change it and that no longer describes this code.

    Two things it is easy to get wrong about that number. Calls saved is not cost
    saved -- the prompt puts the article first so all 50 groups share a prefix, so
    skipping calls does not skip the article's prefill; the same run cut calls
    8.5x but GPU time only 3.4x. And the recall cost is still being measured
    against current weights.
    """
    return re.compile(rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])", re.IGNORECASE)


def compile_gate(universe: dict | None = None) -> dict[str, re.Pattern]:
    """group_key -> one compiled alternation over that group's keywords."""
    if universe is None:
        universe = load_group_universe()
    gate: dict[str, re.Pattern] = {}
    for gk, gv in universe.items():
        keywords = gv.get("keywords")
        if not keywords:
            raise ValueError(
                f"group {gk!r}: the keyword gate requires a non-empty 'keywords' list"
            )
        gate[gk] = _boundary("|".join(re.escape(k) for k in keywords))
    return gate


def gate_text(headline: str, paragraphs: list[str]) -> str:
    """The article content the gate matches: headline + paragraphs.

    Deliberately NOT the full prompt the model sees. That prompt ends in
    `[ASSET_GROUP] {group name} | ...`, so matching against it would make every
    group's own name a keyword hit, fire the gate on all 50 groups for every
    article, and turn the gate into a no-op at full cost.
    """
    return (headline or "") + "\n" + "\n".join(paragraphs)
