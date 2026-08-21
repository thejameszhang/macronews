"""
Keyword pre-filter gate for the article -> asset-group mapper.

Skips the LLM call for an (article, group) pair when the article contains none
of that group's keywords. The gate is recall-optimized: a false positive costs
one LLM call (the model still rejects the pair), while a false negative loses
the tag for good.

The gate is a production cost optimization and runs on djnw only; see MapperConfig
in config/runconfig.py. Keywords live in `config/group_universe.yaml` and are
also read by `kg/link_groups.py`, where a false positive is NOT free -- it
mints a spurious entity->asset-group edge with no model in the loop to reject
it. Broaden a keyword for gate recall and you narrow KG link precision.
"""

import re

from macronews.utils.groups import load_mapper_group_universe


def _boundary(body: str) -> re.Pattern:
    """Alphanumeric-boundary, case-insensitive match.

    The gate's measured recall and fire-rate figures are tied to this exact
    construction -- re-measure before changing it.
    """
    return re.compile(rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])", re.IGNORECASE)


def compile_gate(universe: dict | None = None) -> dict[str, re.Pattern]:
    """group_key -> one compiled alternation over that group's keywords.

    Defaults to the MAPPER's universe (39 groups), not the full 50. The gate is a
    mapper-only concept -- nothing else gates -- and gate_fires() below calls this
    with no argument. A default of 50 here would make gate_fires disagree with the
    39-group pipeline, and everything that MEASURES the gate reads gate_fires.
    """
    if universe is None:
        universe = load_mapper_group_universe()
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
    group's own name a keyword hit, fire the gate on all 39 groups for every
    article, and turn the gate into a no-op at full cost.

    Also NOT the raw `text` field of the cleaned shard. The loader's paragraphs are
    shorter than it for 67% of articles, so matching the raw text matches ~124 characters
    per article that the gate never sees. A scorecard that did exactly that reported the
    gate as cheaper than it is.
    """
    return (headline or "") + "\n" + "\n".join(paragraphs)


def gate_fires(articles: list[dict]) -> dict[str, set[str]]:
    """article_id -> the group keys whose keywords fire on it.

    The ONE definition of "which pairs does the gate let through", for everything that
    measures the gate rather than runs it: the corpus call-count, the recall scorecard,
    and anything after them. `pipeline.run_pipeline` necessarily makes the same decision
    inline -- it is assembling the batch as it goes -- so it is the one place the loop is
    written twice, and test_gate_wiring pins the two together.

    Every consumer that wrote this loop for itself got it wrong: one matched paragraphs
    but not the headline, another matched the raw text the loader had already trimmed.
    Both drifted quietly, and both made the gate look cheaper than it is.

    Screened articles are absent -- the mapper never reaches the gate for them.
    """
    gate = compile_gate()
    fires: dict[str, set[str]] = {}
    for a in articles:
        if a.get("filtered_reasons"):
            continue
        text = gate_text(a["headline"], a["paragraphs"])
        fires[a["id"]] = {gk for gk, pat in gate.items() if pat.search(text)}
    return fires
