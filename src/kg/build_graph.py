"""Build a NetworkX KG from KG-extractor v2 JSONL output.

v2 schema: each row has `facts` only (no separate `entities` array).
Nodes are derived from the union of (subject, subject_type) and
(object, object_type) across all facts. Surface-form variants that
differ only in case are merged (case-insensitive matching). The
displayed node label is the first-seen surface form, title-cased
on render with acronyms preserved.

Nodes are keyed by the title-cased display name. Node attributes:
  - entity_type: str (taken from the first occurrence; warns on conflict)
  - source_articles: sorted list of article_ids where the entity appears

Edges are merged by (subject, relation, object) under the same
case-insensitive matching. Edge attributes:
  - relation: str
  - sources: list of {article_id, date, paragraphs} — one per article
             that asserts this triple
  - count: len(sources), useful for edge-weight visualization
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _display(name: str) -> str:
    """Title-case for KG rendering with acronym preservation.

    All-lowercase words get their first letter capitalized; words
    containing any uppercase letter (acronyms like 'GDP', 'S&P',
    'U.S.') are left alone. Used both as the canonical display name
    and as the node ID after case-insensitive merging.
    """
    return " ".join(
        w.capitalize() if w.islower() else w
        for w in name.split(" ")
    )


def build_graph(jsonl_path: Path) -> nx.MultiDiGraph:
    """Read KG v2 JSONL and return a NetworkX MultiDiGraph.

    Entities are matched case-insensitively across articles. The first
    surface form seen for a given casefold-key is canonicalized via
    `_display()` (acronym-preserving title-case) and used as the node
    ID for all subsequent occurrences.
    """
    g = nx.MultiDiGraph()

    # Case-insensitive merge: keep canonical display name per casefold key.
    canonical: dict[str, str] = {}  # casefold(name) -> _display(first_seen)
    entity_type: dict[str, str] = {}  # canonical -> type
    entity_articles: dict[str, set[str]] = defaultdict(set)
    edge_sources: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    type_conflicts: dict[str, set[str]] = defaultdict(set)

    def _canon(name: str) -> str:
        key = name.casefold()
        if key not in canonical:
            canonical[key] = _display(name)
        return canonical[key]

    def _record_endpoint(name: str, typ: str, aid: str) -> None:
        c = _canon(name)
        if c in entity_type and entity_type[c] != typ:
            type_conflicts[c].add(typ)
            type_conflicts[c].add(entity_type[c])
        else:
            entity_type[c] = typ
        entity_articles[c].add(aid)

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            aid = row["article_id"]
            date = row.get("date", "")
            headline = row.get("headline", "")

            if "entities" in row:
                raise ValueError(
                    f"{jsonl_path} looks like a v1 sidecar (has top-level "
                    f"`entities` key). build_graph v2 only reads v2 output."
                )

            for fct in row.get("facts", []):
                _record_endpoint(fct["subject"], fct["subject_type"], aid)
                _record_endpoint(fct["object"], fct["object_type"], aid)
                s = _canon(fct["subject"])
                o = _canon(fct["object"])
                key = (s, fct["relation"], o)
                edge_sources[key].append({
                    "article_id": aid,
                    "date": date,
                    "headline": headline,
                    "paragraphs": fct.get("evidence_paragraphs", []),
                })

    for name, et in entity_type.items():
        g.add_node(
            name,
            entity_type=et,
            source_articles=sorted(entity_articles[name]),
        )

    for (s, r, o), sources in edge_sources.items():
        g.add_edge(
            s, o,
            key=r,
            relation=r,
            sources=sources,
            count=len(sources),
        )

    if type_conflicts:
        for name, types in type_conflicts.items():
            logger.warning("Type conflict for %r: %s", name, sorted(types))

    return g


def graph_stats(g: nx.MultiDiGraph) -> dict:
    by_type: dict[str, int] = defaultdict(int)
    for _, d in g.nodes(data=True):
        by_type[d.get("entity_type", "UNKNOWN")] += 1
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "unique_relations": len({d["relation"] for _, _, d in g.edges(data=True)}),
        "by_entity_type": dict(sorted(by_type.items())),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path, help="KG v2 JSONL")
    args = p.parse_args()

    g = build_graph(args.jsonl)
    stats = graph_stats(g)
    print(f"Graph: {stats['nodes']} nodes, {stats['edges']} edges, "
          f"{stats['unique_relations']} relation types")
    print(f"By entity type:")
    for et, n in stats["by_entity_type"].items():
        print(f"  {et:>22}: {n}")


if __name__ == "__main__":
    main()
