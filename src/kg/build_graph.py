"""Build a NetworkX KG from KG-extractor TemporalEvent JSONL output.

Schema: each row has `events` (list of TemporalEvent objects), each event
has a `triplets` list. Nodes are derived from the union of
(subject, subject_type) and (object, object_type) across all triplets.
Surface-form variants that differ only in case are merged
(case-insensitive matching). The displayed node label is the first-seen
surface form, title-cased on render with acronyms preserved.

Nodes are keyed by the title-cased display name. Node attributes:
  - entity_type: str (taken from the first occurrence; warns on conflict)
  - source_articles: sorted list of article_ids where the entity appears

Edges are merged by (subject, relation, object) under the same
case-insensitive matching. Edge attributes:
  - relation: str
  - sources: list of {article_id, date, headline, paragraphs, statement,
             value, statement_type, temporal_type, valid_at, invalid_at} —
             one per event that asserts this triple; `statement` and the
             temporal fields come from the parent event, `value` from the
             individual triplet
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

from kg.schemas import ASSET_GROUP_NODE_TYPE, ASSET_GROUP_RELATION

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


def _group_display_names() -> dict[str, str]:
    """group_key -> human display name, from group_universe.yaml."""
    from utils.groups import load_group_universe  # deferred: skip utils load when entity-groups unused
    return {k: gv["name"] for k, gv in load_group_universe().items()}


def _article_date(article_id: str) -> str:
    """YYYY-MM-DD from the YYYYMMDD prefix of an accession id (else '')."""
    s = str(article_id)
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}" if len(s) >= 8 and s[:8].isdigit() else ""


def build_graph(jsonl_path: Path, entity_groups_path: Path | None = None) -> nx.MultiDiGraph:
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
    article_headline: dict[str, str] = {}  # article_id -> headline (for synthetic-edge sources)

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
            article_headline[aid] = headline

            for ev in row.get("events", []):
                for t in ev.get("triplets", []):
                    _record_endpoint(t["subject"], t["subject_type"], aid)
                    _record_endpoint(t["object"], t["object_type"], aid)
                    s = _canon(t["subject"])
                    o = _canon(t["object"])
                    key = (s, t["relation"], o)
                    edge_sources[key].append({
                        "article_id": aid,
                        "date": date,
                        "headline": headline,
                        "paragraphs": ev.get("evidence_paragraphs", []),
                        "statement": ev.get("statement", ""),
                        "value": t.get("value"),
                        "statement_type": ev.get("statement_type"),
                        "temporal_type": ev.get("temporal_type"),
                        "valid_at": ev.get("valid_at"),
                        "invalid_at": ev.get("invalid_at"),
                        "id": ev.get("id"),
                        "invalidated_by": ev.get("invalidated_by"),
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

    if entity_groups_path is not None:
        entity_groups = json.loads(Path(entity_groups_path).read_text())
        # entity_groups keys are casefold(name); node IDs are _display(first_seen).
        node_by_casefold = {n.casefold(): n for n in g.nodes()}
        names = _group_display_names()                    # group_key -> display name
        misses = 0
        unknown_keys: set[str] = set()
        for cf_name, rec in entity_groups.items():
            node_id = node_by_casefold.get(cf_name)
            if node_id is None:                            # entity filtered out / absent
                misses += 1
                continue
            for grp in rec["groups"]:
                if grp["key"] not in names:
                    unknown_keys.add(grp["key"])           # stale entity_groups vs universe
                anchor = f"[ASSET_GROUP] {names.get(grp['key'], grp['key'])}"
                if anchor not in g:
                    g.add_node(anchor, entity_type=ASSET_GROUP_NODE_TYPE, source_articles=[])
                srcs = [{"article_id": a, "date": _article_date(a),
                         "headline": article_headline.get(a, "")}
                        for a in g.nodes[node_id].get("source_articles", [])]
                g.add_edge(node_id, anchor, key=ASSET_GROUP_RELATION,
                           relation=ASSET_GROUP_RELATION, synthetic=True,
                           method=grp["method"], sources=srcs, count=len(srcs))
        if misses:
            logger.debug("Skipped %d entity_groups entries with no matching graph node", misses)
        if unknown_keys:
            logger.warning("entity_groups has %d group key(s) absent from the universe "
                           "(using raw key as anchor): %s", len(unknown_keys), sorted(unknown_keys))

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
    p.add_argument("--entity-groups", type=Path, default=None,
                   help="entity_groups.json (adds ASSET_GROUP anchors + edges)")
    args = p.parse_args()

    g = build_graph(args.jsonl, entity_groups_path=args.entity_groups)
    stats = graph_stats(g)
    print(f"Graph: {stats['nodes']} nodes, {stats['edges']} edges, "
          f"{stats['unique_relations']} relation types")
    print(f"By entity type:")
    for et, n in stats["by_entity_type"].items():
        print(f"  {et:>22}: {n}")


if __name__ == "__main__":
    main()
