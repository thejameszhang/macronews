"""Render a KG MultiDiGraph as a self-contained cosmos.gl HTML page.

WebGL2 GPU force simulation via cosmos.gl 2.6.4 (vendored, kg/assets/).
Replaces the previous sigma/ForceAtlas2 backend. The full graph is filtered
to a renderable view (min_count / min_degree, both default 1 = full graph),
serialized to JSON, and inlined into an HTML template.

Vendored JS versions are recorded in kg/assets/VERSIONS.txt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import networkx as nx

from macronews.kg.schemas import (
    ENTITY_TYPES_TUPLE, RELATION_TYPES_TUPLE, ASSET_GROUP_NODE_TYPE, ASSET_GROUP_RELATION,
)
from macronews.utils.groups import load_group_universe
from macronews.kg.article_meta import load_display_dates, shard_path

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "graph.html.template"

_UNKNOWN_COLOR = "#9e9e9e"

# Node sizing: base + scale * sqrt(distinct-neighbor degree), capped.
# sqrt (not linear) so a 50-edge hub doesn't render as a giant disk that
# its own edges disappear behind; the cap bounds the largest hubs.
NODE_BASE = 9
NODE_SCALE = 6.0
NODE_MAX = 85


def _make_palette(types: tuple[str, ...], cmap_names: tuple[str, ...]) -> dict[str, str]:
    """Distinct color per type from matplotlib qualitative colormaps,
    deterministic so a type always gets the same color. Cycles if the
    type list exceeds the combined palette size."""
    palette: list[str] = []
    for name in cmap_names:
        cmap = mpl.colormaps[name]
        for i in range(cmap.N):
            palette.append(mcolors.to_hex(cmap(i)))
    return {t: palette[i % len(palette)] for i, t in enumerate(types)}


# tab20b is shared as the OVERFLOW palette for both entity and relation
# colors. Entity types are now 19 extractor types + 1 synthetic ASSET_GROUP
# = 20, which fills tab20 EXACTLY (ASSET_GROUP at index 19, the last slot);
# the 19 relations fit in tab20c. So tab20b is still never reached and there
# is no entity/relation color collision — but entities are now AT the 20 cap:
# one more entity type would overflow into shared tab20b. Revisit then.
# ASSET_GROUP_NODE_TYPE is appended last so existing type→color assignments
# don't shift.
ENTITY_COLORS = _make_palette(ENTITY_TYPES_TUPLE + (ASSET_GROUP_NODE_TYPE,), ("tab20", "tab20b"))
RELATION_COLORS = _make_palette(RELATION_TYPES_TUPLE, ("tab20c", "tab20b"))


def _distinct_degree(g: nx.MultiDiGraph, node) -> int:
    """Number of distinct neighbors (in or out); parallel edges count once."""
    nbrs = set(g.successors(node)) | set(g.predecessors(node))
    nbrs.discard(node)
    return len(nbrs)


def filter_graph(g: nx.MultiDiGraph, min_count: int, min_degree: int) -> nx.MultiDiGraph:
    """Return a new filtered MultiDiGraph (does not mutate g).

    1. Keep edges with count >= min_count.
    2. Recompute distinct-neighbor degree on the count-filtered subgraph
       (NOT the original graph).
    3. Keep nodes with degree >= min_degree.
    4. Drop edges whose endpoint was removed in step 3.

    Note: a node can satisfy min_degree in step 2 yet end up isolated in
    the output if all its qualifying neighbors were themselves dropped in
    step 3 (the cascade is not re-run). This is intentional — the degree
    threshold is a filter on the count-surviving structure, not a
    guarantee about the final rendered degree.
    """
    h = nx.MultiDiGraph()
    h.add_nodes_from(g.nodes(data=True))
    for u, v, k, d in g.edges(keys=True, data=True):
        if d.get("count", 1) >= min_count:
            h.add_edge(u, v, key=k, **d)

    keep = {n for n in h.nodes() if _distinct_degree(h, n) >= min_degree}

    out = nx.MultiDiGraph()
    out.add_nodes_from((n, g.nodes[n]) for n in keep)
    for u, v, k, d in h.edges(keys=True, data=True):
        if u in keep and v in keep:
            out.add_edge(u, v, key=k, **d)
    return out


def _edge_date_range(sources: list[dict]) -> str:
    dates = sorted({s["date"] for s in sources if s.get("date")})
    if not dates:
        return ""
    if len(dates) == 1:
        return dates[0]
    return f"{dates[0]} → {dates[-1]}"


def _edge_field(sources: list[dict], key: str) -> str:
    """Summarize one field across an edge's sources: the unanimous value, or
    'mixed' when sources disagree, or '' when no source carries it."""
    vals = {s.get(key) for s in sources if s.get(key)}
    if not vals:
        return ""
    if len(vals) == 1:
        return next(iter(vals))
    return "mixed"


def serialize_graph(g: nx.MultiDiGraph, display_dates: dict[str, str] | None = None) -> dict:
    """Serialize to {"nodes": [...], "edges": [...]} for cosmos.gl.

    display_dates (optional) maps article_id -> exact release timestamp string;
    when supplied, each edge source carries it as ``display_date`` for the
    Reader tab's within-day ordering. Omitted -> ``display_date == ""``.
    """
    dd = display_dates or {}
    nodes = []
    for n, d in g.nodes(data=True):
        et = d.get("entity_type", "UNKNOWN")
        deg = _distinct_degree(g, n)
        nodes.append({
            "key": n,
            "label": n,
            "entityType": et,
            "color": ENTITY_COLORS.get(et, _UNKNOWN_COLOR),
            "size": min(NODE_MAX, NODE_BASE + NODE_SCALE * (deg ** 0.5)),
            "articleCount": len(d.get("source_articles", [])),
        })
    # Supersession: a source may carry invalidated_by = the id of the event that
    # closed it. Collect referenced superseder ids so we emit a source's `id` ONLY
    # when something points at it — keeps the embedded JSON lean (most sources are
    # never a superseder).
    referenced_ids = {
        s.get("invalidated_by")
        for _, _, d in g.edges(data=True)
        for s in d.get("sources", [])
        if s.get("invalidated_by")
    }
    edges = []
    for u, v, d in g.edges(data=True):
        rel = d.get("relation", "")
        raw_sources = d.get("sources", [])
        seen: set[str] = set()
        sources: list[dict] = []
        # Dedup is per-(edge, article): if one article emits two events on the same
        # (subject, relation, object), only the first survives. Rare consequence: a
        # superseder whose id lives on the dropped event won't carry its `id` here, so
        # the Reader can't resolve that one supersession (it falls back gracefully).
        for s in raw_sources:
            aid = s.get("article_id")
            if aid is None or aid in seen:
                continue
            seen.add(aid)
            src = {
                "article_id": aid,
                "date": s.get("date", ""),
                "display_date": dd.get(aid, ""),
                "headline": s.get("headline", ""),
                "statement": s.get("statement", ""),
                "value": s.get("value"),
                "paragraphs": s.get("paragraphs", []),
                "statement_type": s.get("statement_type"),
                "temporal_type": s.get("temporal_type"),
                "valid_at": s.get("valid_at"),
                "invalid_at": s.get("invalid_at"),
            }
            invalidated_by = s.get("invalidated_by")
            if invalidated_by:
                src["invalidated_by"] = invalidated_by
            if s.get("id") in referenced_ids:
                src["id"] = s.get("id")
            sources.append(src)
        sources.sort(key=lambda s: (s["date"], s["article_id"]))
        count = len(sources) if sources else 1
        edges.append({
            "source": u,
            "target": v,
            "label": rel,
            "relationType": rel,
            "color": RELATION_COLORS.get(rel, _UNKNOWN_COLOR),
            "count": count,
            "dateRange": _edge_date_range(d.get("sources", [])),
            "statementType": _edge_field(d.get("sources", []), "statement_type"),
            "temporalType": _edge_field(d.get("sources", []), "temporal_type"),
            "sources": sources,
            "synthetic": bool(d.get("synthetic", False)),
            "method": d.get("method", ""),
        })
    return {"nodes": nodes, "edges": edges}


def build_legend_html(g: nx.MultiDiGraph) -> str:
    """Legend rows for entity types + relations actually present, with
    color swatch + count. Clickable rows (JS wires the toggle)."""
    type_counts: dict[str, int] = {}
    for _, d in g.nodes(data=True):
        et = d.get("entity_type", "UNKNOWN")
        type_counts[et] = type_counts.get(et, 0) + 1
    rel_counts: dict[str, int] = {}
    for _, _, d in g.edges(data=True):
        r = d.get("relation", "")
        rel_counts[r] = rel_counts.get(r, 0) + 1

    # `name` is an entity-type or relation CODE (schema Literal: uppercase
    # ASCII like CENTRAL_BANK / RAISES), never a free-text / LLM-generated
    # entity name — so it is safe to interpolate into the HTML/JS attribute
    # without escaping. (Entity names are not passed to this function.)
    def rows(counts, colors, attr):
        out = []
        for name in sorted(counts):
            color = colors.get(name, _UNKNOWN_COLOR)
            out.append(
                f'<li data-{attr}="{name}" onclick="kgToggle(\'{attr}\', \'{name}\')">'
                f'<span class="kg-swatch" style="background:{color}"></span>'
                f'{name} <span class="count">({counts[name]})</span></li>'
            )
        return "\n".join(out)

    entity_rows = rows(type_counts, ENTITY_COLORS, "entitytype")
    relation_rows = rows(rel_counts, RELATION_COLORS, "relationtype")
    return (
        f'<h4>Entity Types ({len(type_counts)})</h4><ul>{entity_rows}</ul>'
        f'<h4>Relations ({len(rel_counts)})</h4><ul>{relation_rows}</ul>'
    )


def build_asset_groups(g: nx.MultiDiGraph) -> list[dict]:
    """For each of the 50 asset groups, the rendered-graph node keys forming its
    "bubble", for the panel navigator. Sorted by (asset_class, group name); all
    50 groups are always returned (a group with no bubble is greyed, count 0).

    When the resolution layer is present (build_graph was given entity-groups, so
    the graph has ASSET_GROUP anchor nodes), a group's bubble = its
    ``[ASSET_GROUP] <name>`` anchor plus every entity linked to it by a
    RELATED_TO_ASSET_GROUP edge; ``count`` is the number of linked members.

    Fallback (no resolution layer in the graph — e.g. a render without
    ``--entity-groups``): exact name match — keys are nodes whose name equals the
    group name or a member name/short_name.
    """
    universe = load_group_universe()
    has_anchors = any(d.get("entity_type") == ASSET_GROUP_NODE_TYPE
                      for _, d in g.nodes(data=True))

    out = []
    if has_anchors:
        for gv in universe.values():
            anchor = f"[ASSET_GROUP] {gv['name']}"
            if anchor in g:
                members = sorted({u for u, _, d in g.in_edges(anchor, data=True)
                                  if d.get("relation") == ASSET_GROUP_RELATION})
                keys, count = [anchor, *members], len(members)
            else:
                keys, count = [], 0
            out.append({"name": gv["name"], "assetClass": gv["asset_class"],
                        "keys": keys, "count": count})
    else:
        by_casefold: dict[str, list[str]] = {}
        for n in g.nodes():
            by_casefold.setdefault(n.casefold(), []).append(n)
        for gv in universe.values():
            terms = {gv["name"].casefold()}
            for m in gv["members"]:
                terms.add(m["name"].casefold())
                terms.add(m["short_name"].casefold())
            keys = sorted({k for t in terms for k in by_casefold.get(t, [])})
            out.append({"name": gv["name"], "assetClass": gv["asset_class"],
                        "keys": keys, "count": len(keys)})
    out.sort(key=lambda d: (d["assetClass"], d["name"]))
    return out


# Vendored JS bundle for cosmos.gl GPU force simulation.
# Names + versions recorded in assets/VERSIONS.txt.
_VENDORED_JS = ("cosmos.graph.umd.js",)


def _js_literal(obj) -> str:
    """JSON for inlining as a JS literal inside a <script> block: guard the
    `</` and `<!--` sequences so a string can't break out of the block."""
    return (json.dumps(obj, ensure_ascii=False)
            .replace("</", "<\\/")
            .replace("<!--", "<\\!--"))


def render_html(graph_data: dict, legend_html: str, title: str,
                asset_groups: list[dict]) -> str:
    """Fill the HTML template: inline vendored JS, graph JSON, legend, and the
    asset-group navigator data.

    The graph/asset JSON is embedded as a JS literal (see `_js_literal`).
    """
    template = TEMPLATE_PATH.read_text()
    vendored = "\n".join((ASSETS_DIR / fn).read_text() for fn in _VENDORED_JS)
    return (
        template
        .replace("{{TITLE}}", title)
        .replace("{{VENDORED_JS}}", vendored)
        .replace("{{GRAPH_DATA}}", _js_literal(graph_data))
        .replace("{{LEGEND_HTML}}", legend_html)
        .replace("{{ASSET_GROUPS_JSON}}", _js_literal(asset_groups))
    )


def visualize(g: nx.MultiDiGraph, output_html: Path,
              min_count: int = 1, min_degree: int = 1,
              title: str = "Macro KG",
              display_dates: dict[str, str] | None = None) -> dict:
    """Filter -> serialize -> write a self-contained cosmos.gl HTML page."""
    n0, e0 = g.number_of_nodes(), g.number_of_edges()
    fg = filter_graph(g, min_count, min_degree)
    html = render_html(serialize_graph(fg, display_dates), build_legend_html(fg), title,
                       build_asset_groups(fg))
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html)
    stats = {"nodes_total": n0, "nodes_rendered": fg.number_of_nodes(),
             "edges_total": e0, "edges_rendered": fg.number_of_edges()}
    print(f"Wrote {output_html} — rendering {stats['nodes_rendered']}/{n0} nodes, "
          f"{stats['edges_rendered']}/{e0} edges "
          f"(min_count={min_count}, min_degree={min_degree})")
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path, help="KG v2 JSONL")
    p.add_argument("--output", type=Path, default=Path("results/kg/graph.html"))
    p.add_argument("--min-count", type=int, default=1,
                   help="Drop edges asserted by fewer than N articles (default 1).")
    p.add_argument("--min-degree", type=int, default=1,
                   help="Drop nodes with fewer than N distinct neighbors (default 1).")
    p.add_argument("--title", type=str, default="Macro KG")
    p.add_argument("--entity-groups", type=Path, default=None,
                   help="entity_groups.json from link_groups.py (adds ASSET_GROUP anchors)")
    p.add_argument("--shards", nargs="+", default=None,
                   help="cleaned shard stems (e.g. 2014-05a) to join display_date "
                        "for within-day Reader ordering; omit for day-level ordering")
    args = p.parse_args()

    from macronews.kg.build_graph import build_graph
    g = build_graph(args.jsonl, entity_groups_path=args.entity_groups)
    display_dates = (load_display_dates([shard_path(s) for s in args.shards])
                     if args.shards else None)
    visualize(g, args.output, args.min_count, args.min_degree, args.title, display_dates)


if __name__ == "__main__":
    main()
