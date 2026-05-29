"""Render a KG MultiDiGraph as a self-contained cosmos.gl HTML page.

WebGL2 GPU force simulation via cosmos.gl 2.6.4 (vendored, src/kg/assets/).
Replaces the previous sigma/ForceAtlas2 backend. The full graph is filtered
to a renderable view (min_count / min_degree, both default 1 = full graph),
serialized to JSON, and inlined into an HTML template.

Vendored JS versions are recorded in src/kg/assets/VERSIONS.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import networkx as nx

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from kg.schemas import ENTITY_TYPES_TUPLE, RELATION_TYPES_TUPLE  # noqa: E402

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
# colors. With the current 18-type / 18-relation schemas, entities fit
# entirely in tab20 and relations in tab20c, so tab20b is never reached
# and there is no entity/relation color collision. If either list grows
# past 20, overflow would pull shared tab20b colors — revisit then.
ENTITY_COLORS = _make_palette(ENTITY_TYPES_TUPLE, ("tab20", "tab20b"))
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


def serialize_graph(g: nx.MultiDiGraph) -> dict:
    """Serialize to {"nodes": [...], "edges": [...]} for cosmos.gl."""
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
    edges = []
    for u, v, d in g.edges(data=True):
        rel = d.get("relation", "")
        raw_sources = d.get("sources", [])
        seen: set[str] = set()
        sources: list[dict] = []
        for s in raw_sources:
            aid = s.get("article_id")
            if aid is None or aid in seen:
                continue
            seen.add(aid)
            sources.append({
                "article_id": aid,
                "date": s.get("date", ""),
                "headline": s.get("headline", ""),
            })
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
            "sources": sources,
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


# Vendored JS bundle for cosmos.gl GPU force simulation.
# Names + versions recorded in assets/VERSIONS.txt.
_VENDORED_JS = ("cosmos.graph.umd.js",)


def render_html(graph_data: dict, legend_html: str, title: str) -> str:
    """Fill the HTML template: inline vendored JS, graph JSON, legend.

    The graph JSON is embedded as a JS literal. json.dumps escapes JSON
    structure; we additionally guard the `</` sequence so a node name
    containing `</script>` can't break out of the inlined <script> block.
    """
    template = TEMPLATE_PATH.read_text()
    vendored = "\n".join((ASSETS_DIR / fn).read_text() for fn in _VENDORED_JS)
    data_json = (json.dumps(graph_data, ensure_ascii=False)
                 .replace("</", "<\\/")
                 .replace("<!--", "<\\!--"))
    return (
        template
        .replace("{{TITLE}}", title)
        .replace("{{VENDORED_JS}}", vendored)
        .replace("{{GRAPH_DATA}}", data_json)
        .replace("{{LEGEND_HTML}}", legend_html)
    )


def visualize(g: nx.MultiDiGraph, output_html: Path,
              min_count: int = 1, min_degree: int = 1,
              title: str = "Macro KG") -> dict:
    """Filter -> serialize -> write a self-contained cosmos.gl HTML page."""
    n0, e0 = g.number_of_nodes(), g.number_of_edges()
    fg = filter_graph(g, min_count, min_degree)
    html = render_html(serialize_graph(fg), build_legend_html(fg), title)
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
    args = p.parse_args()

    from kg.build_graph import build_graph
    g = build_graph(args.jsonl)
    visualize(g, args.output, args.min_count, args.min_degree, args.title)


if __name__ == "__main__":
    main()
