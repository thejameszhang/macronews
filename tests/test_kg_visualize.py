"""Tests for src/kg/visualize.py (cosmos.gl renderer, Python side)."""

import sys
from pathlib import Path

import networkx as nx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kg.visualize import (  # noqa: E402
    filter_graph,
    serialize_graph,
    build_legend_html,
)


def _g():
    """A small MultiDiGraph mirroring build_graph's output shape."""
    g = nx.MultiDiGraph()
    g.add_node("Federal Reserve", entity_type="CENTRAL_BANK",
               source_articles=["a1", "a2"])
    g.add_node("U.S. Federal Funds Rate", entity_type="INTEREST_RATE",
               source_articles=["a1"])
    g.add_node("Gold", entity_type="COMMODITY", source_articles=["a3"])
    g.add_edge("Federal Reserve", "U.S. Federal Funds Rate", key="RAISES",
               relation="RAISES", count=2,
               sources=[{"article_id": "a1", "date": "2026-01-01", "headline": "Fed Raises Rates", "paragraphs": [0]},
                        {"article_id": "a2", "date": "2026-02-01", "headline": "Fed Hikes Again", "paragraphs": [1]}])
    g.add_edge("Federal Reserve", "Gold", key="IMPACT",
               relation="IMPACT", count=1,
               sources=[{"article_id": "a3", "date": "2026-03-01", "headline": "Gold Reacts to Fed", "paragraphs": [2]}])
    return g


def test_filter_defaults_return_full_graph():
    g = _g()
    out = filter_graph(g, min_count=1, min_degree=1)
    assert out.number_of_nodes() == 3
    assert out.number_of_edges() == 2


def test_filter_does_not_mutate_input():
    g = _g()
    _ = filter_graph(g, min_count=2, min_degree=1)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


def test_filter_min_count_drops_low_edges():
    g = _g()
    out = filter_graph(g, min_count=2, min_degree=1)
    assert out.number_of_edges() == 1
    assert "Gold" not in out
    assert set(out.nodes()) == {"Federal Reserve", "U.S. Federal Funds Rate"}


def test_filter_min_degree_applied_after_edge_filter():
    g = _g()
    out = filter_graph(g, min_count=1, min_degree=2)
    assert set(out.nodes()) == {"Federal Reserve"}
    assert out.number_of_edges() == 0


def test_filter_distinct_neighbor_degree():
    """Parallel edges between the same pair count as ONE neighbor."""
    g = nx.MultiDiGraph()
    g.add_node("A", entity_type="PERSON", source_articles=["a1"])
    g.add_node("B", entity_type="PERSON", source_articles=["a1"])
    g.add_edge("A", "B", key="OWNS", relation="OWNS", count=1, sources=[])
    g.add_edge("A", "B", key="ANNOUNCES", relation="ANNOUNCES", count=1, sources=[])
    out = filter_graph(g, min_count=1, min_degree=2)
    assert out.number_of_nodes() == 0


def test_serialize_shape():
    g = _g()
    data = serialize_graph(g)
    assert set(data.keys()) == {"nodes", "edges"}
    node = next(n for n in data["nodes"] if n["key"] == "Federal Reserve")
    assert node["label"] == "Federal Reserve"
    assert node["entityType"] == "CENTRAL_BANK"
    assert node["articleCount"] == 2
    assert "color" in node and node["color"].startswith("#")
    assert node["size"] > 0


def test_serialize_size_scales_with_degree():
    g = _g()
    data = serialize_graph(g)
    by_key = {n["key"]: n for n in data["nodes"]}
    assert by_key["Federal Reserve"]["size"] > by_key["Gold"]["size"]


def test_serialize_edge_fields():
    g = _g()
    data = serialize_graph(g)
    e = next(e for e in data["edges"]
             if e["source"] == "Federal Reserve" and e["target"] == "U.S. Federal Funds Rate")
    assert e["relationType"] == "RAISES"
    assert e["label"] == "RAISES"
    assert e["count"] == 2
    assert "color" in e and e["color"].startswith("#")
    assert e["dateRange"] == "2026-01-01 → 2026-02-01"
    # sources list: deduped, sorted, each with article_id/date/headline
    assert "sources" in e
    assert e["count"] == len(e["sources"])
    assert all({"article_id", "date", "headline"} <= set(s.keys()) for s in e["sources"])
    # sorted by (date, article_id)
    dates = [s["date"] for s in e["sources"]]
    assert dates == sorted(dates)
    assert e["sources"][0]["headline"] == "Fed Raises Rates"
    assert e["sources"][1]["headline"] == "Fed Hikes Again"


def test_serialize_single_date_edge():
    g = _g()
    data = serialize_graph(g)
    e = next(e for e in data["edges"] if e["target"] == "Gold")
    assert e["dateRange"] == "2026-03-01"


def test_legend_lists_present_types_and_relations():
    g = _g()
    html = build_legend_html(g)
    assert "CENTRAL_BANK" in html
    assert "INTEREST_RATE" in html
    assert "COMMODITY" in html
    assert "RAISES" in html
    assert "IMPACT" in html


def test_legend_has_counts():
    g = _g()
    html = build_legend_html(g)
    # Fixture: 3 entity types each count 1, plus 2 relations each count 1.
    # Every type/relation row carries a "(1)" badge -> 5 total.
    assert html.count("(1)") == 5


def test_render_html_substitutes_all_placeholders():
    from kg.visualize import render_html
    html = render_html({"nodes": [], "edges": []}, "<h4>Legend</h4>", "My Title")
    assert "{{TITLE}}" not in html
    assert "{{VENDORED_JS}}" not in html
    assert "{{GRAPH_DATA}}" not in html
    assert "{{LEGEND_HTML}}" not in html
    assert "My Title" in html
    assert "<h4>Legend</h4>" in html
    # cosmos.gl bundle inlined (globalThis.cosmos= appears in the vendored bundle)
    assert "globalThis.cosmos=" in html


def test_vendored_js_is_cosmos_bundle():
    """_VENDORED_JS must name the cosmos bundle and the file must exist on disk."""
    from kg.visualize import _VENDORED_JS, ASSETS_DIR
    assert _VENDORED_JS == ("cosmos.graph.umd.js",), (
        f"Expected (\"cosmos.graph.umd.js\",), got {_VENDORED_JS!r}"
    )
    assert (ASSETS_DIR / "cosmos.graph.umd.js").exists(), (
        "cosmos.graph.umd.js not found under ASSETS_DIR"
    )


def test_render_html_uses_cosmos_graph():
    """render_html output must reference cosmos API and have no leftover
    {{ }} placeholders (which would indicate an un-filled template slot)."""
    from kg.visualize import render_html, serialize_graph, build_legend_html
    g = _g()
    html = render_html(serialize_graph(g), build_legend_html(g), "T")
    assert "globalThis.cosmos=" in html, "cosmos bundle not inlined"
    assert "new cosmos.Graph" in html, "cosmos.Graph constructor not present"
    assert "{{" not in html, "un-filled template placeholder remains"


def test_render_html_escapes_script_close_in_node_names():
    """A node name containing '</script>' must not break out of the
    inlined <script> block — the '</' sequence is escaped in the JSON."""
    from kg.visualize import render_html
    data = {"nodes": [{"key": "evil</script>", "label": "evil</script>",
                       "entityType": "EVENT", "color": "#fff", "size": 5,
                       "articleCount": 1}], "edges": []}
    html = render_html(data, "", "T")
    # The raw close-tag must not appear inside the data literal; it's escaped.
    assert "evil<\\/script>" in html


def test_render_html_escapes_comment_open_in_node_names():
    """A node name containing '<!--' must not flip the HTML tokenizer into
    script-data-escaped state — the '<!--' sequence is escaped in the JSON."""
    from kg.visualize import render_html
    data = {"nodes": [{"key": "comment<!--x", "label": "comment<!--x",
                       "entityType": "EVENT", "color": "#fff", "size": 5,
                       "articleCount": 1}], "edges": []}
    html = render_html(data, "", "T")
    # The escaped form must be present (raw '<!--' in the node name is escaped).
    assert "comment<\\!--x" in html
    # And the raw node-name form must NOT appear (escaped form replaces it).
    assert "comment<!--x" not in html


def test_serialize_emits_deduped_sources():
    """Edge with duplicate article_id in raw sources must be deduped;
    count == len(sources); sorted by (date, article_id); keys include
    article_id, date, headline."""
    g = nx.MultiDiGraph()
    g.add_node("A", entity_type="CENTRAL_BANK", source_articles=["x1", "x2"])
    g.add_node("B", entity_type="INTEREST_RATE", source_articles=["x1", "x2"])
    # x1 appears TWICE (duplicate), x2 once — dedup should yield 2 distinct sources
    g.add_edge("A", "B", key="RAISES", relation="RAISES", count=3,
               sources=[
                   {"article_id": "x1", "date": "2022-03-02", "headline": "First article"},
                   {"article_id": "x2", "date": "2022-03-01", "headline": "Second article"},
                   {"article_id": "x1", "date": "2022-03-02", "headline": "First article (dup)"},
               ])
    data = serialize_graph(g)
    e = next(ed for ed in data["edges"] if ed["source"] == "A" and ed["target"] == "B")
    sources = e["sources"]
    # Deduped: only 2 distinct article_ids
    assert len(sources) == 2, f"expected 2 deduped sources, got {len(sources)}"
    assert e["count"] == len(sources)
    # Sorted by date ascending (x2 is 2022-03-01, x1 is 2022-03-02)
    assert sources[0]["article_id"] == "x2"
    assert sources[1]["article_id"] == "x1"
    # Each source has exactly the three required keys
    for s in sources:
        assert {"article_id", "date", "headline"} <= set(s.keys())
    # Headlines come from the FIRST occurrence of each article_id
    assert sources[0]["headline"] == "Second article"
    assert sources[1]["headline"] == "First article"


def test_build_graph_captures_headline():
    """build_graph must store headline in each edge source entry."""
    import json
    import tempfile
    from pathlib import Path
    from kg.build_graph import build_graph

    rows = [
        {
            "article_id": "art001",
            "date": "2022-03-15",
            "headline": "Fed Raises Rates by 25bps",
            "paragraphs": ["The Fed raised rates."],
            "facts": [
                {"subject": "Federal Reserve", "subject_type": "CENTRAL_BANK",
                 "relation": "RAISES", "object": "Federal Funds Rate",
                 "object_type": "INTEREST_RATE", "evidence_paragraphs": [0]},
            ],
        },
        {
            "article_id": "art002",
            "date": "2022-03-16",
            "headline": "Markets React to Fed Hike",
            "paragraphs": ["Markets fell after the hike."],
            "facts": [
                {"subject": "Federal Reserve", "subject_type": "CENTRAL_BANK",
                 "relation": "RAISES", "object": "Federal Funds Rate",
                 "object_type": "INTEREST_RATE", "evidence_paragraphs": [0]},
            ],
        },
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        tmp_path = Path(f.name)

    try:
        g = build_graph(tmp_path)
        # Find the edge between Federal Reserve and Federal Funds Rate
        edge_data = None
        for u, v, d in g.edges(data=True):
            if d.get("relation") == "RAISES":
                edge_data = d
                break
        assert edge_data is not None, "RAISES edge not found"
        sources = edge_data["sources"]
        assert len(sources) == 2
        assert all("headline" in s for s in sources)
        headlines = {s["article_id"]: s["headline"] for s in sources}
        assert headlines["art001"] == "Fed Raises Rates by 25bps"
        assert headlines["art002"] == "Markets React to Fed Hike"
    finally:
        tmp_path.unlink(missing_ok=True)
