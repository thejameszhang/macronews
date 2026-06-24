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
    build_asset_groups,
    _edge_field,
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
               sources=[{"article_id": "a1", "date": "2026-01-01", "headline": "Fed Raises Rates", "paragraphs": [0], "statement_type": "FACT", "temporal_type": "STATIC", "valid_at": "2026-01-01T00:00:00", "invalid_at": None},
                        {"article_id": "a2", "date": "2026-02-01", "headline": "Fed Hikes Again", "paragraphs": [1], "statement_type": "FACT", "temporal_type": "STATIC", "valid_at": "2026-01-01T00:00:00", "invalid_at": None}])
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


def test_edge_field_summary():
    """_edge_field: unanimous -> that value; both -> 'mixed'; none -> ''."""
    assert _edge_field([{"statement_type": "FACT"}, {"statement_type": "FACT"}], "statement_type") == "FACT"
    assert _edge_field([{"temporal_type": "STATIC"}], "temporal_type") == "STATIC"
    assert _edge_field([{"statement_type": "FACT"}, {"statement_type": "OPINION"}], "statement_type") == "mixed"
    assert _edge_field([{"article_id": "a"}], "statement_type") == ""
    assert _edge_field([{"statement_type": None}], "statement_type") == ""
    assert _edge_field([], "statement_type") == ""


def test_serialize_carries_statement_and_temporal_type():
    g = _g()
    data = serialize_graph(g)
    raises = next(e for e in data["edges"]
                  if e["source"] == "Federal Reserve" and e["target"] == "U.S. Federal Funds Rate")
    assert raises["statementType"] == "FACT" and raises["temporalType"] == "STATIC"
    assert all(s.get("statement_type") == "FACT" for s in raises["sources"])
    impact = next(e for e in data["edges"] if e["target"] == "Gold")
    assert impact["statementType"] == "" and impact["temporalType"] == ""


def test_build_graph_reads_events_and_carries_temporal_fields(tmp_path):
    import json
    from kg.build_graph import build_graph
    row = {"article_id": "art1", "date": "2014-05-27", "events": [
        {"article_id": "art1", "statement": "The Fed held rates.",
         "statement_type": "FACT", "temporal_type": "DYNAMIC",
         "valid_at": "2014-05-27T00:00:00", "invalid_at": None,
         "evidence_paragraphs": [0],
         "triplets": [{"subject": "Federal Reserve", "subject_type": "CENTRAL_BANK",
                       "relation": "LEAVES_UNCHANGED", "object": "US Federal Funds Rate",
                       "object_type": "INTEREST_RATE", "value": None}]}]}
    p = tmp_path / "e.jsonl"; p.write_text(json.dumps(row) + "\n")
    g = build_graph(p)
    d = next(d for _, _, d in g.edges(data=True) if d["relation"] == "LEAVES_UNCHANGED")
    src = d["sources"][0]
    assert src["statement_type"] == "FACT" and src["temporal_type"] == "DYNAMIC"
    assert src["valid_at"] == "2014-05-27T00:00:00"


def test_serialize_single_date_edge():
    g = _g()
    data = serialize_graph(g)
    e = next(e for e in data["edges"] if e["target"] == "Gold")
    assert e["dateRange"] == "2026-03-01"


def test_serialize_source_carries_statement_value_and_validity():
    """Each serialized edge source exposes statement text + triplet value +
    valid_at/invalid_at for the provenance modal."""
    g = nx.MultiDiGraph()
    g.add_node("Federal Reserve", entity_type="CENTRAL_BANK", source_articles=["a1"])
    g.add_node("U.S. Federal Funds Rate", entity_type="INTEREST_RATE", source_articles=["a1"])
    g.add_edge("Federal Reserve", "U.S. Federal Funds Rate", key="RAISES",
               relation="RAISES", count=1,
               sources=[{"article_id": "a1", "date": "2026-01-01",
                         "headline": "Fed Raises", "paragraphs": [0, 3],
                         "statement": "The Fed raised its policy rate.",
                         "value": "by 25 bps",
                         "statement_type": "FACT", "temporal_type": "STATIC",
                         "valid_at": "2026-01-01T00:00:00", "invalid_at": None}])
    s = serialize_graph(g)["edges"][0]["sources"][0]
    assert s["statement"] == "The Fed raised its policy rate."
    assert s["value"] == "by 25 bps"
    assert s["paragraphs"] == [0, 3]
    assert s["valid_at"] == "2026-01-01T00:00:00"
    assert s["invalid_at"] is None


def test_build_asset_groups_exact_match_completeness_and_sort():
    """All 50 groups returned (even 0-match), exact-match on group name AND
    member short_names, sorted by (asset_class, name)."""
    g = nx.MultiDiGraph()
    g.add_node("Crude Oil", entity_type="COMMODITY", source_articles=["a1"])
    g.add_node("WTI Crude Oil", entity_type="COMMODITY", source_articles=["a1"])
    g.add_node("Some Unrelated Concept", entity_type="CONCEPT", source_articles=["a1"])
    groups = build_asset_groups(g)
    assert len(groups) == 50                              # every group listed
    crude = next(x for x in groups if x["name"] == "Crude Oil")
    assert "Crude Oil" in crude["keys"]                   # group-name match
    assert "WTI Crude Oil" in crude["keys"]               # member short_name match
    assert crude["count"] == len(crude["keys"])
    assert any(x["count"] == 0 for x in groups)           # empties still present
    keyed = [(x["assetClass"], x["name"]) for x in groups]
    assert keyed == sorted(keyed)                         # asset_class then name


def test_build_asset_groups_uses_anchors_when_resolution_layer_present():
    """When ASSET_GROUP anchors exist (the resolution layer), a group's bubble =
    its [ASSET_GROUP] anchor + the entities linked by RELATED_TO_ASSET_GROUP —
    NOT exact name matches. Groups without an anchor are greyed (count 0)."""
    from kg.schemas import ASSET_GROUP_NODE_TYPE
    g = nx.MultiDiGraph()
    g.add_node("Brent Crude", entity_type="COMMODITY", source_articles=["a1"])
    anchor = "[ASSET_GROUP] Crude Oil"
    g.add_node(anchor, entity_type=ASSET_GROUP_NODE_TYPE, source_articles=[])
    g.add_edge("Brent Crude", anchor, key="RELATED_TO_ASSET_GROUP",
               relation="RELATED_TO_ASSET_GROUP", synthetic=True, method="mapper-para",
               count=1, sources=[{"article_id": "a1", "date": "2014-05-01"}])
    groups = build_asset_groups(g)
    assert len(groups) == 50
    crude = next(x for x in groups if x["name"] == "Crude Oil")
    assert crude["count"] == 1                             # one linked member (the bubble)
    assert anchor in crude["keys"] and "Brent Crude" in crude["keys"]
    natgas = next(x for x in groups if x["name"] == "Natural Gas")
    assert natgas["count"] == 0 and natgas["keys"] == []   # no anchor -> greyed, not exact-matched


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
    html = render_html({"nodes": [], "edges": []}, "<h4>Legend</h4>", "My Title", [])
    assert "{{TITLE}}" not in html
    assert "{{VENDORED_JS}}" not in html
    assert "{{GRAPH_DATA}}" not in html
    assert "{{LEGEND_HTML}}" not in html
    assert "{{ASSET_GROUPS_JSON}}" not in html
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
    html = render_html(serialize_graph(g), build_legend_html(g), "T", build_asset_groups(g))
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
    html = render_html(data, "", "T", [])
    # The raw close-tag must not appear inside the data literal; it's escaped.
    assert "evil<\\/script>" in html


def test_render_html_escapes_comment_open_in_node_names():
    """A node name containing '<!--' must not flip the HTML tokenizer into
    script-data-escaped state — the '<!--' sequence is escaped in the JSON."""
    from kg.visualize import render_html
    data = {"nodes": [{"key": "comment<!--x", "label": "comment<!--x",
                       "entityType": "EVENT", "color": "#fff", "size": 5,
                       "articleCount": 1}], "edges": []}
    html = render_html(data, "", "T", [])
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
            "events": [
                {"article_id": "art001", "statement": "The Fed raised rates.",
                 "statement_type": "FACT", "temporal_type": "DYNAMIC",
                 "valid_at": "2022-03-15T00:00:00", "invalid_at": None,
                 "evidence_paragraphs": [0],
                 "triplets": [
                     {"subject": "Federal Reserve", "subject_type": "CENTRAL_BANK",
                      "relation": "RAISES", "object": "Federal Funds Rate",
                      "object_type": "INTEREST_RATE", "value": None}
                 ]},
            ],
        },
        {
            "article_id": "art002",
            "date": "2022-03-16",
            "headline": "Markets React to Fed Hike",
            "events": [
                {"article_id": "art002", "statement": "Markets fell after the hike.",
                 "statement_type": "FACT", "temporal_type": "DYNAMIC",
                 "valid_at": "2022-03-16T00:00:00", "invalid_at": None,
                 "evidence_paragraphs": [0],
                 "triplets": [
                     {"subject": "Federal Reserve", "subject_type": "CENTRAL_BANK",
                      "relation": "RAISES", "object": "Federal Funds Rate",
                      "object_type": "INTEREST_RATE", "value": None}
                 ]},
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


def test_asset_group_appended_to_palette_no_shift():
    from kg.visualize import ENTITY_COLORS, _make_palette
    from kg.schemas import ENTITY_TYPES_TUPLE, ASSET_GROUP_NODE_TYPE
    assert ASSET_GROUP_NODE_TYPE in ENTITY_COLORS
    assert all(t in ENTITY_COLORS for t in ENTITY_TYPES_TUPLE)
    # appending ASSET_GROUP must not RECOLOR any pre-existing type (not just keep keys).
    pre = _make_palette(ENTITY_TYPES_TUPLE, ("tab20", "tab20b"))
    assert all(ENTITY_COLORS[t] == pre[t] for t in ENTITY_TYPES_TUPLE)


def test_serialize_includes_asset_group_nodes_and_edges():
    import networkx as nx
    from kg.schemas import ASSET_GROUP_NODE_TYPE
    g = nx.MultiDiGraph()
    g.add_node("Crude Oil", entity_type="COMMODITY", source_articles=["a1"])
    g.add_node("[ASSET_GROUP] Crude Oil", entity_type=ASSET_GROUP_NODE_TYPE, source_articles=[])
    g.add_edge("Crude Oil", "[ASSET_GROUP] Crude Oil", key="RELATED_TO_ASSET_GROUP",
               relation="RELATED_TO_ASSET_GROUP", synthetic=True, method="exact",
               count=1, sources=[{"article_id": "a1", "date": "2014-05-01"}])
    data = serialize_graph(g)
    assert any(n["entityType"] == ASSET_GROUP_NODE_TYPE for n in data["nodes"])
    e = next(e for e in data["edges"] if e["relationType"] == "RELATED_TO_ASSET_GROUP")
    assert e["synthetic"] is True and e["method"] == "exact"


def test_serialize_graph_adds_display_date_when_map_supplied():
    g = _g()
    dd = {"a1": "20260101T120000.000Z", "a2": "20260201T080000.000Z"}
    data = serialize_graph(g, display_dates=dd)
    e = next(e for e in data["edges"]
             if e["source"] == "Federal Reserve" and e["target"] == "U.S. Federal Funds Rate")
    by_aid = {s["article_id"]: s for s in e["sources"]}
    assert by_aid["a1"]["display_date"] == "20260101T120000.000Z"
    assert by_aid["a2"]["display_date"] == "20260201T080000.000Z"
    # partial map: a3 (the Gold edge source) is absent from dd -> falls back to "".
    gold = next(ed for ed in data["edges"] if ed["target"] == "Gold")
    assert gold["sources"][0]["display_date"] == ""


def test_serialize_graph_display_date_empty_without_map():
    g = _g()
    data = serialize_graph(g)  # no display_dates arg — must not error
    for e in data["edges"]:
        for s in e["sources"]:
            assert s["display_date"] == ""


def test_build_graph_carries_id_and_invalidated_by(tmp_path):
    import json
    from kg.build_graph import build_graph
    rows = [
        {"article_id": "a1", "date": "2014-05-01", "headline": "H1", "events": [
            {"id": "E1", "article_id": "a1", "statement": "Fed will raise.",
             "statement_type": "PREDICTION", "temporal_type": "DYNAMIC",
             "valid_at": "2014-05-01T00:00:00Z", "invalid_at": "2014-05-02T00:00:00Z",
             "invalidated_by": "E2", "evidence_paragraphs": [0],
             "triplets": [{"subject": "Fed", "subject_type": "CENTRAL_BANK", "relation": "RAISES",
                           "object": "Rate", "object_type": "INTEREST_RATE", "value": None}]}]},
        {"article_id": "a2", "date": "2014-05-02", "headline": "H2", "events": [
            {"id": "E2", "article_id": "a2", "statement": "Fed held.",
             "statement_type": "FACT", "temporal_type": "STATIC",
             "valid_at": "2014-05-02T00:00:00Z", "invalid_at": None,
             "invalidated_by": None, "evidence_paragraphs": [0],
             "triplets": [{"subject": "Fed", "subject_type": "CENTRAL_BANK", "relation": "LEAVES_UNCHANGED",
                           "object": "Rate", "object_type": "INTEREST_RATE", "value": None}]}]},
    ]
    p = tmp_path / "e.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    g = build_graph(p)
    raises = next(d for _, _, d in g.edges(data=True) if d["relation"] == "RAISES")
    s = raises["sources"][0]
    assert s["id"] == "E1" and s["invalidated_by"] == "E2"


def test_serialize_threads_supersession_leanly():
    g = nx.MultiDiGraph()
    g.add_node("Fed", entity_type="CENTRAL_BANK", source_articles=["a1", "a2"])
    g.add_node("Rate", entity_type="INTEREST_RATE", source_articles=["a1", "a2"])
    # superseded (older) fact: RAISES, invalidated_by = the superseder id "E2"
    g.add_edge("Fed", "Rate", key="RAISES", relation="RAISES", count=1,
               sources=[{"article_id": "a1", "date": "2014-05-01", "id": "E1", "invalidated_by": "E2"}])
    # superseder (newer) fact: LEAVES_UNCHANGED, id "E2", not itself superseded
    g.add_edge("Fed", "Rate", key="LEAVES_UNCHANGED", relation="LEAVES_UNCHANGED", count=1,
               sources=[{"article_id": "a2", "date": "2014-05-02", "id": "E2", "invalidated_by": None}])
    data = serialize_graph(g)
    old = next(e for e in data["edges"] if e["relationType"] == "RAISES")["sources"][0]
    new = next(e for e in data["edges"] if e["relationType"] == "LEAVES_UNCHANGED")["sources"][0]
    assert old["invalidated_by"] == "E2"   # superseded fact points at its superseder
    assert "id" not in old                 # E1 is not referenced by anyone -> id omitted (lean)
    assert new["id"] == "E2"               # E2 IS referenced -> its id is emitted
    assert "invalidated_by" not in new     # not superseded -> field omitted (lean)
