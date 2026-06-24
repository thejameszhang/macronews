import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kg.build_graph import build_graph  # noqa: E402
from kg.schemas import (  # noqa: E402
    ASSET_GROUP_NODE_TYPE, ENTITY_TYPES_TUPLE, RELATION_TYPES_TUPLE,
)


def test_asset_group_constants():
    assert ASSET_GROUP_NODE_TYPE == "ASSET_GROUP"
    # both synthetic: kept OUT of the extractor's entity/relation type sets.
    assert ASSET_GROUP_NODE_TYPE not in ENTITY_TYPES_TUPLE
    assert "RELATED_TO_ASSET_GROUP" not in RELATION_TYPES_TUPLE


def test_build_graph_materializes_anchor_and_membership(tmp_path):
    kg = tmp_path / "k.jsonl"
    kg.write_text(json.dumps({
        "article_id": "20140501000001", "date": "2014-05-01", "headline": "Oil up",
        "events": [{"statement": "Oil rose.", "evidence_paragraphs": [0],
                    "triplets": [{"subject": "supply", "subject_type": "CONCEPT",
                                  "relation": "CAUSES_RISE_IN", "object": "Crude Oil",
                                  "object_type": "COMMODITY", "value": None}]}]}) + "\n")
    eg = tmp_path / "eg.json"
    eg.write_text(json.dumps({"crude oil": {"type": "COMMODITY",
                                            "groups": [{"key": "crude_oil", "method": "exact"}]}}))
    g = build_graph(kg, entity_groups_path=eg)
    anchor = "[ASSET_GROUP] Crude Oil"
    assert anchor in g.nodes and g.nodes[anchor]["entity_type"] == "ASSET_GROUP"
    assert g.has_edge("Crude Oil", anchor)
    d = next(d for _, _, d in g.edges("Crude Oil", data=True)
             if d["relation"] == "RELATED_TO_ASSET_GROUP")
    assert d["synthetic"] is True and d["method"] == "exact"
    assert d["sources"][0]["date"] == "2014-05-01"        # provenance from accession prefix
    assert d["sources"][0]["headline"] == "Oil up"        # headline joined from the article row


def test_build_graph_without_entity_groups_unchanged(tmp_path):
    kg = tmp_path / "k.jsonl"
    kg.write_text(json.dumps({"article_id": "a1", "events": [
        {"statement": "x", "evidence_paragraphs": [0], "triplets": [
            {"subject": "A", "subject_type": "CONCEPT", "relation": "RELATED_TO",
             "object": "B", "object_type": "CONCEPT", "value": None}]}]}) + "\n")
    g = build_graph(kg)
    assert not any(d.get("relation") == "RELATED_TO_ASSET_GROUP"
                   for _, _, d in g.edges(data=True))
