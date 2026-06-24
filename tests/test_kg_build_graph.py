"""Tests for src/kg/build_graph.py edge-source provenance."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kg.build_graph import build_graph  # noqa: E402


def test_build_graph_carries_statement_and_value(tmp_path):
    """Each edge source must carry the parent statement text + the triplet's
    value (alongside the existing temporal fields), for the provenance modal."""
    row = {
        "article_id": "a1", "date": "2014-05-01", "headline": "Oil up",
        "events": [{
            "statement": "Oil prices rose on supply concerns.",
            "statement_type": "FACT", "temporal_type": "DYNAMIC",
            "valid_at": "2014-05-01T00:00:00", "invalid_at": None,
            "evidence_paragraphs": [2],
            "triplets": [{
                "subject": "supply concerns", "subject_type": "CONCEPT",
                "relation": "CAUSES_RISE_IN",
                "object": "Crude Oil", "object_type": "COMMODITY",
                "value": "sharply",
            }],
        }],
    }
    p = tmp_path / "k.jsonl"
    p.write_text(json.dumps(row) + "\n")
    g = build_graph(p)
    src = g.edges[("Supply Concerns", "Crude Oil", "CAUSES_RISE_IN")]["sources"][0]
    assert src["statement"] == "Oil prices rose on supply concerns."
    assert src["value"] == "sharply"
    assert src["valid_at"] == "2014-05-01T00:00:00"
    assert src["statement_type"] == "FACT"
