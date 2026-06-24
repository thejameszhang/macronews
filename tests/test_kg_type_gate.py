import json
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kg.type_gate import gate_article  # noqa: E402


def _article():
    return {"article_id": "a1", "events": [
        {"id": "e1", "statement": "oil fell in the US", "triplets": [
            {"subject": "oil prices", "subject_type": "COMMODITY", "relation": "CAUSES_FALL_IN",
             "object": "United States", "object_type": "GPE", "value": None},          # VIOLATION
            {"subject": "supply glut", "subject_type": "CONCEPT", "relation": "CAUSES_FALL_IN",
             "object": "oil price", "object_type": "ASSET_METRIC", "value": None},      # legal
        ]},
        {"id": "e2", "statement": "Fed raised rates", "triplets": [
            {"subject": "Federal Reserve", "subject_type": "CENTRAL_BANK", "relation": "RAISES",
             "object": "policy rate", "object_type": "INTEREST_RATE", "value": "to 1%"}, # legal
        ]},
    ]}


def test_split_is_non_lossy():
    art = _article()
    cleaned, rejected = gate_article(art)
    kept = sum(len(e["triplets"]) for e in cleaned["events"])
    inp = sum(len(e["triplets"]) for e in art["events"])
    assert kept + len(rejected) == inp


def test_violation_moved_with_reason_and_provenance():
    cleaned, rejected = gate_article(_article())
    assert len(rejected) == 1
    r = rejected[0]
    assert r["object_type"] == "GPE" and "GPE" in r["type_violation"]
    assert r["article_id"] == "a1" and r["event_id"] == "e1"
    assert r["relation"] == "CAUSES_FALL_IN"
    assert set(r.keys()) == {
        "article_id", "event_id", "statement", "subject", "subject_type",
        "relation", "object", "object_type", "value", "type_violation",
    }


def test_zero_triplet_event_survives():
    art = {"article_id": "a", "events": [{"id": "e", "statement": "s", "triplets": [
        {"subject": "x", "subject_type": "COMMODITY", "relation": "CAUSES_RISE_IN",
         "object": "Putin", "object_type": "PERSON", "value": None}]}]}   # only triplet is a violation
    cleaned, rejected = gate_article(art)
    assert len(cleaned["events"]) == 1
    assert cleaned["events"][0]["triplets"] == []
    assert len(rejected) == 1


def test_original_article_not_mutated():
    art = _article()
    gate_article(art)
    assert len(art["events"][0]["triplets"]) == 2   # input untouched


def test_cli_round_trip(tmp_path):
    src = tmp_path / "events.jsonl"
    src.write_text(json.dumps({"article_id": "a1", "events": [
        {"id": "e1", "statement": "s", "triplets": [
            {"subject": "oil", "subject_type": "COMMODITY", "relation": "CAUSES_FALL_IN",
             "object": "US", "object_type": "GPE", "value": None},
            {"subject": "supply", "subject_type": "CONCEPT", "relation": "CAUSES_FALL_IN",
             "object": "oil price", "object_type": "ASSET_METRIC", "value": None},
        ]}]}) + "\n")
    clean = tmp_path / "clean.jsonl"
    rej = tmp_path / "rej.jsonl"
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "kg.type_gate", str(src),
         "--clean", str(clean), "--rejected", str(rej), "--summary"],
        cwd=repo, env={**os.environ, "PYTHONPATH": "src"}, check=True,
        capture_output=True, text=True)
    cleaned = [json.loads(x) for x in clean.read_text().splitlines() if x.strip()]
    rejected = [json.loads(x) for x in rej.read_text().splitlines() if x.strip()]
    assert len(cleaned) == 1
    assert len(cleaned[0]["events"][0]["triplets"]) == 1      # GPE one dropped
    assert len(rejected) == 1 and rejected[0]["object_type"] == "GPE"
    assert "CAUSES_FALL_IN" in result.stdout   # --summary prints the dropped relation
