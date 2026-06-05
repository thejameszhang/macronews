"""Tests for src/kg/disambiguate.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from kg.disambiguate import (  # noqa: E402
    collect_entity_counts,
    cluster_within_type,
    select_canonical,
    disambiguate,
)


# --- collect_entity_counts ---

def test_collect_counts_basic():
    rows = [
        {"facts": [
            {"subject": "Fed", "subject_type": "CENTRAL_BANK",
             "object": "FFR", "object_type": "INTEREST_RATE"},
        ]},
    ]
    counts = collect_entity_counts(rows)
    assert counts == {
        ("Fed", "CENTRAL_BANK"): 1,
        ("FFR", "INTEREST_RATE"): 1,
    }


def test_collect_counts_aggregates_across_rows():
    rows = [
        {"facts": [
            {"subject": "Fed", "subject_type": "CENTRAL_BANK",
             "object": "FFR", "object_type": "INTEREST_RATE"},
        ]},
        {"facts": [
            {"subject": "Fed", "subject_type": "CENTRAL_BANK",
             "object": "USD", "object_type": "CURRENCY"},
        ]},
    ]
    counts = collect_entity_counts(rows)
    assert counts[("Fed", "CENTRAL_BANK")] == 2
    assert counts[("FFR", "INTEREST_RATE")] == 1
    assert counts[("USD", "CURRENCY")] == 1


def test_collect_counts_handles_empty_facts():
    rows = [{"facts": []}, {}]
    assert collect_entity_counts(rows) == {}


def test_collect_counts_same_name_different_type_counted_separately():
    rows = [
        {"facts": [
            {"subject": "Apple", "subject_type": "COMPANY",
             "object": "iPhone", "object_type": "FIN_INSTRUMENT"},
            {"subject": "Apple", "subject_type": "COMMODITY",
             "object": "Fruit", "object_type": "CONCEPT"},
        ]},
    ]
    counts = collect_entity_counts(rows)
    assert counts[("Apple", "COMPANY")] == 1
    assert counts[("Apple", "COMMODITY")] == 1


# --- cluster_within_type ---

def test_cluster_single_entity():
    embs = np.array([[1.0, 0.0]])
    clusters = cluster_within_type(["solo"], embs, threshold=0.5)
    assert clusters == [["solo"]]


def test_cluster_empty_input():
    embs = np.empty((0, 4))
    assert cluster_within_type([], embs, threshold=0.5) == []


def test_cluster_two_similar_names_merge():
    # Two near-identical unit vectors → cosine ~1.0 → merge above 0.9.
    embs = np.array([[1.0, 0.0], [0.99, 0.01]])
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    clusters = cluster_within_type(["A", "B"], embs, threshold=0.9)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == ["A", "B"]


def test_cluster_dissimilar_names_stay_separate():
    # Orthogonal vectors → cosine 0 → no merge even at threshold 0.5.
    embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    clusters = cluster_within_type(["A", "B"], embs, threshold=0.5)
    assert len(clusters) == 2


def test_cluster_complete_linkage_no_transitive_merge():
    """Complete-linkage must not merge A,B,C when A~B and B~C but A!~C.

    Construct three unit vectors with known pairwise cosines:
      A.B ~ 0.80, B.C ~ 0.80, A.C ~ 0.50.
    With threshold 0.7, complete-linkage requires ALL pairs > 0.7 to form
    a single cluster. A.C = 0.50 < 0.7 → A and C cannot be in one cluster.
    Expected: at most 2-element clusters (e.g., {A,B} + {C} or {B,C} + {A}).
    """
    A = np.array([1.0, 0.0, 0.0])
    C = np.array([0.5, np.sqrt(0.75), 0.0])  # |C| = 1, A.C = 0.5
    B = np.array([0.8, 0.462, 0.383])         # |B| = 1, A.B = 0.8, B.C ~ 0.8
    A /= np.linalg.norm(A)
    B /= np.linalg.norm(B)
    C /= np.linalg.norm(C)
    embs = np.stack([A, B, C])

    clusters = cluster_within_type(["A", "B", "C"], embs, threshold=0.7)
    # Must NOT collapse all three into one cluster.
    assert all(len(c) <= 2 for c in clusters), \
        f"complete-linkage should prevent transitive merge: got {clusters}"


# --- select_canonical ---

def test_canonical_most_frequent_wins():
    freq = {"short": 10, "longer name": 3}
    assert select_canonical(["short", "longer name"], freq) == "short"


def test_canonical_tiebreak_by_length():
    freq = {"Fed": 5, "Federal Reserve": 5}
    assert select_canonical(["Fed", "Federal Reserve"], freq) == "Federal Reserve"


def test_canonical_single_element():
    freq = {"sole": 1}
    assert select_canonical(["sole"], freq) == "sole"


# --- disambiguate end-to-end (with mocked SentenceTransformer) ---

def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _patch_st_model(monkeypatch, encode_fn):
    """Patch SentenceTransformer so encode() returns whatever encode_fn returns
    for a given list of names. Avoids loading the real model."""
    class MockModel:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, names, **kwargs):
            return encode_fn(names)
    monkeypatch.setattr("kg.disambiguate.SentenceTransformer", MockModel)


def test_disambiguate_roundtrip_no_merges(tmp_path, monkeypatch):
    rows = [
        {"article_id": "a1", "date": "2026-01-01", "headline": "h",
         "facts": [
             {"evidence_paragraphs": [0], "subject": "Fed",
              "subject_type": "CENTRAL_BANK", "relation": "RAISES",
              "object": "FFR", "object_type": "INTEREST_RATE"},
         ],
         "paragraphs": {"0": "para 0"}},
    ]
    input_file = tmp_path / "input.jsonl"
    _write_jsonl(input_file, rows)
    _patch_st_model(monkeypatch, lambda names: np.eye(len(names)))

    output_file = tmp_path / "output.jsonl"
    summary = disambiguate(input_file, output_file, threshold=0.5)

    out_rows = [json.loads(line) for line in
                output_file.read_text().splitlines() if line.strip()]
    assert out_rows == rows
    assert summary["merges_n"] == 0


def test_disambiguate_merges_duplicate_central_banks(tmp_path, monkeypatch):
    rows = [
        {"article_id": "a1", "date": "2026-01-01", "headline": "h",
         "facts": [
             {"evidence_paragraphs": [0], "subject": "Fed",
              "subject_type": "CENTRAL_BANK", "relation": "RAISES",
              "object": "FFR", "object_type": "INTEREST_RATE"},
         ],
         "paragraphs": {"0": "para 0"}},
        {"article_id": "a2", "date": "2026-01-02", "headline": "h2",
         "facts": [
             {"evidence_paragraphs": [1], "subject": "Federal Reserve",
              "subject_type": "CENTRAL_BANK", "relation": "RAISES",
              "object": "FFR", "object_type": "INTEREST_RATE"},
         ],
         "paragraphs": {"1": "para 1"}},
    ]
    input_file = tmp_path / "input.jsonl"
    _write_jsonl(input_file, rows)

    def encode_fn(names):
        embs = []
        for n in names:
            if n in ("Fed", "Federal Reserve"):
                embs.append([1.0, 0.0, 0.0])
            elif n == "FFR":
                embs.append([0.0, 1.0, 0.0])
            else:
                embs.append([0.0, 0.0, 1.0])
        return np.array(embs)

    _patch_st_model(monkeypatch, encode_fn)
    output_file = tmp_path / "output.jsonl"
    summary = disambiguate(input_file, output_file, threshold=0.9)

    out_rows = [json.loads(line) for line in
                output_file.read_text().splitlines() if line.strip()]
    # Both rows now reference the longer name (length tiebreak).
    assert out_rows[0]["facts"][0]["subject"] == "Federal Reserve"
    assert out_rows[1]["facts"][0]["subject"] == "Federal Reserve"
    assert summary["merges_n"] == 1


def test_disambiguate_does_not_merge_across_types(tmp_path, monkeypatch):
    """Two entities with the IDENTICAL surface form but different types must
    stay distinct. Uses the same string "Apple" in both COMPANY and COMMODITY
    so the only thing preventing a merge is type-blocking."""
    rows = [
        {"article_id": "a1", "date": "2026-01-01", "headline": "h",
         "facts": [
             {"evidence_paragraphs": [0], "subject": "Apple",
              "subject_type": "COMPANY", "relation": "PRODUCES",
              "object": "iPhone", "object_type": "FIN_INSTRUMENT"},
             {"evidence_paragraphs": [0], "subject": "Apple",
              "subject_type": "COMMODITY", "relation": "RELATED_TO",
              "object": "Fruit", "object_type": "CONCEPT"},
         ],
         "paragraphs": {"0": "para 0"}},
    ]
    input_file = tmp_path / "input.jsonl"
    _write_jsonl(input_file, rows)

    def encode_fn(names):
        # Single embedding for "Apple" regardless of which type called it.
        # If type-blocking is broken, the two would land in one cluster.
        embs = []
        for n in names:
            if n == "Apple":
                embs.append([1.0, 0.0, 0.0])
            else:
                embs.append([0.0, 1.0, 0.0])
        return np.array(embs)

    _patch_st_model(monkeypatch, encode_fn)
    output_file = tmp_path / "output.jsonl"
    summary = disambiguate(input_file, output_file, threshold=0.9)

    out_rows = [json.loads(line) for line in
                output_file.read_text().splitlines() if line.strip()]
    # Same surface form, different types -> 2 separate clusters, 0 merges.
    assert summary["clusters_n"] == 4   # Apple-COMPANY, Apple-COMMODITY, iPhone, Fruit
    assert summary["merges_n"] == 0
    # Both Apple facts still say "Apple" — neither was rewritten to the other.
    assert out_rows[0]["facts"][0]["subject"] == "Apple"
    assert out_rows[0]["facts"][1]["subject"] == "Apple"
    # And their types remained distinct.
    assert out_rows[0]["facts"][0]["subject_type"] == "COMPANY"
    assert out_rows[0]["facts"][1]["subject_type"] == "COMMODITY"


def test_disambiguate_preserves_metadata(tmp_path, monkeypatch):
    rows = [
        {"article_id": "a1", "date": "2026-01-01", "headline": "Test Headline",
         "facts": [
             {"evidence_paragraphs": [0, 1], "subject": "Fed",
              "subject_type": "CENTRAL_BANK", "relation": "RAISES",
              "object": "FFR", "object_type": "INTEREST_RATE"},
         ],
         "paragraphs": {"0": "para zero", "1": "para one"}},
    ]
    input_file = tmp_path / "input.jsonl"
    _write_jsonl(input_file, rows)
    _patch_st_model(monkeypatch, lambda names: np.eye(len(names)))

    output_file = tmp_path / "output.jsonl"
    disambiguate(input_file, output_file, threshold=0.5)

    out_rows = [json.loads(line) for line in
                output_file.read_text().splitlines() if line.strip()]
    assert out_rows[0]["article_id"] == "a1"
    assert out_rows[0]["date"] == "2026-01-01"
    assert out_rows[0]["headline"] == "Test Headline"
    assert out_rows[0]["paragraphs"] == {"0": "para zero", "1": "para one"}
    assert out_rows[0]["facts"][0]["evidence_paragraphs"] == [0, 1]
    # Per-fact type tags + relation must also pass through unchanged.
    fact_out = out_rows[0]["facts"][0]
    assert fact_out["subject_type"] == "CENTRAL_BANK"
    assert fact_out["object_type"] == "INTEREST_RATE"
    assert fact_out["relation"] == "RAISES"


def test_disambiguate_writes_clusters_sidecar(tmp_path, monkeypatch):
    rows = [
        {"article_id": "a1", "date": "2026-01-01", "headline": "h",
         "facts": [
             {"evidence_paragraphs": [0], "subject": "Fed",
              "subject_type": "CENTRAL_BANK", "relation": "RAISES",
              "object": "FFR", "object_type": "INTEREST_RATE"},
             {"evidence_paragraphs": [0], "subject": "Federal Reserve",
              "subject_type": "CENTRAL_BANK", "relation": "ANNOUNCES",
              "object": "Rate Hike", "object_type": "EVENT"},
         ],
         "paragraphs": {"0": "para"}},
    ]
    input_file = tmp_path / "input.jsonl"
    _write_jsonl(input_file, rows)

    def encode_fn(names):
        embs = []
        for n in names:
            if n in ("Fed", "Federal Reserve"):
                embs.append([1.0, 0.0, 0.0])
            elif n == "FFR":
                embs.append([0.0, 1.0, 0.0])
            else:
                embs.append([0.0, 0.0, 1.0])
        return np.array(embs)

    _patch_st_model(monkeypatch, encode_fn)
    output_file = tmp_path / "output.jsonl"
    clusters_file = tmp_path / "clusters.json"
    disambiguate(input_file, output_file, threshold=0.9,
                 clusters_sidecar=clusters_file)

    clusters = json.loads(clusters_file.read_text())
    # Nested by type now.
    assert "CENTRAL_BANK" in clusters
    assert "Federal Reserve" in clusters["CENTRAL_BANK"]
    assert sorted(clusters["CENTRAL_BANK"]["Federal Reserve"]) == ["Fed", "Federal Reserve"]


def test_disambiguate_titlecases_lowercase_canonical(tmp_path, monkeypatch):
    """When a lowercase surface form wins canonical selection (e.g. it's the
    most frequent), the stored canonical is title-cased so the JSONL and
    clusters sidecar are not ugly. Acronyms are preserved."""
    rows = [
        {"article_id": "a1", "date": "2026-01-01", "headline": "h",
         "facts": [
             # "middle east conflict" appears 2x (wins on frequency),
             # "Middle East Conflict" appears 1x.
             {"evidence_paragraphs": [0], "subject": "middle east conflict",
              "subject_type": "EVENT", "relation": "CAUSES_RISE_IN",
              "object": "Crude Oil", "object_type": "COMMODITY"},
         ],
         "paragraphs": {"0": "p"}},
        {"article_id": "a2", "date": "2026-01-02", "headline": "h",
         "facts": [
             {"evidence_paragraphs": [0], "subject": "middle east conflict",
              "subject_type": "EVENT", "relation": "CAUSES_RISE_IN",
              "object": "Crude Oil", "object_type": "COMMODITY"},
         ],
         "paragraphs": {"0": "p"}},
        {"article_id": "a3", "date": "2026-01-03", "headline": "h",
         "facts": [
             {"evidence_paragraphs": [0], "subject": "Middle East Conflict",
              "subject_type": "EVENT", "relation": "CAUSES_RISE_IN",
              "object": "Crude Oil", "object_type": "COMMODITY"},
         ],
         "paragraphs": {"0": "p"}},
    ]
    input_file = tmp_path / "input.jsonl"
    _write_jsonl(input_file, rows)

    def encode_fn(names):
        # Both conflict variants identical vector -> merge; Crude Oil distinct.
        embs = []
        for n in names:
            if "conflict" in n.lower():
                embs.append([1.0, 0.0])
            else:
                embs.append([0.0, 1.0])
        return np.array(embs)

    _patch_st_model(monkeypatch, encode_fn)
    output_file = tmp_path / "output.jsonl"
    clusters_file = tmp_path / "clusters.json"
    disambiguate(input_file, output_file, threshold=0.9,
                 clusters_sidecar=clusters_file)

    out_rows = [json.loads(line) for line in
                output_file.read_text().splitlines() if line.strip()]
    # Even though the lowercase form won on frequency, the stored canonical
    # is title-cased.
    for row in out_rows:
        assert row["facts"][0]["subject"] == "Middle East Conflict"
    clusters = json.loads(clusters_file.read_text())
    assert "Middle East Conflict" in clusters["EVENT"]


def test_disambiguate_preserves_acronyms_in_canonical(tmp_path, monkeypatch):
    """Title-casing must not mangle acronyms: 'U.S. CPI Inflation' stays
    exactly as-is (not 'U.s. Cpi Inflation')."""
    rows = [
        {"article_id": "a1", "date": "2026-01-01", "headline": "h",
         "facts": [
             {"evidence_paragraphs": [0], "subject": "U.S. CPI Inflation",
              "subject_type": "ECON_INDICATOR", "relation": "CAUSES_RISE_IN",
              "object": "Gold", "object_type": "COMMODITY"},
         ],
         "paragraphs": {"0": "p"}},
    ]
    input_file = tmp_path / "input.jsonl"
    _write_jsonl(input_file, rows)
    _patch_st_model(monkeypatch, lambda names: np.eye(len(names)))

    output_file = tmp_path / "output.jsonl"
    disambiguate(input_file, output_file, threshold=0.9)

    out_rows = [json.loads(line) for line in
                output_file.read_text().splitlines() if line.strip()]
    assert out_rows[0]["facts"][0]["subject"] == "U.S. CPI Inflation"
