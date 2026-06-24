"""Tests for src/kg/disambiguate.py."""

import json
import sys
from pathlib import Path

import numpy as np

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
        {"events": [{"triplets": [
            {"subject": "Fed", "subject_type": "CENTRAL_BANK",
             "object": "FFR", "object_type": "INTEREST_RATE"},
        ]}]},
    ]
    counts = collect_entity_counts(rows)
    assert counts == {
        ("Fed", "CENTRAL_BANK"): 1,
        ("FFR", "INTEREST_RATE"): 1,
    }


def test_collect_counts_aggregates_across_rows():
    rows = [
        {"events": [{"triplets": [
            {"subject": "Fed", "subject_type": "CENTRAL_BANK",
             "object": "FFR", "object_type": "INTEREST_RATE"},
        ]}]},
        {"events": [{"triplets": [
            {"subject": "Fed", "subject_type": "CENTRAL_BANK",
             "object": "USD", "object_type": "CURRENCY"},
        ]}]},
    ]
    counts = collect_entity_counts(rows)
    assert counts[("Fed", "CENTRAL_BANK")] == 2
    assert counts[("FFR", "INTEREST_RATE")] == 1
    assert counts[("USD", "CURRENCY")] == 1


def test_collect_counts_handles_empty_events():
    rows = [{"events": []}, {}]
    assert collect_entity_counts(rows) == {}


def test_collect_counts_same_name_different_type_counted_separately():
    rows = [
        {"events": [{"triplets": [
            {"subject": "Apple", "subject_type": "COMPANY",
             "object": "iPhone", "object_type": "FIN_INSTRUMENT"},
            {"subject": "Apple", "subject_type": "COMMODITY",
             "object": "Fruit", "object_type": "CONCEPT"},
        ]}]},
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


# --- Rule A normalization pre-merge ---

def test_norm_key_rule_a():
    from kg.disambiguate import _norm_key
    assert _norm_key("Natural-gas") == _norm_key("Natural Gas") == "natural gas"
    assert _norm_key("Foreign-Exchange") == _norm_key("foreign exchange") == "foreign exchange"
    # hyphen becomes a SPACE, not nothing: 'e-mini' -> 'e mini', not 'emini'
    assert _norm_key("e-mini") == "e mini" and _norm_key("emini") == "emini"


def test_cluster_normalized_variants_merge_below_threshold():
    # Rule A: 'Natural-gas' / 'Natural Gas' share a normalized key, so they merge
    # even though their embeddings are only 0.866 cosine (< the 0.90 threshold).
    embs = np.array([[1.0, 0.0], [0.866, 0.5]])   # cosine = 0.866
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    clusters = cluster_within_type(["Natural Gas", "Natural-gas"], embs, threshold=0.9)
    assert len(clusters) == 1
    assert sorted(clusters[0]) == ["Natural Gas", "Natural-gas"]


def test_cluster_hyphen_not_merged_with_concatenation():
    # Hyphen -> space (not removal): 'e-mini' ('e mini') must NOT collapse into
    # 'emini'. With dissimilar embeddings the two stay in separate clusters.
    embs = np.array([[1.0, 0.0], [0.0, 1.0]])     # orthogonal -> cosine 0
    clusters = cluster_within_type(["e-mini", "emini"], embs, threshold=0.9)
    assert len(clusters) == 2


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


def _trip(subj, stype, rel, obj, otype):
    """Minimal triplet dict for test rows."""
    return {"subject": subj, "subject_type": stype, "relation": rel,
            "object": obj, "object_type": otype, "value": None}


def _ev_row_meta(article_id, date, triplets, *, headline=None, paragraphs=None):
    """Events-format row with optional metadata fields."""
    row = {"article_id": article_id, "date": date,
           "events": [{"triplets": triplets}]}
    if headline is not None:
        row["headline"] = headline
    if paragraphs is not None:
        row["paragraphs"] = paragraphs
    return row


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
        _ev_row_meta("a1", "2026-01-01",
                     [_trip("Fed", "CENTRAL_BANK", "RAISES", "FFR", "INTEREST_RATE")]),
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
        _ev_row_meta("a1", "2026-01-01",
                     [_trip("Fed", "CENTRAL_BANK", "RAISES", "FFR", "INTEREST_RATE")]),
        _ev_row_meta("a2", "2026-01-02",
                     [_trip("Federal Reserve", "CENTRAL_BANK", "RAISES", "FFR", "INTEREST_RATE")]),
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
    assert out_rows[0]["events"][0]["triplets"][0]["subject"] == "Federal Reserve"
    assert out_rows[1]["events"][0]["triplets"][0]["subject"] == "Federal Reserve"
    assert summary["merges_n"] == 1


def test_disambiguate_does_not_merge_across_types(tmp_path, monkeypatch):
    """Two entities with the IDENTICAL surface form but different types must
    stay distinct. Uses the same string "Apple" in both COMPANY and COMMODITY
    so the only thing preventing a merge is type-blocking."""
    rows = [
        _ev_row_meta("a1", "2026-01-01", [
            _trip("Apple", "COMPANY", "PRODUCES", "iPhone", "FIN_INSTRUMENT"),
            _trip("Apple", "COMMODITY", "RELATED_TO", "Fruit", "CONCEPT"),
        ]),
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
    trips = out_rows[0]["events"][0]["triplets"]
    # Both Apple triplets still say "Apple" — neither was rewritten to the other.
    assert trips[0]["subject"] == "Apple"
    assert trips[1]["subject"] == "Apple"
    # And their types remained distinct.
    assert trips[0]["subject_type"] == "COMPANY"
    assert trips[1]["subject_type"] == "COMMODITY"


def test_disambiguate_preserves_metadata(tmp_path, monkeypatch):
    rows = [
        _ev_row_meta("a1", "2026-01-01",
                     [_trip("Fed", "CENTRAL_BANK", "RAISES", "FFR", "INTEREST_RATE")],
                     headline="Test Headline",
                     paragraphs={"0": "para zero", "1": "para one"}),
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
    # Per-triplet type tags + relation must pass through unchanged.
    trip_out = out_rows[0]["events"][0]["triplets"][0]
    assert trip_out["subject_type"] == "CENTRAL_BANK"
    assert trip_out["object_type"] == "INTEREST_RATE"
    assert trip_out["relation"] == "RAISES"


def test_disambiguate_writes_clusters_sidecar(tmp_path, monkeypatch):
    rows = [
        _ev_row_meta("a1", "2026-01-01", [
            _trip("Fed", "CENTRAL_BANK", "RAISES", "FFR", "INTEREST_RATE"),
            _trip("Federal Reserve", "CENTRAL_BANK", "ANNOUNCES", "Rate Hike", "EVENT"),
        ]),
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
    # "middle east conflict" appears 2x (wins on frequency),
    # "Middle East Conflict" appears 1x.
    rows = [
        _ev_row_meta("a1", "2026-01-01",
                     [_trip("middle east conflict", "EVENT", "CAUSES_RISE_IN", "Crude Oil", "COMMODITY")]),
        _ev_row_meta("a2", "2026-01-02",
                     [_trip("middle east conflict", "EVENT", "CAUSES_RISE_IN", "Crude Oil", "COMMODITY")]),
        _ev_row_meta("a3", "2026-01-03",
                     [_trip("Middle East Conflict", "EVENT", "CAUSES_RISE_IN", "Crude Oil", "COMMODITY")]),
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
        assert row["events"][0]["triplets"][0]["subject"] == "Middle East Conflict"
    clusters = json.loads(clusters_file.read_text())
    assert "Middle East Conflict" in clusters["EVENT"]


def test_disambiguate_preserves_acronyms_in_canonical(tmp_path, monkeypatch):
    """Title-casing must not mangle acronyms: 'U.S. CPI Inflation' stays
    exactly as-is (not 'U.s. Cpi Inflation')."""
    rows = [
        _ev_row_meta("a1", "2026-01-01",
                     [_trip("U.S. CPI Inflation", "ECON_INDICATOR", "CAUSES_RISE_IN", "Gold", "COMMODITY")]),
    ]
    input_file = tmp_path / "input.jsonl"
    _write_jsonl(input_file, rows)
    _patch_st_model(monkeypatch, lambda names: np.eye(len(names)))

    output_file = tmp_path / "output.jsonl"
    disambiguate(input_file, output_file, threshold=0.9)

    out_rows = [json.loads(line) for line in
                output_file.read_text().splitlines() if line.strip()]
    assert out_rows[0]["events"][0]["triplets"][0]["subject"] == "U.S. CPI Inflation"


# ---------------------------------------------------------------------------
# Rule-A guard: pre-collapse must survive the cluster_by_cosine migration.
# ---------------------------------------------------------------------------

def _unit_rows(rows):
    a = np.asarray(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_rule_a_merges_hyphen_case_variants_below_threshold():
    # "Natural Gas" and "Natural-gas" share a _norm_key -> pre-merged unconditionally,
    # even though their embeddings are only 0.86 cosine (below the 0.90 cutoff).
    # "Gold" (orthogonal) stays separate.
    names = ["Natural Gas", "Natural-gas", "Gold"]
    embs = _unit_rows([[1, 0, 0], [0.86, 0.51, 0], [0, 1, 0]])
    clusters = cluster_within_type(names, embs, threshold=0.90)
    cluster_sets = sorted([frozenset(c) for c in clusters], key=lambda s: sorted(s)[0])
    assert frozenset({"Natural Gas", "Natural-gas"}) in cluster_sets
    assert frozenset({"Gold"}) in cluster_sets


# ---------------------------------------------------------------------------
# Events-format tests (new sidecar schema: rows have "events"/"triplets",
# no "facts" key).
# ---------------------------------------------------------------------------

def _ev_row(subj, stype):
    """Minimal events-format row with a single triplet."""
    return {"article_id": "a", "date": "2014-05-01", "events": [
        {"article_id": "a", "statement": "s", "statement_type": "FACT",
         "temporal_type": "STATIC", "created_at": "2014-05-01T00:00:00",
         "triplets": [{"subject": subj, "subject_type": stype, "relation": "RAISES",
                       "object": "Rate", "object_type": "INTEREST_RATE",
                       "value": None}]}]}


def _subjects(out_path):
    return {t["subject"]
            for line in out_path.read_text().splitlines()
            for ev in json.loads(line)["events"] for t in ev["triplets"]}


def test_events_format_does_not_merge_acronym_with_expansion(tmp_path):
    # The acronym sub-stage was removed: an acronym and its spelled-out form are
    # NOT merged (they embed far apart and there is no initials rule), so "ECB"
    # and "European Central Bank" stay distinct. The events output format is
    # unaffected.
    rows = [_ev_row("European Central Bank", "CENTRAL_BANK")] * 3 + \
           [_ev_row("ECB", "CENTRAL_BANK")]
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "out.jsonl"
    disambiguate(src, out, threshold=0.95)
    subjects = _subjects(out)
    assert len(subjects) == 2
    assert "ECB" in subjects and "European Central Bank" in subjects
    first = json.loads(out.read_text().splitlines()[0])
    assert "events" in first and "facts" not in first


def test_default_rejected_path_helper():
    from kg.disambiguate import _default_self_ref_rejected
    from pathlib import Path
    assert _default_self_ref_rejected(Path("/x/2014-05.relv3.disambig.jsonl")) == \
        Path("/x/2014-05.relv3.disambig.self_ref_rejected.jsonl")


def test_disambiguate_self_reference_filter(tmp_path, monkeypatch):
    import json, numpy as np
    from kg.disambiguate import disambiguate
    rows = [{
        "article_id": "a1", "date": "20140501", "headline": "h",
        "events": [{
            "id": "e1", "article_id": "a1", "statement": "Brent fell",
            "triplets": [
                # Tier-1 exact self-loop (any relation)
                {"subject": "EUR/USD", "subject_type": "CURRENCY", "relation": "DECREASES",
                 "object": "EUR/USD", "object_type": "CURRENCY", "value": None},
                # Tier-2 same-asset, cross-type (high cosine), directional -> dropped
                {"subject": "Brent Crude Oil", "subject_type": "COMMODITY", "relation": "DECREASES",
                 "object": "Brent Crude Price", "object_type": "ASSET_METRIC", "value": "108"},
                # cross-asset (orthogonal) -> kept
                {"subject": "Brent Crude Oil", "subject_type": "COMMODITY", "relation": "IMPACT",
                 "object": "Gold", "object_type": "COMMODITY", "value": None},
            ],
        }],
    }]
    input_file = tmp_path / "in.jsonl"
    input_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
    output_file = tmp_path / "out.jsonl"
    rejected_file = tmp_path / "rej.jsonl"

    # Controlled embeddings: Brent Oil ~ Brent Price (cos ~0.997), everything else orthogonal.
    vecs = {
        "EUR/USD": [1, 0, 0, 0],
        "Brent Crude Oil": [0, 1, 0, 0],
        "Brent Crude Price": [0, 0.92, 0.39, 0],   # dot with Brent Oil = 0.92 >= 0.85
        "Gold": [0, 0, 0, 1],
    }
    def encode_fn(names):
        assert set(names) <= set(vecs), f"unexpected entity in encode: {set(names) - set(vecs)}"
        return np.array([np.array(vecs[n], dtype=float) /
                         np.linalg.norm(vecs[n]) for n in names])
    _patch_st_model(monkeypatch, encode_fn)

    summary = disambiguate(input_file, output_file, threshold=0.99,   # 0.99: no entity merges
                           self_ref_threshold=0.85, self_ref_rejected_jsonl=rejected_file)

    out = [json.loads(l) for l in output_file.read_text().splitlines()]
    kept = out[0]["events"][0]["triplets"]
    assert len(kept) == 1 and kept[0]["object"] == "Gold"            # only cross-asset survives
    rej = [json.loads(l) for l in rejected_file.read_text().splitlines()]
    assert len(rej) == 2
    assert {r["self_ref_reason"] for r in rej} == {"self_loop", "same_asset_cosine"}
    assert summary["self_ref_dropped"] == 2
