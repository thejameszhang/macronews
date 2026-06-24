import json
import numpy as np
from kg.invalidate_llm import load_eligible_events, save_embeddings, load_embeddings


def _row(events):
    return {"article_id": "a", "date": "2014-05-02", "headline": "h", "events": events}


def _event(eid, temp="DYNAMIC", valid="2014-05-02T00:00:00Z"):
    return {"id": eid, "article_id": "a", "statement": f"stmt {eid}",
            "statement_type": "FACT", "temporal_type": temp, "triplets": [],
            "valid_at": valid, "invalid_at": None, "created_at": "2014-05-02T00:00:00Z",
            "expired_at": None, "invalidated_by": None}


def test_eligibility_filters_atemporal_and_null_valid(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = [_row([
        _event("e1", "DYNAMIC", "2014-05-02T00:00:00Z"),     # eligible
        _event("e2", "ATEMPORAL", "2014-05-02T00:00:00Z"),   # excluded (ATEMPORAL)
        _event("e3", "DYNAMIC", None),                        # excluded (null valid_at)
        _event("e4", "STATIC", "2014-05-03T00:00:00Z"),       # eligible (can be secondary)
    ])]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    eligible, allrows = load_eligible_events(p)
    assert {e["id"] for e in eligible} == {"e1", "e4"}
    assert len(allrows) == 1   # all source rows retained for non-lossy output


def test_embeddings_npz_roundtrip(tmp_path):
    p = tmp_path / "e.npz"
    ids = ["e1", "e2"]
    vecs = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    save_embeddings(p, ids, vecs)
    got = load_embeddings(p)
    assert np.allclose(got["e1"], [1, 0]) and np.allclose(got["e2"], [0, 1])


def test_inspect_candidates_dump(tmp_path):
    from kg.invalidate_llm import inspect_candidates
    events = [
        {"id": "e1", "statement": "Gold rises", "valid_at": "2014-05-02T00:00:00Z",
         "statement_type": "FACT", "triplets": [{"relation": "RAISES"}]},
        {"id": "e2", "statement": "Gold falls", "valid_at": "2014-05-03T00:00:00Z",
         "statement_type": "FACT", "triplets": [{"relation": "DECREASES"}]},
    ]
    by_id = {e["id"]: e for e in events}
    pairs = [("e1", "e2")]
    out = tmp_path / "candidates.txt"
    stats = inspect_candidates(pairs, by_id, out)
    assert stats["primaries_with_candidates"] == 1
    assert stats["total_pairs"] == 1
    text = out.read_text()
    assert "Gold rises" in text and "Gold falls" in text


def test_embeddings_npz_roundtrip_suffixless(tmp_path):
    # save_embeddings/load_embeddings must agree even when the path lacks ".npz".
    p = tmp_path / "emb"          # no .npz suffix
    save_embeddings(p, ["e1"], np.asarray([[1, 2, 3]], dtype=np.float32))
    got = load_embeddings(p)
    assert np.allclose(got["e1"], [1, 2, 3])


def test_invalidation_verdict_schema():
    from kg.invalidate_llm import InvalidationVerdict
    assert InvalidationVerdict.model_validate_json('{"invalidated": true}').invalidated is True
    assert InvalidationVerdict().invalidated is False


def test_render_pair_prompt_includes_triplets_conditionally():
    from kg.invalidate_llm import render_pair_prompt
    # No triplets -> no 'Triplet:' line at all (cookbook {% if primary_triplet %})
    p0 = {"statement": "Rates will rise", "valid_at": "2014-05-02T00:00:00Z", "invalid_at": None}
    s0 = {"statement": "Rates held steady", "valid_at": "2014-05-08T00:00:00Z", "invalid_at": None}
    prompt0 = render_pair_prompt(p0, s0)
    assert "Rates will rise" in prompt0 and "Rates held steady" in prompt0
    assert "2014-05-02T00:00:00Z" in prompt0 and "2014-05-08T00:00:00Z" in prompt0
    assert "Invalidation Guidelines" in prompt0 and 'Return: "True"' in prompt0
    assert "Triplet:" not in prompt0                              # absent when the event has no triplets
    # With triplets -> a 'Triplet:' line in cookbook (subject, relation, object) format, '; '-joined
    p1 = {"statement": "Rates will rise", "valid_at": "2014-05-02T00:00:00Z", "invalid_at": None,
          "triplets": [{"subject": "Fed", "relation": "RAISES", "object": "Rates"},
                       {"subject": "Fed", "relation": "RELATED_TO", "object": "Policy"}]}
    prompt1 = render_pair_prompt(p1, s0)
    assert "Triplet: (Fed, RAISES, Rates); (Fed, RELATED_TO, Policy)" in prompt1


from kg.invalidate_llm import build_pairs, apply_verdicts


def _e(eid, temp, valid, invalid=None, created="2014-05-01T00:00:00Z"):
    return {"id": eid, "temporal_type": temp, "valid_at": valid,
            "invalid_at": invalid, "created_at": created,
            "invalidated_by": None, "expired_at": None, "invalidation_method": None}


def _ev2(eid, valid, stype="FACT", temp="DYNAMIC", invalid=None):
    """Fixture event with a 'Gold' triplet so all events are entity-shared."""
    return {"id": eid, "statement": f"Gold statement {eid}",
            "statement_type": stype, "temporal_type": temp,
            "valid_at": valid, "invalid_at": invalid,
            "created_at": valid,
            "invalidated_by": None, "expired_at": None, "invalidation_method": None,
            "triplets": [{"subject": "Gold", "object": "Market",
                           "relation": "RAISES", "subject_type": "COMMODITY",
                           "object_type": "MARKET", "value": None}]}


def test_build_pairs_cookbook_funnel():
    """Primary with a temporal range produces pairs with later events inside that range."""
    # e1 is a DYNAMIC FACT with invalid_at so its range is [May-01, May-15]
    e1 = _ev2("e1", "2014-05-01T00:00:00Z", invalid="2014-05-15T00:00:00Z")
    # e2 and e3 fall within e1's range and start after e1's valid_at
    e2 = _ev2("e2", "2014-05-05T00:00:00Z")
    e3 = _ev2("e3", "2014-05-10T00:00:00Z")
    ref = np.array([1.0, 0.0], dtype=np.float32)
    emb = {"e1": ref.copy(), "e2": ref.copy(), "e3": ref.copy()}
    pairs = build_pairs([e1, e2, e3], emb)
    assert ("e1", "e2") in pairs
    assert ("e1", "e3") in pairs


def test_build_pairs_open_ended_dynamic_primary():
    """Regression: an open-ended DYNAMIC FACT primary (invalid_at=None) must still
    pair with a LATER FACT secondary that shares an entity. The old per-PRIMARY
    direction collapsed the open-ended primary to a point window (start==end) and
    found no later secondary, so the agent no-op'd on the bulk of primaries."""
    # p1: open-ended DYNAMIC FACT (invalid_at None), valid 2014-05-02
    p1 = _ev2("p1", "2014-05-02T00:00:00Z")
    # s1: later FACT secondary, valid 2014-05-20; shares "Gold" + near-identical emb
    s1 = _ev2("s1", "2014-05-20T00:00:00Z")
    ref = np.array([1.0, 0.0], dtype=np.float32)
    emb = {"p1": ref.copy(), "s1": ref.copy()}
    pairs = build_pairs([p1, s1], emb)
    assert ("p1", "s1") in pairs


def test_build_pairs_static_primary_not_eligible():
    """A STATIC FACT is not eligible as a primary."""
    e1 = _ev2("e1", "2014-05-01T00:00:00Z", temp="STATIC")   # STATIC FACT
    e2 = _ev2("e2", "2014-05-05T00:00:00Z")
    ref = np.array([1.0, 0.0], dtype=np.float32)
    emb = {"e1": ref.copy(), "e2": ref.copy()}
    pairs = build_pairs([e1, e2], emb)
    assert ("e1", "e2") not in pairs


def test_build_pairs_fact_primary_no_prediction_secondary():
    """FACT primary does NOT pair with a later PREDICTION (type rule enforced)."""
    # fact_p has range [May-01, May-15]; pred_s and fact_s both fall inside
    fact_p = _ev2("fp", "2014-05-01T00:00:00Z", invalid="2014-05-15T00:00:00Z")
    pred_s = _ev2("ps", "2014-05-05T00:00:00Z", stype="PREDICTION")
    fact_s = _ev2("fs", "2014-05-05T00:00:00Z", stype="FACT")
    ref = np.array([1.0, 0.0], dtype=np.float32)
    emb = {"fp": ref.copy(), "ps": ref.copy(), "fs": ref.copy()}
    pairs = build_pairs([fact_p, pred_s, fact_s], emb)
    assert ("fp", "ps") not in pairs   # FACT→PREDICTION forbidden
    assert ("fp", "fs") in pairs       # FACT→FACT allowed


def test_build_pairs_prediction_primary_pairs_with_prediction():
    """PREDICTION primary DOES pair with a later PREDICTION secondary."""
    pred_p = _ev2("pp", "2014-05-01T00:00:00Z", stype="PREDICTION",
                  invalid="2014-05-15T00:00:00Z")
    pred_s = _ev2("ps", "2014-05-05T00:00:00Z", stype="PREDICTION")
    ref = np.array([1.0, 0.0], dtype=np.float32)
    emb = {"pp": ref.copy(), "ps": ref.copy()}
    pairs = build_pairs([pred_p, pred_s], emb)
    assert ("pp", "ps") in pairs


def test_dedup_same_triplet_set_and_date_collapses():
    """Two events with the SAME set of (subject,relation,object) triplets AND same valid_at collapse
    to one rep — even differently worded (same-fact paraphrase). Different date stays separate."""
    from kg.invalidate_llm import dedup_representatives
    t = lambda s, o, r="RAISES": [{"subject": s, "object": o, "relation": r}]
    evs = [
        {"id": "a", "statement": "Oil up",   "valid_at": "2014-05-02T00:00:00Z", "triplets": t("Oil", "Mkt")},
        {"id": "b", "statement": "Oil rose", "valid_at": "2014-05-02T00:00:00Z", "triplets": t("Oil", "Mkt")},  # paraphrase
        {"id": "c", "statement": "Oil rose", "valid_at": "2014-05-09T00:00:00Z", "triplets": t("Oil", "Mkt")},  # later date
    ]
    reps, members = dedup_representatives(evs)
    assert [e["id"] for e in reps] == ["a", "c"]                # b folds into a (same triplet+date); c distinct (date)
    assert members == {"a": ["a", "b"], "c": ["c"]}


def test_dedup_full_set_not_first_triplet():
    """Regression (reviewer BLOCKING): key on the FULL triplet set, not triplets[0]. Two events
    sharing only their FIRST triplet but differing in a second are NOT merged."""
    from kg.invalidate_llm import dedup_representatives
    tr = lambda s, o, r="RAISES": {"subject": s, "object": o, "relation": r}
    evs = [
        {"id": "a", "statement": "Fed up, oil fell", "valid_at": "2014-05-02T00:00:00Z",
         "triplets": [tr("Fed", "Rates"), tr("Oil", "Mkt", "FALLS")]},
        {"id": "b", "statement": "Fed up, gold rose", "valid_at": "2014-05-02T00:00:00Z",
         "triplets": [tr("Fed", "Rates"), tr("Gold", "Mkt")]},   # shares triplets[0], differs in [1]
    ]
    reps, _ = dedup_representatives(evs)
    assert [e["id"] for e in reps] == ["a", "b"]                # NOT merged — full triplet sets differ


def test_dedup_empty_triplets_use_statement():
    """Regression (reviewer BLOCKING): triplet-less events fall back to (statement, valid_at) — same
    text collapses, different text stays separate (they do NOT all merge on the empty set + date)."""
    from kg.invalidate_llm import dedup_representatives
    evs = [
        {"id": "a", "statement": "X", "valid_at": "2014-05-02T00:00:00Z", "triplets": []},
        {"id": "b", "statement": "X", "valid_at": "2014-05-02T00:00:00Z", "triplets": []},   # same text+date
        {"id": "c", "statement": "Y", "valid_at": "2014-05-02T00:00:00Z", "triplets": []},   # diff text, same date
    ]
    reps, members = dedup_representatives(evs)
    assert [e["id"] for e in reps] == ["a", "c"]                # b folds into a; c distinct (text differs)
    assert members == {"a": ["a", "b"], "c": ["c"]}


def test_propagate_updates_fans_to_all_copies():
    from kg.invalidate_llm import propagate_updates
    updates = {"a": {"invalid_at": "2014-05-09T00:00:00Z", "invalidation_method": "llm"}}
    members = {"a": ["a", "b", "c"]}
    out = propagate_updates(updates, members)
    assert set(out) == {"a", "b", "c"}                          # rep verdict fanned to b, c
    assert out["b"] == out["c"] == updates["a"]                 # value-equal
    out["b"]["invalid_at"] = "MUT"                              # ...but each copy is an independent dict
    assert out["c"]["invalid_at"] != "MUT" and updates["a"]["invalid_at"] != "MUT"


def test_asset_gate_keeps_linked_drops_unlinked(tmp_path):
    """Option B: with --asset-groups, only events with >=1 asset-linked triplet entity survive."""
    from kg.invalidate_llm import load_eligible_events, load_asset_linked_keys
    rows = [
        {"events": [
            {"id": "a", "statement": "Crude oil rose", "statement_type": "FACT",
             "temporal_type": "DYNAMIC", "valid_at": "2014-05-01T00:00:00Z", "invalid_at": None,
             "triplets": [{"subject": "Crude Oil", "object": "Market", "relation": "RAISES"}]},
            {"id": "b", "statement": "Hugo Boss opened a store", "statement_type": "FACT",
             "temporal_type": "DYNAMIC", "valid_at": "2014-05-02T00:00:00Z", "invalid_at": None,
             "triplets": [{"subject": "Hugo Boss AG", "object": "China", "relation": "OPERATES_IN"}]},
        ]},
    ]
    dis = tmp_path / "d.jsonl"
    dis.write_text("\n".join(json.dumps(r) for r in rows))
    eg = tmp_path / "entity_groups.json"
    eg.write_text(json.dumps({"crude oil": {"type": "COMMODITY",
                                            "groups": [{"key": "crude_oil", "method": "exact"}]}}))
    keys = load_asset_linked_keys(eg)
    assert keys == {"crude oil"}                                  # casefolded
    gated, _ = load_eligible_events(dis, keys)
    assert [e["id"] for e in gated] == ["a"]                      # Hugo Boss dropped (no asset entity)
    ungated, _ = load_eligible_events(dis)                        # no gate = both
    assert {e["id"] for e in ungated} == {"a", "b"}


def test_build_pairs_static_prediction_primary_pairs():
    """A STATIC PREDICTION primary (a point-in-time forecast) must still pair with a
    LATER FACT/PREDICTION secondary. Regression: the cookbook's STATIC temporal filter
    (start <= valid_at <= end) excludes any candidate before the secondary's window, so a
    strictly-earlier STATIC PREDICTION got zero candidates — silently dropping ~12% of
    eligible events even though is_primary_eligible admits every PREDICTION."""
    sp = _ev2("sp", "2014-05-01T00:00:00Z", stype="PREDICTION", temp="STATIC")
    s1 = _ev2("s1", "2014-05-20T00:00:00Z", stype="FACT")          # resolves the forecast
    ref = np.array([1.0, 0.0], dtype=np.float32)
    emb = {"sp": ref.copy(), "s1": ref.copy()}
    pairs = build_pairs([sp, s1], emb)
    assert ("sp", "s1") in pairs


def test_apply_basic_close():
    a = _e("a", "DYNAMIC", "2014-05-02T00:00:00Z"); b = _e("b", "STATIC", "2014-05-05T00:00:00Z")
    by_id = {"a": a, "b": b}
    apply_verdicts(by_id, [("a", "b")], [True])
    assert a["invalid_at"] == "2014-05-05T00:00:00Z" and a["invalidated_by"] == "b"
    assert a["invalidation_method"] == "llm" and a["expired_at"] == a["created_at"]


def test_apply_refine_earlier_only():
    a = _e("a", "DYNAMIC", "2014-05-02T00:00:00Z", invalid="2014-05-10T00:00:00Z")
    b = _e("b", "STATIC", "2014-05-05T00:00:00Z")
    apply_verdicts({"a": a, "b": b}, [("a", "b")], [True])
    assert a["invalid_at"] == "2014-05-05T00:00:00Z"        # refined earlier
    a2 = _e("a2", "DYNAMIC", "2014-05-02T00:00:00Z", invalid="2014-05-05T00:00:00Z")
    z = _e("z", "STATIC", "2014-05-20T00:00:00Z")
    apply_verdicts({"a2": a2, "z": z}, [("a2", "z")], [True])
    assert a2["invalid_at"] == "2014-05-05T00:00:00Z"       # later invalidator does NOT push it back


def test_apply_earliest_wins():
    a = _e("a", "DYNAMIC", "2014-05-02T00:00:00Z")
    b = _e("b", "STATIC", "2014-05-09T00:00:00Z"); c = _e("c", "STATIC", "2014-05-05T00:00:00Z")
    apply_verdicts({"a": a, "b": b, "c": c}, [("a", "b"), ("a", "c")], [True, True])
    assert a["invalidated_by"] == "c" and a["invalid_at"] == "2014-05-05T00:00:00Z"


def test_apply_false_verdict_noop():
    a = _e("a", "DYNAMIC", "2014-05-02T00:00:00Z"); b = _e("b", "STATIC", "2014-05-05T00:00:00Z")
    apply_verdicts({"a": a, "b": b}, [("a", "b")], [False])
    assert a["invalid_at"] is None and a["invalidation_method"] is None


def test_parse_verdict_fallback():
    from kg.invalidate_llm import InvalidationAgent
    ag = InvalidationAgent.__new__(InvalidationAgent)   # no vLLM init
    assert ag._parse('{"invalidated": true}').invalidated is True
    assert ag._parse('garbage').invalidated is False     # safe default


def test_write_output_non_lossy_excludes_embedding(tmp_path):
    from kg.invalidate_llm import write_output
    rows = [{"article_id": "a", "events": [
        {"id": "e1", "statement": "s", "temporal_type": "DYNAMIC",
         "valid_at": "2014-05-02T00:00:00Z", "invalid_at": None,
         "invalidated_by": None, "invalidation_method": None,
         "triplets": [{"relation": "RAISES"}], "embedding": [0.1, 0.2]}]}]
    updates = {"e1": {"invalid_at": "2014-05-05T00:00:00Z", "invalidated_by": "e2",
                      "expired_at": "2014-05-02T00:00:00Z", "invalidation_method": "llm"}}
    out = tmp_path / "o.jsonl"
    write_output(rows, updates, out)
    written = [json.loads(l) for l in out.read_text().splitlines()]
    ev = written[0]["events"][0]
    assert ev["invalid_at"] == "2014-05-05T00:00:00Z" and ev["invalidation_method"] == "llm"
    assert ev["triplets"] == [{"relation": "RAISES"}]      # unchanged
    assert "embedding" not in ev                            # excluded from JSONL
    assert len(written) == len(rows)                        # row count preserved


# ---------------------------------------------------------------------------
# Commit-1 primitive tests
# ---------------------------------------------------------------------------

def test_temporal_bounds_atemporal_returns_none():
    from kg.invalidate_llm import temporal_bounds
    ev = {"temporal_type": "ATEMPORAL", "valid_at": "2014-05-01T00:00:00Z", "invalid_at": None}
    assert temporal_bounds(ev) is None


def test_temporal_bounds_no_valid_at_returns_none():
    from kg.invalidate_llm import temporal_bounds
    ev = {"temporal_type": "DYNAMIC", "valid_at": None, "invalid_at": None}
    assert temporal_bounds(ev) is None


def test_temporal_bounds_dynamic_with_invalid_at():
    from kg.invalidate_llm import temporal_bounds, _dt
    ev = {"temporal_type": "DYNAMIC", "valid_at": "2014-05-01T00:00:00Z",
          "invalid_at": "2014-05-10T00:00:00Z"}
    start, end = temporal_bounds(ev)
    assert start == _dt("2014-05-01T00:00:00Z")
    assert end == _dt("2014-05-10T00:00:00Z")


def test_temporal_bounds_static_returns_equal():
    from kg.invalidate_llm import temporal_bounds
    ev = {"temporal_type": "STATIC", "valid_at": "2014-05-05T00:00:00Z", "invalid_at": None}
    start, end = temporal_bounds(ev)
    assert start == end


def test_temporal_bounds_dynamic_no_invalid_at():
    from kg.invalidate_llm import temporal_bounds
    ev = {"temporal_type": "DYNAMIC", "valid_at": "2014-05-05T00:00:00Z", "invalid_at": None}
    start, end = temporal_bounds(ev)
    assert start == end


def test_overlaps_dynamic():
    from kg.invalidate_llm import _overlaps_dynamic, _dt
    start = _dt("2014-05-05T00:00:00Z")
    end = _dt("2014-05-10T00:00:00Z")

    # Condition 1: ee not None and es <= start <= ee
    c1 = {"temporal_type": "DYNAMIC", "valid_at": "2014-05-01T00:00:00Z",
          "invalid_at": "2014-05-07T00:00:00Z"}
    assert _overlaps_dynamic(c1, start, end) is True

    # Condition 2: ee is None and es <= start (ongoing, started before start)
    c2 = {"temporal_type": "DYNAMIC", "valid_at": "2014-05-01T00:00:00Z", "invalid_at": None}
    assert _overlaps_dynamic(c2, start, end) is True

    # Condition 3: start <= es <= end (starts within interval)
    c3 = {"temporal_type": "DYNAMIC", "valid_at": "2014-05-07T00:00:00Z", "invalid_at": None}
    assert _overlaps_dynamic(c3, start, end) is True

    # False: starts after end
    c4 = {"temporal_type": "DYNAMIC", "valid_at": "2014-05-15T00:00:00Z", "invalid_at": None}
    assert _overlaps_dynamic(c4, start, end) is False

    # False: not DYNAMIC (STATIC candidate)
    c5 = {"temporal_type": "STATIC", "valid_at": "2014-05-07T00:00:00Z", "invalid_at": None}
    assert _overlaps_dynamic(c5, start, end) is False


def test_type_pair_allowed():
    from kg.invalidate_llm import type_pair_allowed
    fact = {"statement_type": "FACT"}
    pred = {"statement_type": "PREDICTION"}
    opin = {"statement_type": "OPINION"}

    assert type_pair_allowed(fact, fact) is True          # FACT→FACT
    assert type_pair_allowed(pred, fact) is True          # PREDICTION→FACT (resolved)
    assert type_pair_allowed(fact, pred) is False         # FACT→PREDICTION not allowed
    assert type_pair_allowed(pred, pred) is True          # PREDICTION→PREDICTION (revised)
    assert type_pair_allowed(opin, fact) is False         # OPINION never participates
    assert type_pair_allowed(opin, pred) is False


def test_gather_entity_shared():
    from kg.invalidate_llm import gather_entity_shared, build_entity_index
    gold1 = {"id": "e1", "triplets": [{"subject": "Gold", "object": "Investors"}]}
    gold2 = {"id": "e2", "triplets": [{"subject": "Gold", "object": "Traders"}]}
    unrel = {"id": "e3", "triplets": [{"subject": "Oil", "object": "OPEC"}]}
    idx = build_entity_index([gold1, gold2, unrel])
    result = gather_entity_shared(gold1, idx)
    ids = {e["id"] for e in result}
    assert "e2" in ids          # shares "Gold"
    assert "e3" not in ids      # no shared entity
    assert "e1" not in ids      # primary excluded from its own candidates


def test_filter_by_embedding_similarity():
    from kg.invalidate_llm import filter_by_embedding_similarity
    # Unit vectors along x-axis for primary; candidates at varying angles
    primary = {"id": "p"}
    c_high = {"id": "c1"}   # cos ≈ 1.0 (same direction)
    c_low  = {"id": "c2"}   # cos = 0.0 (orthogonal, below floor 0.5)
    c_mid  = {"id": "c3"}   # cos ≈ 0.707 (45°, above floor)
    emb = {
        "p":  np.array([1.0, 0.0], dtype=np.float32),
        "c1": np.array([1.0, 0.0], dtype=np.float32),
        "c2": np.array([0.0, 1.0], dtype=np.float32),
        "c3": np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2),
    }
    result = filter_by_embedding_similarity(primary, [c_high, c_low, c_mid], emb)
    ids = [e["id"] for e in result]
    assert "c1" in ids and "c3" in ids    # both >= 0.5
    assert "c2" not in ids                # orthogonal, below floor
    assert ids.index("c1") < ids.index("c3")   # sorted descending


def test_filter_by_embedding_similarity_top_k():
    from kg.invalidate_llm import filter_by_embedding_similarity
    primary = {"id": "p"}
    ref = np.array([1.0, 0.0], dtype=np.float32)
    # 12 identical candidates, all cosine=1.0
    cands = [{"id": f"c{i}"} for i in range(12)]
    emb = {"p": ref.copy(), **{f"c{i}": ref.copy() for i in range(12)}}
    result = filter_by_embedding_similarity(primary, cands, emb, top_k=10)
    assert len(result) == 10


def test_is_primary_eligible():
    from kg.invalidate_llm import is_primary_eligible
    dyn_fact  = {"statement_type": "FACT",       "temporal_type": "DYNAMIC",   "valid_at": "2014-05-01T00:00:00Z"}
    sta_fact   = {"statement_type": "FACT",       "temporal_type": "STATIC",    "valid_at": "2014-05-01T00:00:00Z"}
    prediction = {"statement_type": "PREDICTION", "temporal_type": "DYNAMIC",   "valid_at": "2014-05-01T00:00:00Z"}
    opinion    = {"statement_type": "OPINION",    "temporal_type": "DYNAMIC",   "valid_at": "2014-05-01T00:00:00Z"}
    atemporal  = {"statement_type": "FACT",       "temporal_type": "ATEMPORAL", "valid_at": "2014-05-01T00:00:00Z"}
    assert is_primary_eligible(dyn_fact) is True
    assert is_primary_eligible(sta_fact) is False
    assert is_primary_eligible(prediction) is True
    assert is_primary_eligible(opinion) is False
    assert is_primary_eligible(atemporal) is False


def test_load_eligible_events_excludes_opinion(tmp_path):
    p = tmp_path / "d.jsonl"
    fact_ev = {**_event("e1"), "statement_type": "FACT"}
    pred_ev = {**_event("e2"), "statement_type": "PREDICTION"}
    opin_ev = {**_event("e3"), "statement_type": "OPINION"}
    rows = [_row([fact_ev, pred_ev, opin_ev])]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    eligible, _ = load_eligible_events(p)
    ids = {e["id"] for e in eligible}
    assert "e1" in ids and "e2" in ids
    assert "e3" not in ids   # OPINION excluded
