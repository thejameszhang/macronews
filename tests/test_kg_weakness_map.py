import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from kg_weakness_map import axis_fail_by_slice  # noqa: E402


def _row(relation, sub_t, relation_ok=True, supported=True):
    return {"grader_verdict_present": True, "relation": relation,
            "subject_type": sub_t, "object_type": "CONCEPT",
            "relation_ok": relation_ok, "supported": supported,
            "non_trivial": True, "macro_relevant": True,
            "subject_type_ok": True, "object_type_ok": True}


def test_fail_rate_and_small_slice_guard():
    rows = [_row("IMPACT", "EVENT", relation_ok=(i % 5 != 0)) for i in range(25)]
    rows += [_row("OWNS", "COMPANY", relation_ok=False) for _ in range(3)]
    out = axis_fail_by_slice(rows, axis="relation_ok", slice_key="relation")
    assert out["IMPACT"]["n"] == 25
    assert out["IMPACT"]["fail_pct"] == 20.0          # 5 of 25
    assert out["OWNS"]["insufficient_sample"] is True  # n=3 < 20
    assert "fail_pct" not in out["OWNS"]


def test_slice_boundary_n_equals_20_is_not_guarded():
    # Guard is strict `< MIN_SLICE`, so exactly 20 is reported as a percentage.
    rows = [_row("IMPACT", "EVENT", relation_ok=(i != 0)) for i in range(20)]
    out = axis_fail_by_slice(rows, axis="relation_ok", slice_key="relation")
    assert out["IMPACT"]["n"] == 20
    assert out["IMPACT"]["fail_pct"] == 5.0  # 1 of 20
    assert "insufficient_sample" not in out["IMPACT"]


def test_type_consistency_flags_multi_typed_entity():
    from kg_weakness_map import type_consistency
    rows = [
        {"grader_verdict_present": True, "subject": "Apple", "subject_type": "COMPANY",
         "object": "iPhone", "object_type": "CONCEPT"},
        {"grader_verdict_present": True, "subject": "Apple", "subject_type": "SECTOR",
         "object": "x", "object_type": "CONCEPT"},
    ]
    assert type_consistency(rows) == [{"entity": "apple", "types": ["COMPANY", "SECTOR"]}]


def test_fix_candidates_tally():
    from kg_weakness_map import fix_candidates
    rows = [
        {"grader_verdict_present": True, "subject_type_fix": "COMMODITY",
         "object_type_fix": "", "relation_fix": ""},
        {"grader_verdict_present": True, "subject_type_fix": "COMMODITY",
         "object_type_fix": "TRADE_AGREEMENT", "relation_fix": ""},
    ]
    out = fix_candidates(rows)
    assert ("subject_type_fix", "COMMODITY", 2) in out
    assert ("object_type_fix", "TRADE_AGREEMENT", 1) in out


def test_ideal_relation_analysis():
    from kg_weakness_map import ideal_relation_analysis
    codes = {"NEGATIVE_IMPACT_ON", "IMPACT"}
    rows = [
        {"grader_verdict_present": True, "relation_ok": False, "ideal_relation": "NEGATIVE_IMPACT_ON"},  # mispick
        {"grader_verdict_present": True, "relation_ok": False, "ideal_relation": "directs"},             # schema gap
        {"grader_verdict_present": True, "relation_ok": True, "ideal_relation": "urges"},                # gap despite ok
        {"grader_verdict_present": True, "relation_ok": True, "ideal_relation": "IMPACT"},               # fine
    ]
    out = ideal_relation_analysis(rows, codes)
    assert out["n"] == 4
    assert out["ideal_in_schema"] == 2 and out["ideal_out_of_schema"] == 2
    assert out["relation_ok_fails"] == 2
    assert out["fails_extractor_mispick"] == 1 and out["fails_schema_gap"] == 1
    assert dict(out["phrase_candidates"]) == {"directs": 1, "urges": 1}


def test_ideal_relation_analysis_blank_not_counted_as_gap():
    # A blank ideal_relation must NOT inflate the schema-gap counts.
    from kg_weakness_map import ideal_relation_analysis
    rows = [
        {"grader_verdict_present": True, "relation_ok": False, "ideal_relation": ""},
        {"grader_verdict_present": True, "relation_ok": False, "ideal_relation": "directs"},
    ]
    out = ideal_relation_analysis(rows, {"IMPACT"})
    assert out["ideal_empty"] == 1
    assert out["ideal_out_of_schema"] == 1   # only "directs", not the blank
    assert out["fails_schema_gap"] == 1      # only "directs"
    assert out["fails_unknown"] == 1         # the blank failure
    assert dict(out["phrase_candidates"]) == {"directs": 1}
