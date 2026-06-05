"""Aggregate a KG grader sidecar into a per-axis weakness map.

  module load Python/3.12.3-GCCcore-13.3.0
  .venv/bin/python scripts/kg_weakness_map.py \\
      --grader results/kg/dev/march_2022_dev.v2.4.grader.jsonl \\
      --out results/kg/dev/march_2022_dev.v2.4.weakness_map.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kg.schemas import RELATION_TYPES_TUPLE  # noqa: E402

AXES = ["supported", "relation_ok", "subject_type_ok", "object_type_ok",
        "macro_relevant", "non_trivial"]
RELATION_CODES = set(RELATION_TYPES_TUPLE)
MIN_SLICE = 20


def axis_fail_by_slice(rows: list[dict], axis: str, slice_key: str) -> dict:
    """Per-slice fail rate for one axis; slices with n<MIN_SLICE are guarded."""
    by = defaultdict(lambda: {"n": 0, "fail": 0})
    for r in rows:
        if not r.get("grader_verdict_present"):
            continue
        key = r.get(slice_key, "?")
        by[key]["n"] += 1
        if r.get(axis) is False:
            by[key]["fail"] += 1
    out = {}
    for key, c in sorted(by.items()):
        if c["n"] < MIN_SLICE:
            out[key] = {"n": c["n"], "insufficient_sample": True}
        else:
            out[key] = {"n": c["n"], "fail": c["fail"],
                        "fail_pct": round(100 * c["fail"] / c["n"], 1)}
    return out


def type_consistency(rows: list[dict]) -> list[dict]:
    """Cheap non-LLM check (per spec): group facts by normalized entity name,
    flag any entity assigned more than one type across facts."""
    types_by_name = defaultdict(set)
    for r in rows:
        if not r.get("grader_verdict_present"):
            continue
        types_by_name[r.get("subject", "?").strip().lower()].add(r["subject_type"])
        types_by_name[r.get("object", "?").strip().lower()].add(r["object_type"])
    return [{"entity": name, "types": sorted(ts)}
            for name, ts in sorted(types_by_name.items()) if len(ts) > 1]


def fix_candidates(rows: list[dict]) -> list[tuple]:
    """Tally the judge's structured *_fix suggestions across graded facts.

    Returns (field, suggested_value, count) sorted by count desc. A value that
    is one of the 18 codes = a mis-type; a value that is NOT = a missing-type
    (Phase-2 escape-hatch) candidate. Reads the structured fields, not the
    free-text critique."""
    c = Counter()
    for r in rows:
        if not r.get("grader_verdict_present"):
            continue
        for fld in ("subject_type_fix", "object_type_fix", "relation_fix"):
            val = (r.get(fld) or "").strip()
            if val:
                c[(fld, val)] += 1
    return [(fld, val, n) for (fld, val), n in c.most_common()]


def ideal_relation_analysis(rows: list[dict], relation_codes: set) -> dict:
    """Decompose the relation axis using the unconstrained `ideal_relation`.

    `ideal_relation` is the judge's best relation in plain English (a code when
    one fits, else a phrase). A phrase = the 18 codes don't cover this link
    (schema gap). Splits the relation_ok failures into extractor-mispicks (ideal
    IS a code → a better code existed) vs schema-gaps (ideal is a phrase → no
    code fits), and tallies the out-of-schema phrases (missing-relation
    candidates)."""
    graded = [r for r in rows if r.get("grader_verdict_present")]

    def classify(r: dict) -> str:
        v = (r.get("ideal_relation") or "").strip()
        if not v:
            return "empty"  # judge left it blank — count separately, NOT as a gap
        return "code" if v.upper().replace(" ", "_") in relation_codes else "phrase"

    cls = [classify(r) for r in graded]
    fails = [r for r in graded if r.get("relation_ok") is False]
    fcls = [classify(r) for r in fails]
    phrases: Counter = Counter()
    for r in graded:
        if classify(r) == "phrase":
            phrases[(r.get("ideal_relation") or "").strip().lower()] += 1
    return {
        "n": len(graded),
        "ideal_in_schema": cls.count("code"),
        "ideal_out_of_schema": cls.count("phrase"),
        "ideal_empty": cls.count("empty"),
        "relation_ok_fails": len(fails),
        "fails_extractor_mispick": fcls.count("code"),
        "fails_schema_gap": fcls.count("phrase"),
        "fails_unknown": fcls.count("empty"),
        "phrase_candidates": phrases.most_common(),
    }


def _table(title: str, d: dict) -> str:
    lines = [f"### {title}", "", "| slice | n | fail% |", "|---|---|---|"]
    for k, v in d.items():
        cell = "— (insufficient sample)" if v.get("insufficient_sample") else f"{v['fail_pct']}% ({v['fail']}/{v['n']})"
        lines.append(f"| {k} | {v['n']} | {cell} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grader", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    rows = [json.loads(l) for l in args.grader.read_text().splitlines() if l.strip()]
    graded = [r for r in rows if r.get("grader_verdict_present")]

    md = ["# KG extractor weakness map", "",
          f"Graded facts: {len(graded)} (skipped: {len(rows) - len(graded)})", ""]
    overall = {ax: sum(1 for r in graded if r.get(ax) is False) for ax in AXES}
    md.append("## Overall axis fail counts\n")
    md += [f"- `{ax}`: {overall[ax]} ({round(100*overall[ax]/max(len(graded),1),1)}%)"
           for ax in AXES]
    md.append("")

    for ax in ("relation_ok", "subject_type_ok"):
        md.append(_table(f"{ax} fail rate x relation", axis_fail_by_slice(graded, ax, "relation")))
    md.append(_table("subject_type_ok fail rate x subject_type",
                     axis_fail_by_slice(graded, "subject_type_ok", "subject_type")))

    # Missing-type / mistype candidate harvest: tally the judge's structured
    # *_fix suggestions (no NLP on the critique needed).
    md.append("## Missing-type / mistype candidates (judge's *_fix suggestions)\n")
    for fld, val, n in fix_candidates(graded):
        md.append(f"- {fld} -> {val}: {n}")

    md.append("\n## Schema adequacy — ideal_relation (best relation in all English)\n")
    ira = ideal_relation_analysis(rows, RELATION_CODES)
    md.append(f"- ideal_relation IS a schema code (the 18 cover it): {ira['ideal_in_schema']}/{ira['n']} "
              f"({round(100*ira['ideal_in_schema']/max(ira['n'],1),1)}%)")
    md.append(f"- ideal_relation is an OUT-OF-SCHEMA phrase (gap): {ira['ideal_out_of_schema']}/{ira['n']} "
              f"({round(100*ira['ideal_out_of_schema']/max(ira['n'],1),1)}%)")
    if ira["ideal_empty"]:
        md.append(f"- ideal_relation left blank (excluded from the split): {ira['ideal_empty']}/{ira['n']}")
    md.append(f"- of the {ira['relation_ok_fails']} relation_ok failures: "
              f"**{ira['fails_extractor_mispick']} extractor-mispick** (ideal is a code) vs "
              f"**{ira['fails_schema_gap']} schema-gap** (ideal is a phrase)"
              + (f" vs {ira['fails_unknown']} blank" if ira["fails_unknown"] else ""))
    md.append("\n### Missing-relation candidates (ideal_relation phrases not in the 18)\n")
    for v, c in ira["phrase_candidates"]:
        md.append(f"- {v}: {c}")

    md.append("\n## Cross-entity type inconsistencies (same name, >1 type)\n")
    for c in type_consistency(graded):
        md.append(f"- {c['entity']}: {c['types']}")
    args.out.write_text("\n".join(md) + "\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
