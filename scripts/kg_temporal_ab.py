"""A/B harness: compare 3-pass temporal extractor vs single-pass baseline on directional relations.

Usage:
  module load Python/3.12.3-GCCcore-13.3.0
  .venv/bin/python scripts/kg_temporal_ab.py \\
      --three-pass results/kg/dev/2014-05c.temporal.jsonl \\
      --single-pass results/kg/dev/2014-05c.modality.jsonl \\
      --out results/kg/dev/2014-05c.ab_comparison.csv

Output: one row per article_id (outer join), with per-relation counts from each
pipeline and epistemic-status breakdowns (statement_type for 3-pass,
modality for single-pass). Rows sorted by article_id.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

DIRECTIONAL = {"RAISES", "DECREASES", "LEAVES_UNCHANGED", "CAUSES_RISE_IN", "CAUSES_FALL_IN", "IMPACT"}

REL_COLS = [r.lower() for r in ("RAISES", "DECREASES", "LEAVES_UNCHANGED", "CAUSES_RISE_IN", "CAUSES_FALL_IN", "IMPACT")]

FIELDNAMES = (
    ["article_id", "n_directional_singlepass", "n_directional_3pass"]
    + [f"{r}_{side}" for r in REL_COLS for side in ("singlepass", "3pass")]
    + ["fact_3pass", "prediction_3pass", "opinion_3pass", "realized_singlepass", "expected_singlepass"]
)


def _empty_3pass() -> dict:
    d: dict = {"n_directional_3pass": 0, "fact_3pass": 0, "prediction_3pass": 0, "opinion_3pass": 0}
    for r in REL_COLS:
        d[f"{r}_3pass"] = 0
    return d


def _empty_singlepass() -> dict:
    d: dict = {"n_directional_singlepass": 0, "realized_singlepass": 0, "expected_singlepass": 0}
    for r in REL_COLS:
        d[f"{r}_singlepass"] = 0
    return d


def read_three_pass(path: Path) -> dict[str, dict]:
    """article_id → aggregated counts from the 3-pass temporal sidecar."""
    records: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            aid = row["article_id"]
            rec = _empty_3pass()
            for event in row.get("events", []):
                st = (event.get("statement_type") or "").upper()
                if st == "FACT":
                    rec["fact_3pass"] += 1
                elif st == "PREDICTION":
                    rec["prediction_3pass"] += 1
                elif st == "OPINION":
                    rec["opinion_3pass"] += 1
                for triplet in event.get("triplets", []):
                    rel = (triplet.get("relation") or "").upper()
                    if rel in DIRECTIONAL:
                        rec["n_directional_3pass"] += 1
                        rec[f"{rel.lower()}_3pass"] += 1
            records[aid] = rec
    return records


def read_single_pass(path: Path) -> dict[str, dict]:
    """article_id → aggregated counts from the single-pass baseline sidecar."""
    records: dict[str, dict] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            aid = row["article_id"]
            rec = _empty_singlepass()
            for fact in row.get("facts", []):
                rel = (fact.get("relation") or "").upper()
                if rel in DIRECTIONAL:
                    rec["n_directional_singlepass"] += 1
                    rec[f"{rel.lower()}_singlepass"] += 1
                mod = (fact.get("modality") or "").lower()
                if mod == "realized":
                    rec["realized_singlepass"] += 1
                elif mod == "expected":
                    rec["expected_singlepass"] += 1
            records[aid] = rec
    return records


def main() -> None:
    p = argparse.ArgumentParser(description="A/B comparison: 3-pass vs single-pass KG extractor")
    p.add_argument("--three-pass", required=True, type=Path, dest="three_pass")
    p.add_argument("--single-pass", required=True, type=Path, dest="single_pass")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    three = read_three_pass(args.three_pass)
    single = read_single_pass(args.single_pass)

    all_ids = sorted(three.keys() | single.keys())

    rows = []
    for aid in all_ids:
        rec: dict = {"article_id": aid}
        rec.update(single.get(aid, _empty_singlepass()))
        rec.update(three.get(aid, _empty_3pass()))
        rows.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    total_sp = sum(r["n_directional_singlepass"] for r in rows)
    total_tp = sum(r["n_directional_3pass"] for r in rows)
    print(f"Articles: {len(rows)} | directional singlepass: {total_sp} | directional 3pass: {total_tp}")


if __name__ == "__main__":
    main()
