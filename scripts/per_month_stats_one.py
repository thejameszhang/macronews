#!/usr/bin/env python3
"""Compute tabular_body stats for ONE month. Writes one pretty-printed JSON file.

Designed to run as a SLURM array task (one per month). The aggregator
(per_month_stats_aggregate.py) reads all per-month JSON files into a
single markdown report.

Output JSON schema (pretty-printed, multi-line):
    {
      "month": "2002-09",
      "counts": {
          "total":         int,  # total articles in cleaned shards
          "existing":      int,  # caught by existing tabular_subject/headline
          "tab_body":      int,  # flagged by new tabular_body filter
          "overlap":       int,  # caught by both
          "additional":    int,  # caught only by new filter
          "missed_by_us":  int,  # caught only by existing (not new) — tells us
                                 # what the structural rule still misses
          "no_sidecar":    int   # in cleaned shards, missing from sidecar
      },
      "additional": [             # caught only by tabular_body — pattern-mining
        {"accession_number": str, "headline": str},
        ...
      ],
      "missed":     [             # caught only by existing tabular filters
        {"accession_number": str, "headline": str},
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path("/nfs/roberts/project/pi_btk22/jyz32/macronews")
SOURCE_DIR = Path("/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles")
TABULAR_DIR = REPO / "results" / "tabular"

# Mirror src/loaders.py:434-477 — keep in sync.
_TABULAR_SUBJECTS = {"N/TAB", "N/DTA"}
_TABLE_HEADLINE_RE = re.compile(r"^\s*(Table|TABLE)\s*:")
_CALENDAR_HEADLINE_RE = re.compile(
    r"(?:"
    r"-\s+(?:Week|Month)\s+Ahead\b"
    r"|(?:Political,?\s+Economic|Corporate\s+And\s+Economic|Economic\s+Indicators|Financial)\s+Calendar\b"
    r"|Calendar\s+[Oo]f\s+(?:Corporate|Corporate\s+Earnings)\s+(?:Events|Conference\s+Calls)\b"
    r"|Calendar\s+[Oo]f\s+[A-Za-z .]*?Earnings\s+(?:Expected|Conference\s+Calls)"
    r"|Calendar\s+[Oo]f\s+(?:Debt|Equity\s+Issues|Wealth\s+Management)"
    r"|International\s+Debt\s+Calendar|U\.?S\.?\s+Treasury\s+Calendar|(?:DJ\s+)?Muni\s+Pricing\s+Calendar"
    r"|[A-Za-z, ]+\s+Calendar\s*[-–]\s*(?:\d{4},?\s*)?(?:\d{4}\s+)?Futures,?\s+Options\s+Dates"
    r"|Holiday\s+Advisory\b|Markets,\s+Banks,\s+Government\s+Offices\s+Closed"
    r")",
    re.IGNORECASE,
)


def existing_tabular_match(article: dict) -> bool:
    codes = (article.get("codes") or {}).get("subject") or []
    if set(codes) & _TABULAR_SUBJECTS:
        return True
    headline = article.get("headline") or ""
    if _TABLE_HEADLINE_RE.match(headline) or _CALENDAR_HEADLINE_RE.search(headline):
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", required=True, help='YYYY-MM, e.g. "2002-09"')
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path (one row).")
    ap.add_argument("--sidecar-dir", type=Path, default=TABULAR_DIR,
                    help="Directory containing {YYYY-MM}.jsonl sidecars "
                         "(default: results/tabular)")
    args = ap.parse_args()

    sidecar_path = args.sidecar_dir / f"{args.month}.jsonl"
    if not sidecar_path.exists():
        raise FileNotFoundError(f"sidecar missing: {sidecar_path}")

    flags = {}
    with open(sidecar_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            flags[r["accession_number"]] = r["is_tabular_body"]

    # All cleaned shards for this month: 2002-09a, 2002-09b, ...
    shards = sorted(SOURCE_DIR.glob(f"{args.month}*_clean.jsonl"))
    if not shards:
        raise FileNotFoundError(f"no cleaned shards for {args.month}")

    total = n_existing = n_tab_body = n_overlap = n_additional = n_missed = n_no_sidecar = 0
    additional: list[dict] = []
    missed: list[dict] = []
    for sp in shards:
        with open(sp) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                aid = r.get("accession_number")
                headline = r.get("headline", "") or ""
                is_existing = existing_tabular_match(r)
                if aid in flags:
                    is_body = flags[aid]
                else:
                    is_body = False
                    n_no_sidecar += 1
                if is_existing:
                    n_existing += 1
                if is_body:
                    n_tab_body += 1
                if is_existing and is_body:
                    n_overlap += 1
                if is_body and not is_existing:
                    n_additional += 1
                    additional.append({"accession_number": aid, "headline": headline})
                if is_existing and not is_body:
                    n_missed += 1
                    missed.append({"accession_number": aid, "headline": headline})

    additional.sort(key=lambda d: d["accession_number"])
    missed.sort(key=lambda d: d["accession_number"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "month": args.month,
            "counts": {
                "total": total,
                "existing": n_existing,
                "tab_body": n_tab_body,
                "overlap": n_overlap,
                "additional": n_additional,
                "missed_by_us": n_missed,
                "no_sidecar": n_no_sidecar,
            },
            "additional": additional,
            "missed": missed,
        }, f, indent=2)
        f.write("\n")
    print(f"[stats] {args.month}: total={total:,}, tab_body={n_tab_body:,} "
          f"({n_tab_body/total*100:.1f}%), additional={n_additional:,} "
          f"({n_additional/total*100:.2f}%), missed_by_us={n_missed:,} "
          f"({n_missed/total*100:.2f}%)")


if __name__ == "__main__":
    main()
