#!/usr/bin/env python3
"""Compute the filter waterfall counts for ONE month.

Designed to run as a SLURM array task (one per month). The aggregator
(per_month_waterfall_aggregate.py) sums all per-month JSONs into the
final results/filter_waterfall.csv + a markdown report.

Output JSON schema (one row):
    {
      "month": "2014-10",
      "total": int,                  # articles in cleaned shards
      "removed": {
          "date_floor": int,         # excluded by < 1996-01 (always 0 for
                                     #   post-1996 months; left for symmetry)
          "lifestyle": int,          # 16-code lifestyle/sports denylist
          "tabular_subject": int,    # N/TAB or N/DTA
          "tabular_headline": int,   # TABLE: prefix or calendar regex
          "tabular_body": int,       # NML structural detector (sidecar)
          "sat": int,                # N/SAT (Seeking Alpha)
          "exchange_pr": int,        # exchange-PR products
          "unembeddable": int,       # empty body / wire-marker / corrections
          "insider": int,            # N/ISD, N/144, N/ISS, N/ISB
          "npl": int,                # N/NPL
          "blk": int                 # N/BLK
      },
      "passes_all": int              # articles that survive every filter
    }

Filters are applied in waterfall order — the first filter that matches
"claims" the article. tabular_body is wired in immediately after
tabular_headline (so its count is its MARGINAL contribution after the
upstream tabular filters).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path("/nfs/roberts/project/pi_btk22/jyz32/macronews")
SOURCE_DIR = Path("/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles")
TABULAR_DIR = REPO / "results" / "tabular"
DATE_FLOOR = "1996-01"

# Mirror src/loaders.py + scripts/misc/filter_waterfall.py constants.
LIFESTYLE_DENYLIST = {
    "N/SPT", "N/ART", "N/SPO", "N/BSE", "N/LIF", "N/RVW",
    "N/FTB", "N/BKT", "N/FCG", "N/HKY", "N/SOC", "N/FSH", "N/GLF",
    "N/OLY", "N/TNS", "N/HOL",
}
TABLE_SUBJECTS = {"N/TAB", "N/DTA"}
SAT_SUBJECTS = {"N/SAT"}
INSIDER_SUBJECTS = {"N/ISD", "N/144", "N/ISS", "N/ISB"}
NPL_SUBJECTS = {"N/NPL"}
BLK_SUBJECTS = {"N/BLK"}
EXCHANGE_PR_PRODUCTS = {
    "P/ASX", "P/BSEX", "P/HKEX", "P/MYX", "P/SXH", "P/TSEX", "P/XTAI",
    "P/JSE", "P/KSXC", "P/AKT",
}
TABLE_HEADLINE_RE = re.compile(r"^\s*(Table|TABLE)\s*:")
CALENDAR_HEADLINE_RE = re.compile(
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
WIRE_MARKER_ONLY_RE = re.compile(r"^(-0-?|\(END\)|(\(MORE\)\s*)+)\s*$")


def unembeddable(text: str) -> bool:
    body = (text or "").strip()
    if not body:
        return True
    if WIRE_MARKER_ONLY_RE.match(body):
        return True
    if body.startswith("Corrections & Amplifications") and len(body) < 300:
        return True
    return False


def load_tabular_body_set(month: str) -> set[str]:
    side = TABULAR_DIR / f"{month}.jsonl"
    if not side.exists():
        return set()
    out = set()
    with open(side) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("is_tabular_body"):
                out.add(r["accession_number"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--month", required=True, help='YYYY-MM, e.g. "2014-10"')
    ap.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    args = ap.parse_args()

    tab_body = load_tabular_body_set(args.month)
    is_pre_floor = args.month < DATE_FLOOR

    counts = {
        "date_floor": 0, "lifestyle": 0,
        "tabular_subject": 0, "tabular_headline": 0, "tabular_body": 0,
        "sat": 0, "exchange_pr": 0, "unembeddable": 0,
        "insider": 0, "npl": 0, "blk": 0,
    }
    total = passes = 0

    shards = sorted(SOURCE_DIR.glob(f"{args.month}*_clean.jsonl"))
    for sp in shards:
        with open(sp) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1

                if is_pre_floor:
                    counts["date_floor"] += 1; continue

                subs = set(d.get("codes", {}).get("subject", []) or [])
                products = set(d.get("codes", {}).get("product", []) or [])
                headline = d.get("headline") or ""
                text = d.get("text") or ""

                if subs & LIFESTYLE_DENYLIST:
                    counts["lifestyle"] += 1; continue
                if subs & TABLE_SUBJECTS:
                    counts["tabular_subject"] += 1; continue
                if TABLE_HEADLINE_RE.match(headline) or CALENDAR_HEADLINE_RE.search(headline):
                    counts["tabular_headline"] += 1; continue
                if d.get("accession_number") in tab_body:
                    counts["tabular_body"] += 1; continue
                if subs & SAT_SUBJECTS:
                    counts["sat"] += 1; continue
                if products & EXCHANGE_PR_PRODUCTS:
                    counts["exchange_pr"] += 1; continue
                if unembeddable(text):
                    counts["unembeddable"] += 1; continue
                if subs & INSIDER_SUBJECTS:
                    counts["insider"] += 1; continue
                if subs & NPL_SUBJECTS:
                    counts["npl"] += 1; continue
                if subs & BLK_SUBJECTS:
                    counts["blk"] += 1; continue
                passes += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "month": args.month,
            "total": total,
            "removed": counts,
            "passes_all": passes,
        }, f, indent=2)
        f.write("\n")
    print(f"[waterfall] {args.month}: total={total:,} passes_all={passes:,}")


if __name__ == "__main__":
    main()
