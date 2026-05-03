#!/usr/bin/env python3
"""Aggregate per-month waterfall JSONs into the final filter waterfall.

Inputs:  results/_tmp_waterfall_per_month/{YYYY-MM}.json (created by
         per_month_waterfall.py via SLURM array)
Outputs: results/filter_waterfall.csv
         docs/filter_audit/filter_waterfall.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path("/nfs/roberts/project/pi_btk22/jyz32/macronews")

STAGE_ORDER = [
    ("date_floor",      "Date floor (>= 1996-01-01)"),
    ("lifestyle",       "16-code lifestyle/sports denylist"),
    ("tabular_subject", "Tabular subject (N/TAB, N/DTA)"),
    ("tabular_headline","Tabular headline (TABLE: / calendar regex)"),
    ("tabular_body",    "Tabular body (NML structural detector)"),
    ("sat",             "N/SAT (Seeking Alpha transcripts)"),
    ("exchange_pr",     "Exchange press-release products (10)"),
    ("unembeddable",    "Unembeddable text (empty/wire-marker/corrections)"),
    ("insider",         "Insider filings (N/ISD, N/144, N/ISS, N/ISB)"),
    ("npl",             "Procedural template (N/NPL)"),
    ("blk",             "Block-trade tape subject (N/BLK)"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="Directory of per-month {YYYY-MM}.json files")
    ap.add_argument("--out-csv", type=Path,
                    default=REPO / "results" / "filter_waterfall.csv")
    ap.add_argument("--out-md", type=Path,
                    default=REPO / "docs" / "filter_audit" / "filter_waterfall.md")
    args = ap.parse_args()

    files = sorted(args.in_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No per-month JSONs found in {args.in_dir}")
    print(f"[agg] aggregating {len(files)} per-month JSONs")

    total = 0
    passes = 0
    removed: dict[str, int] = {k: 0 for k, _ in STAGE_ORDER}
    for fp in files:
        d = json.loads(fp.read_text())
        total += d["total"]
        passes += d["passes_all"]
        for k in removed:
            removed[k] += d["removed"].get(k, 0)

    # Build the waterfall: each stage shows remaining/removed/pct of running
    rows = [("Total DJNW articles (cleaned v2, all time)", total, None)]
    running = total
    for key, label in STAGE_ORDER:
        running -= removed[key]
        rows.append((f"After {label}", running, removed[key]))
    # Sanity: running == passes
    assert running == passes, f"running={running} != passes={passes} (off by {running - passes})"

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("stage,filter,remaining,removed,pct_of_prev\n")
        prev = None
        for idx, (name, rem, rmv) in enumerate(rows):
            pct = f"{100*(prev - rem)/prev:.2f}" if prev else ""
            f.write(f"{idx},\"{name}\",{rem},{rmv if rmv is not None else ''},{pct}\n")
            prev = rem
    print(f"[agg] wrote {args.out_csv}")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("# DJNW filter waterfall\n\n")
        f.write(f"- Source: cleaned v2 articles, all time (1979 onward)\n")
        f.write(f"- Months covered: {len(files)}\n")
        f.write(f"- Filters applied in waterfall order — each stage's removal "
                f"count is the marginal contribution after upstream stages.\n\n")
        f.write("| stage | remaining | removed | % of prev |\n")
        f.write("|---|---:|---:|---:|\n")
        prev = None
        for name, rem, rmv in rows:
            dr = f"{rmv:,}" if rmv is not None else ""
            pct = f"{100*(prev - rem)/prev:.2f}%" if prev else ""
            f.write(f"| {name} | {rem:,} | {dr} | {pct} |\n")
            prev = rem
        f.write(f"\n**Final remaining (passes all filters): {passes:,} articles**\n")
        f.write(f"\nThat's {passes/total*100:.1f}% of the cleaned-v2 corpus surviving "
                f"to the mapper.\n")
    print(f"[agg] wrote {args.out_md}")
    print(f"[agg] passes_all: {passes:,} / total {total:,} ({passes/total*100:.1f}%)")


if __name__ == "__main__":
    main()
