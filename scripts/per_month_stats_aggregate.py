#!/usr/bin/env python3
"""Aggregate per-month stats JSON files into a single markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path("/nfs/roberts/project/pi_btk22/jyz32/macronews")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="Directory of per-month {YYYY-MM}.json files")
    ap.add_argument("--out", type=Path,
                    default=REPO / "docs" / "filter_audit" / "tabular_body" / "per_month_stats.md")
    ap.add_argument("--out-json", type=Path,
                    default=REPO / "docs" / "filter_audit" / "tabular_body" / "per_month_stats.json",
                    help="Sibling JSON file with the same aggregate counts (machine-readable)")
    args = ap.parse_args()

    rows = []
    for fp in sorted(args.in_dir.glob("*.json")):
        with open(fp) as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: r["month"])
    print(f"[agg] aggregating {len(rows)} months into {args.out}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Per-month tabular_body filter stats\n\n")
        f.write(f"- Months covered: {len(rows)}\n")
        f.write("- `existing`: caught by existing tabular_subject (N/TAB|N/DTA) or tabular_headline regex\n")
        f.write("- `tab_body`: flagged by new tabular_body filter\n")
        f.write("- `overlap`: caught by **both** existing AND tabular_body\n")
        f.write("- `additional`: caught **only** by tabular_body (new wins over existing)\n")
        f.write("- `missed_by_us`: caught **only** by existing (existing wins over tabular_body)\n")
        f.write("- `no_sidecar`: in cleaned shards but missing from sidecar (id mismatch indicator)\n\n")
        f.write("Per-month {accession_number, headline} lists for `additional`\n")
        f.write("and `missed` are stored in the per-month JSONs at\n")
        f.write("`results/tabular_stats/{YYYY-MM}.json`.\n\n")

        f.write("## Aggregate totals\n\n")
        tot = sum(r["counts"]["total"] for r in rows)
        ex = sum(r["counts"]["existing"] for r in rows)
        tb = sum(r["counts"]["tab_body"] for r in rows)
        ov = sum(r["counts"]["overlap"] for r in rows)
        ad = sum(r["counts"]["additional"] for r in rows)
        mi = sum(r["counts"]["missed_by_us"] for r in rows)
        ns = sum(r["counts"]["no_sidecar"] for r in rows)
        f.write(f"- Total articles: {tot:,}\n")
        f.write(f"- Existing tabular filters caught: {ex:,} ({ex/tot*100:.2f}%)\n")
        f.write(f"- tabular_body caught: {tb:,} ({tb/tot*100:.2f}%)\n")
        f.write(f"- Overlap (both): {ov:,} ({ov/tot*100:.2f}%)\n")
        f.write(f"- **Additional (only tabular_body): {ad:,} ({ad/tot*100:.2f}%)**\n")
        f.write(f"- **Missed by us (only existing): {mi:,} ({mi/tot*100:.2f}%)**\n")
        if ns:
            f.write(f"- WARN: {ns:,} articles missing from sidecar (id-mismatch edge case, ~0.0001% — safe; mapper still runs on them)\n")

        f.write("\n## Per-month\n\n")
        f.write("| month | total | existing | tab_body | overlap | additional | missed_by_us | %tab_body | %additional |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            c = r["counts"]
            total = c["total"]
            pct_body = c["tab_body"] / total * 100 if total else 0
            pct_add = c["additional"] / total * 100 if total else 0
            f.write(f"| {r['month']} | {total:,} | {c['existing']:,} | "
                    f"{c['tab_body']:,} | {c['overlap']:,} | {c['additional']:,} | "
                    f"{c['missed_by_us']:,} | {pct_body:.2f}% | {pct_add:.2f}% |\n")

    print(f"[agg] wrote {args.out}")

    # Sibling JSON: same data, machine-readable. Per-month counts only
    # (no id lists — those live in the per-month JSONs).
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "n_months": len(rows),
            "totals": {
                "total": tot,
                "existing": ex,
                "tab_body": tb,
                "overlap": ov,
                "additional": ad,
                "missed_by_us": mi,
                "no_sidecar": ns,
            },
            "per_month": [
                {"month": r["month"], "counts": r["counts"]}
                for r in rows
            ],
        }, f, indent=2)
        f.write("\n")
    print(f"[agg] wrote {args.out_json}")


if __name__ == "__main__":
    main()
