#!/usr/bin/env python3
"""DJNW tag census and non-macro denylist audit.

This is the general DJNW tag analysis -- companion to djnw_crude_oil_audit.py
(which focuses on the crude-oil case study).

Subcommands:
    census     Full per-code census across all taxonomy fields. Writes
                 results/djnw_tag_census.csv           (per-code totals)
                 results/djnw_tag_coverage_monthly.csv (monthly documented
                                                         vs undocumented)
    denylist   Find non-macro subject codes (sports, lifestyle, obits, ...)
               using keyword match on descriptions + human-reviewable
               article samples. Writes
                 results/djnw_denylist_candidates.csv
                 results/djnw_denylist_candidates.pdf  (if matplotlib)
    all        Both.

Typical usage:
    .venv/bin/python scripts/djnw_tag_audit.py census
    .venv/bin/python scripts/djnw_tag_audit.py denylist
    .venv/bin/python scripts/djnw_tag_audit.py all

The denylist subcommand is designed to produce CANDIDATES with evidence
(sample headlines), not a finalized denylist. The final list is a human-
reviewed subset: open the CSV, read the samples, promote rows that are
unambiguously non-macro into loaders.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = Path("/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles")
DOCS_DIR = Path("/nfs/roberts/project/pi_btk22/jyz32/macronews/docs")
RESULTS_DIR = Path("/nfs/roberts/project/pi_btk22/jyz32/macronews/results")

# census outputs
OUT_CENSUS = RESULTS_DIR / "djnw_tag_census.csv"
OUT_MONTHLY = RESULTS_DIR / "djnw_tag_coverage_monthly.csv"

# denylist outputs
OUT_DENYLIST = RESULTS_DIR / "djnw_denylist_candidates.csv"
OUT_DENYLIST_PLOT = RESULTS_DIR / "djnw_denylist_candidates.pdf"
CACHE_DENYLIST = RESULTS_DIR / "djnw_denylist_scan.cache.json"

# Date floor for the denylist scan. 1996-01 matches the main spec date range
# (pre-1996 is dominated by wire-code spam that isn't representative of the
# selective-tagging era).
DENYLIST_DATE_CUTOFF = "1996-01"
DENYLIST_SAMPLES_PER_CANDIDATE = 8

# ---------------------------------------------------------------------------
# Taxonomy fields
# ---------------------------------------------------------------------------

TAXONOMY_FIELDS = ("subject", "industry", "market", "product", "geo")
ID_FIELDS = ("company", "isin")
ALL_FIELDS = TAXONOMY_FIELDS + ID_FIELDS

# ---------------------------------------------------------------------------
# Denylist keyword config
# ---------------------------------------------------------------------------

# Substring match on CSV description (case-insensitive).
NON_MACRO_KEYWORDS = [
    # direct non-macro categories
    "sport", "recreat", "obituar", "entertain", "arts ",
    "leisure", "travel", "tourism", "lifestyle", "fashion",
    "cooking", "religion", "holiday",
    # sports subcategories
    "basketball", "baseball", "football", "golf", "hockey",
    "olymp", "soccer", "tennis", "boxing",
    # culture / entertainment
    "music", "theater", "cinema", "celebrity",
    # reviews
    "review",
    # personal / household
    "wedding", "funeral", "horoscope", "puzzle", "crossword",
    "gardening", "cartoon", "comics", "hobb",
    # gambling
    "gamble", "lottery", "casino",
]

# Descriptions that mention something OTHER than the non-macro keyword at the
# same time (e.g. "petroleum" contains 'pet', "agriculture" contains 'culture',
# "transportation" contains 'sport'). Keep these -- they are macro-relevant.
MACRO_OVERRIDE_PHRASES = [
    "petroleum", "petrochem", "opec",
    "agricultur",
    "transport",
    "social security", "social insurance",
    "commercial real estate",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_documented_codes() -> dict[str, tuple[str, str]]:
    """Return {code: (source_csv_filename, description)} from all docs CSVs."""
    documented: dict[str, tuple[str, str]] = {}
    for csv_path in sorted(DOCS_DIR.glob("*.csv")):
        with open(csv_path) as fh:
            for row in csv.reader(fh):
                if len(row) < 2:
                    continue
                code, desc = row[0].strip(), row[1].strip()
                if code and code != "symbol" and not code.startswith("#"):
                    # first file that documents a code wins (stable)
                    documented.setdefault(code, (csv_path.name, desc))
    return documented


def iter_jsonl(files: list[Path]):
    for fp in files:
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _extract_code_from_field_value(item) -> str:
    """company_sig / isin_sig entries can be dicts with a 'code' field."""
    if isinstance(item, dict):
        return item.get("code") or item.get("isin") or ""
    return str(item)


def month_key(article: dict) -> str | None:
    docdate = article.get("docdate", "")
    if len(docdate) < 6:
        return None
    return f"{docdate[:4]}-{docdate[4:6]}"


# ---------------------------------------------------------------------------
# census — full per-code census across every taxonomy field
# ---------------------------------------------------------------------------

def cmd_census() -> None:
    documented = load_documented_codes()
    print(f"[census] loaded {len(documented):,} documented codes from "
          f"{len(list(DOCS_DIR.glob('*.csv')))} CSVs")

    # {code: {"field": str, "total": n, "first_month": ym, "last_month": ym}}
    code_counts: dict[str, dict] = {}
    # month -> field -> {"doc": n, "undoc": n, "total": n}
    monthly: dict[str, dict] = defaultdict(
        lambda: {f: {"doc": 0, "undoc": 0, "total": 0} for f in TAXONOMY_FIELDS}
    )
    # month -> aggregate across taxonomy fields
    monthly_all: dict[str, dict] = defaultdict(lambda: {"doc": 0, "undoc": 0, "total": 0})

    total_articles = 0
    files = sorted(RAW_DIR.glob("*_clean.jsonl"))
    print(f"[census] scanning {len(files)} JSONL files ...")

    for art in iter_jsonl(files):
        m = month_key(art)
        if m is None:
            continue
        total_articles += 1
        codes_obj = art.get("codes", {}) or {}

        for field in ALL_FIELDS:
            vals = codes_obj.get(field, []) or []
            if not isinstance(vals, list):
                continue
            for item in vals:
                code = _extract_code_from_field_value(item)
                if not code:
                    continue
                # per-code stats
                info = code_counts.get(code)
                if info is None:
                    code_counts[code] = {
                        "field": field, "total": 1,
                        "first_month": m, "last_month": m,
                    }
                else:
                    info["total"] += 1
                    info["last_month"] = m
                # monthly coverage for taxonomy fields
                if field in TAXONOMY_FIELDS:
                    is_doc = code in documented
                    monthly[m][field]["total"] += 1
                    monthly_all[m]["total"] += 1
                    if is_doc:
                        monthly[m][field]["doc"] += 1
                        monthly_all[m]["doc"] += 1
                    else:
                        monthly[m][field]["undoc"] += 1
                        monthly_all[m]["undoc"] += 1

    OUT_CENSUS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CENSUS, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "code", "field", "documented", "description", "doc_csv",
            "total_occurrences", "first_month", "last_month",
        ])
        for code in sorted(code_counts, key=lambda c: -code_counts[c]["total"]):
            info = code_counts[code]
            is_doc = code in documented
            desc = documented[code][1] if is_doc else ""
            doc_csv = documented[code][0] if is_doc else ""
            w.writerow([
                code, info["field"], is_doc, desc, doc_csv,
                info["total"], info["first_month"], info["last_month"],
            ])
    print(f"[census] wrote {OUT_CENSUS} ({len(code_counts):,} unique codes)")

    months = sorted(monthly)
    with open(OUT_MONTHLY, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["month"]
        for field in TAXONOMY_FIELDS:
            header += [f"{field}_doc", f"{field}_undoc", f"{field}_total", f"{field}_coverage"]
        header += ["all_doc", "all_undoc", "all_total", "all_coverage"]
        w.writerow(header)
        for m in months:
            row = [m]
            for field in TAXONOMY_FIELDS:
                d = monthly[m][field]
                cov = d["doc"] / d["total"] if d["total"] else 1.0
                row += [d["doc"], d["undoc"], d["total"], f"{cov:.4f}"]
            d = monthly_all[m]
            cov = d["doc"] / d["total"] if d["total"] else 1.0
            row += [d["doc"], d["undoc"], d["total"], f"{cov:.4f}"]
            w.writerow(row)
    print(f"[census] wrote {OUT_MONTHLY} ({len(months)} months)")

    # Summary
    print(f"\n[census] total articles scanned: {total_articles:,}")
    field_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"unique": 0, "doc": 0, "undoc": 0})
    for code, info in code_counts.items():
        f = info["field"]
        field_stats[f]["unique"] += 1
        if code in documented:
            field_stats[f]["doc"] += 1
        else:
            field_stats[f]["undoc"] += 1
    print(f"\n{'Field':<12s} {'Unique':>8s} {'Documented':>12s} {'Undocumented':>14s} {'Coverage':>10s}")
    print("-" * 60)
    for f in ALL_FIELDS:
        s = field_stats[f]
        cov = s["doc"] / s["unique"] * 100 if s["unique"] else 0
        print(f"{f:<12s} {s['unique']:>8,} {s['doc']:>12,} {s['undoc']:>14,} {cov:>9.1f}%")

    # Top undocumented
    print("\n[census] Top 30 undocumented taxonomy codes by occurrence:")
    undoc = [
        (code, info) for code, info in code_counts.items()
        if code not in documented and info["field"] in TAXONOMY_FIELDS
    ]
    undoc.sort(key=lambda x: -x[1]["total"])
    for code, info in undoc[:30]:
        print(f"  {code:<16s} field={info['field']:<10s} "
              f"{info['total']:>12,} occurrences  "
              f"({info['first_month']} to {info['last_month']})")


# ---------------------------------------------------------------------------
# denylist — find non-macro N/ codes with evidence
# ---------------------------------------------------------------------------

def _select_denylist_candidates(
    documented: dict[str, tuple[str, str]],
) -> dict[str, tuple[str, str, str]]:
    """{code: (description, matched_keyword, source_csv)} for N/ codes whose
    descriptions look non-macro by keyword heuristic."""
    out: dict[str, tuple[str, str, str]] = {}
    for code, (src_csv, desc) in documented.items():
        if not code.startswith("N/"):
            continue
        d_low = desc.lower()
        if any(ov in d_low for ov in MACRO_OVERRIDE_PHRASES):
            continue
        matched = next((kw for kw in NON_MACRO_KEYWORDS if kw in d_low), None)
        if matched is not None:
            out[code] = (desc, matched, src_csv)
    return out


def _scan_for_denylist(
    candidate_codes: set[str],
    samples_per_code: int,
) -> tuple[Counter, dict[str, list[str]], int]:
    """Single streaming pass over every JSONL >= DENYLIST_DATE_CUTOFF.

    Returns (code_counts, samples_by_code, article_count)."""
    code_counts: Counter = Counter()
    samples: dict[str, list[str]] = {c: [] for c in candidate_codes}
    article_count = 0

    files = sorted(RAW_DIR.glob("*_clean.jsonl"))
    files = [f for f in files if f.name[:7] >= DENYLIST_DATE_CUTOFF]
    print(f"[denylist:scan] {len(files)} monthly files from {files[0].name[:7]} "
          f"to {files[-1].name[:7]}")

    for art in iter_jsonl(files):
        article_count += 1
        subs = art.get("codes", {}).get("subject", []) or []
        for c in subs:
            if not isinstance(c, str):
                continue
            code_counts[c] += 1
            if c in candidate_codes and len(samples[c]) < samples_per_code:
                aid = art.get("accession_number", "")
                hl = (art.get("headline") or "").strip().replace("\n", " ")
                samples[c].append(f"{aid} | {hl[:140]}")
    return code_counts, samples, article_count


def cmd_denylist(refresh: bool = False) -> None:
    documented = load_documented_codes()
    candidates = _select_denylist_candidates(documented)
    print(f"[denylist] {len(candidates)} candidate codes flagged by keyword match")

    # Cache the scan -- scanning 31M articles every run is wasteful.
    if CACHE_DENYLIST.exists() and not refresh:
        print(f"[denylist] using cache: {CACHE_DENYLIST.name} "
              f"(delete or pass --refresh to rescan)")
        cache = json.loads(CACHE_DENYLIST.read_text())
        code_counts = Counter(cache["code_counts"])
        samples = cache["samples"]
        article_count = cache["article_count"]
        missing = [c for c in candidates if c not in samples]
        if missing:
            print(f"[denylist] cache missing samples for {len(missing)} new "
                  f"candidates; rescanning")
            code_counts, samples, article_count = _scan_for_denylist(
                set(candidates), DENYLIST_SAMPLES_PER_CANDIDATE,
            )
            CACHE_DENYLIST.write_text(json.dumps(
                {"code_counts": dict(code_counts), "samples": samples,
                 "article_count": article_count}
            ))
    else:
        code_counts, samples, article_count = _scan_for_denylist(
            set(candidates), DENYLIST_SAMPLES_PER_CANDIDATE,
        )
        CACHE_DENYLIST.parent.mkdir(parents=True, exist_ok=True)
        CACHE_DENYLIST.write_text(json.dumps(
            {"code_counts": dict(code_counts), "samples": samples,
             "article_count": article_count}
        ))
        print(f"[denylist] cached to {CACHE_DENYLIST.name}")
    print(f"[denylist] scanned {article_count:,} articles, "
          f"{len(code_counts):,} unique codes seen")

    # Write candidate CSV
    rows = sorted(candidates.items(), key=lambda kv: -code_counts.get(kv[0], 0))
    OUT_DENYLIST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_DENYLIST, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "code", "description", "matched_keyword", "source_csv",
            "article_count", "share_of_corpus",
            *[f"sample_{i+1}" for i in range(DENYLIST_SAMPLES_PER_CANDIDATE)],
        ])
        for code, (desc, kw, src) in rows:
            freq = code_counts.get(code, 0)
            share = freq / article_count if article_count else 0
            row_samples = samples.get(code, []) + [""] * DENYLIST_SAMPLES_PER_CANDIDATE
            w.writerow([
                code, desc, kw, src, freq, f"{share:.6f}",
                *row_samples[:DENYLIST_SAMPLES_PER_CANDIDATE],
            ])
    print(f"[denylist] wrote {OUT_DENYLIST}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[denylist] matplotlib not available; skipping plot")
    else:
        codes_freq = [(code, desc, code_counts.get(code, 0)) for code, (desc, _, _) in candidates.items()]
        codes_freq.sort(key=lambda x: -x[2])
        fig, ax = plt.subplots(figsize=(10, max(5, len(codes_freq) * 0.25)))
        ax.barh(range(len(codes_freq)), [r[2] for r in codes_freq])
        ax.set_yticks(range(len(codes_freq)))
        ax.set_yticklabels([f"{c}  {d[:40]}" for c, d, _ in codes_freq], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"Article count (DJNW since {DENYLIST_DATE_CUTOFF})")
        ax.set_title("Non-macro denylist candidates by frequency")
        plt.tight_layout()
        plt.savefig(OUT_DENYLIST_PLOT)
        plt.close(fig)
        print(f"[denylist] wrote {OUT_DENYLIST_PLOT}")

    # Print leaderboard
    print()
    print("[denylist] Candidate frequencies (descending). Review sample_N columns")
    print("           in the CSV before promoting a row into the loader denylist:")
    print()
    for code, (desc, kw, _) in rows:
        freq = code_counts.get(code, 0)
        print(f"  {code:<10s} {freq:>9,}  {desc[:50]:<50s} (kw={kw})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("census", help="full per-code census across all fields")
    p_dl = sub.add_parser("denylist", help="non-macro candidate audit with sample headlines")
    p_dl.add_argument("--refresh", action="store_true",
                      help="rescan corpus even if cache exists")
    p_all = sub.add_parser("all", help="census + denylist")
    p_all.add_argument("--refresh", action="store_true",
                       help="rescan corpus for denylist even if cache exists")

    args = p.parse_args()
    if args.cmd == "census":
        cmd_census()
    elif args.cmd == "denylist":
        cmd_denylist(refresh=args.refresh)
    elif args.cmd == "all":
        cmd_census()
        cmd_denylist(refresh=args.refresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
