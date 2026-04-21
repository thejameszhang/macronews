#!/usr/bin/env python3
"""End-to-end DJNW crude-oil tagging audit.

Produces every artifact the DJNW metadata-audit slide deck cites for its
crude-oil case study:

  * results/djnw_crude_oil_tag_counts.csv  -- monthly count per oil code
  * slides/2026-04-14_djnw_metadata_audit/crude_tags_{absolute,
        absolute_wire, percentage, wire_percentage, heatmap}.pdf
  * results/missing_oil_tags_v3.txt        -- articles about oil
                                              missing every oil code
  * results/npet_ncmkt_intersection.csv    -- monthly N/PET vs N/CMKT
  * results/npet_not_ncmkt_sample.txt      -- 50 N/PET-but-no-N/CMKT articles
  * results/{npet,ncmkt,ioil,wire_union}_precision_sample.txt
                                           -- precision samples

Subcommands (run `--help` for each):
    counts        monthly code counts (slow; caches to the CSV above)
    plots         5 PDFs from the counts CSV
    missing       articles about oil missing every oil code
    intersection  N/PET vs N/CMKT overlap + sample
    precision     precision sample for a given code/union
    all           everything, in order

Typical usage:
    .venv/bin/python scripts/djnw_crude_oil_audit.py counts
    .venv/bin/python scripts/djnw_crude_oil_audit.py plots
    .venv/bin/python scripts/djnw_crude_oil_audit.py all

NOTE on DJNW code namespaces: N/ is overloaded (content + wire). In 1979-1996,
wire codes (N/IPR, N/NRG, N/DOI) are stamped on every article. Post-1996
they become selective (~15% in 2022) and carry useful weak-positive signal.
Full writeup in slides/2026-04-14_djnw_metadata_audit/.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = Path("/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles")
PROJECT_ROOT = Path("/nfs/roberts/project/pi_btk22/jyz32/macronews")
RESULTS_DIR = PROJECT_ROOT / "results"
SLIDE_DIR = PROJECT_ROOT / "slides" / "2026-04-14_djnw_metadata_audit"

# ---------------------------------------------------------------------------
# Oil codes — the audit's object of study
# ---------------------------------------------------------------------------

# Content codes (describe what the article is about).
CONTENT_CODES: dict[str, str] = {
    "N/PET":  "Crude Oil & Petroleum Products",
    "N/CMKT": "Crude Spot Market Commentary",
    "N/PRD":  "Oil & Natural Gas Production",
    "N/OPC":  "OPEC",
    "N/REF":  "Refinery Outages",
    "N/RMKT": "Refined Products Spot Market Commentary",
    "N/RPRI": "Refined Products Spot Market Prices",
    "I/OIL":  "Major Oil & Gas",
    "I/OIS":  "Oil Extraction",
    "I/FSL":  "Fossil Fuels",
    "M/ENE":  "Energy market",
    "N/ENY":  "Energy",
    "N/EGY":  "Energy Commentary",
    # Documented S/ codes exist but never appear in the data (kept here for
    # audit completeness / to make the "never appears" point obvious).
    "S/OIL":  "Oil",
    "S/API":  "American Petroleum Institute Weekly Statistics",
    "S/DOE":  "Department of Energy Weekly Data",
}

# Wire codes — which DJ service carried the article. Pre-1996 these are
# stamped on every article; post-1996 they become selective.
WIRE_CODES: dict[str, str] = {
    "N/IPR": "International Petroleum Report (wire)",
    "N/NRG": "Dow Jones Energy Service (wire)",
    "N/DOI": "Dow Jones Oil & Gas Service (wire)",
}

ALL_OIL_CODES: set[str] = set(CONTENT_CODES) | set(WIRE_CODES)
OIL_CODES_SORTED: list[str] = sorted(ALL_OIL_CODES)
CODE_DESCRIPTIONS: dict[str, str] = {**CONTENT_CODES, **WIRE_CODES}

# Codes for the %-share "content" plot. Curated subset.
CONTENT_SHARE_CODES = ["N/PET", "N/CMKT", "M/ENE", "N/OPC", "I/OIL", "N/ENY"]
# Codes that actually appear in the archive (S/ codes have zero occurrences)
CONTENT_CODES_IN_DATA = [
    "N/PET", "N/CMKT", "N/PRD", "N/OPC", "N/REF", "N/RMKT", "N/RPRI",
    "I/OIL", "I/OIS", "I/FSL", "M/ENE", "N/ENY", "N/EGY",
]

# Key oil events for plot annotations.
EVENTS: list[tuple[str, str]] = [
    ("1979-10", "Iran crisis"),
    ("1986-01", "Oil crash"),
    ("1990-08", "Iraq invades Kuwait"),
    ("1998-12", "Oil trough"),
    ("2003-03", "Iraq War"),
    ("2008-07", "$147 peak"),
    ("2014-11", "OPEC refuses cut"),
    ("2020-04", "WTI negative"),
    ("2022-02", "Russia-Ukraine"),
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def iter_jsonl(files: list[Path]):
    """Yield (file_path, article_dict) for every line in the given files."""
    for fp in files:
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield fp, json.loads(line)
                except json.JSONDecodeError:
                    continue


def collect_codes(article: dict, fields=("subject", "industry", "market", "product")) -> set[str]:
    """Flatten code fields into a single set of strings."""
    codes = article.get("codes", {}) or {}
    out: set[str] = set()
    for field in fields:
        vals = codes.get(field, []) or []
        if not isinstance(vals, list):
            continue
        for v in vals:
            if isinstance(v, str):
                out.add(v)
            elif isinstance(v, dict):
                c = v.get("code")
                if c:
                    out.add(c)
    return out


def month_key(article: dict) -> str | None:
    docdate = article.get("docdate", "")
    if len(docdate) < 6:
        return None
    return f"{docdate[:4]}-{docdate[4:6]}"


# ---------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------

OUT_COUNTS = RESULTS_DIR / "djnw_crude_oil_tag_counts.csv"


def cmd_counts() -> None:
    """Scan every JSONL file, count oil codes per month, write CSV."""
    # month -> {"total": n, code: count for code in OIL_CODES_SORTED}
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, **{c: 0 for c in OIL_CODES_SORTED}}
    )
    total = 0
    files = sorted(RAW_DIR.glob("*_clean.jsonl"))
    print(f"[counts] scanning {len(files)} JSONL files ...")

    for fp, art in iter_jsonl(files):
        m = month_key(art)
        if m is None:
            continue
        total += 1
        codes_here = collect_codes(art)
        counts[m]["total"] += 1
        for c in OIL_CODES_SORTED:
            if c in codes_here:
                counts[m][c] += 1

    OUT_COUNTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_COUNTS, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["month", "total_articles", *OIL_CODES_SORTED])
        for m in sorted(counts):
            row = counts[m]
            w.writerow([m, row["total"], *[row[c] for c in OIL_CODES_SORTED]])

    print(f"[counts] wrote {OUT_COUNTS} -- {len(counts)} months, {total:,} articles")
    for c in OIL_CODES_SORTED:
        tot = sum(counts[m][c] for m in counts)
        print(f"  {c:<10s} {CODE_DESCRIPTIONS[c]:<45s} {tot:>10,} ({tot/total:.2%})")


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _load_counts_csv() -> tuple[list[str], dict[str, dict[str, int]]]:
    """Read OUT_COUNTS. Returns (sorted_months, {month: {col: int}})."""
    if not OUT_COUNTS.exists():
        raise FileNotFoundError(
            f"{OUT_COUNTS} not found -- run `counts` subcommand first"
        )
    rows: dict[str, dict[str, int]] = {}
    with open(OUT_COUNTS) as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            m = r["month"]
            rows[m] = {k: int(v) for k, v in r.items() if k != "month"}
    return sorted(rows), rows


def _parse_ym(ym: str):
    """Parse YYYY-MM to datetime.date for matplotlib."""
    import datetime
    return datetime.date(int(ym[:4]), int(ym[5:7]), 1)


def _add_event_lines(ax, y_label_top):
    for ym, label in EVENTS:
        d = _parse_ym(ym)
        ax.axvline(d, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.text(d, y_label_top, label, rotation=90, fontsize=6,
                ha="right", va="top", color="gray", alpha=0.8)


def _iter_series(months: list[str], rows: dict, col: str) -> list:
    """Return list of values (or None for zero) matching _parse_ym(months)."""
    import math
    return [rows[m][col] if rows[m][col] > 0 else math.nan for m in months]


def cmd_plots() -> None:
    """5 PDFs. Requires matplotlib; skips gracefully if missing."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[plots] matplotlib not available; skipping plots subcommand")
        return

    months, rows = _load_counts_csv()
    dates = [_parse_ym(m) for m in months]
    SLIDE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[plots] loaded {len(months)} months; writing to {SLIDE_DIR}/")

    # 1. Absolute content codes, log scale
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab20
    colors = [cmap(i / 20) for i in range(len(CONTENT_CODES_IN_DATA))]
    for i, code in enumerate(CONTENT_CODES_IN_DATA):
        ax.plot(dates, _iter_series(months, rows, code), label=code,
                color=colors[i], linewidth=0.9, alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1, top=50000)
    ax.set_xlabel("Month")
    ax.set_ylabel("Articles tagged (log scale)")
    ax.set_title("DJNW oil content-code usage per month, 1979--2025")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    _add_event_lines(ax, y_label_top=40000)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=10, ncol=1, frameon=False)
    plt.tight_layout()
    out = SLIDE_DIR / "crude_tags_absolute.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  wrote {out}")

    # 2. Absolute wire codes, log scale
    fig, ax = plt.subplots(figsize=(12, 6))
    wcolors = {"N/IPR": "#BD5319", "N/NRG": "#8B2A0A", "N/DOI": "#4A1505"}
    for code in WIRE_CODES:
        ax.plot(dates, _iter_series(months, rows, code),
                label=f"{code} -- {CODE_DESCRIPTIONS[code]}",
                color=wcolors[code], linewidth=1.2, alpha=0.9, linestyle="--")
    ax.set_yscale("log"); ax.set_ylim(bottom=1, top=200000)
    ax.set_xlabel("Month"); ax.set_ylabel("Articles tagged (log scale)")
    ax.set_title("DJNW oil wire-code usage per month, 1979--2025")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)
    _add_event_lines(ax, y_label_top=150000)
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)
    plt.tight_layout()
    out = SLIDE_DIR / "crude_tags_absolute_wire.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  wrote {out}")

    # 3. Content share (0-10%)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    share_colors = {"N/PET": "#00356B", "N/CMKT": "#CC0000",
                    "M/ENE": "#BD5319", "N/OPC": "#2E8B57",
                    "I/OIL": "#286DC0", "N/ENY": "#8B0A50"}
    for code in CONTENT_SHARE_CODES:
        pct = [100 * rows[m][code] / rows[m]["total_articles"] if rows[m]["total_articles"] else 0
               for m in months]
        ax.plot(dates, pct, label=f"{code} -- {CODE_DESCRIPTIONS[code]}",
                color=share_colors[code], linewidth=1.2, alpha=0.9)
    ax.set_xlabel("Month"); ax.set_ylabel("% of DJNW articles tagged")
    ax.set_title("Share of articles carrying content codes (1979--2025)")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, which="major", alpha=0.3)
    ax.set_ylim(bottom=0, top=10)
    _add_event_lines(ax, y_label_top=9.5)
    ax.legend(loc="upper left", fontsize=7, frameon=True, framealpha=0.9)
    plt.tight_layout()
    out = SLIDE_DIR / "crude_tags_percentage.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  wrote {out}")

    # 4. Wire share (0-100%)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for code in WIRE_CODES:
        pct = [100 * rows[m][code] / rows[m]["total_articles"] if rows[m]["total_articles"] else 0
               for m in months]
        ax.plot(dates, pct, label=f"{code} -- {CODE_DESCRIPTIONS[code]}",
                color=wcolors[code], linewidth=1.2, alpha=0.9, linestyle="--")
    ax.set_xlabel("Month"); ax.set_ylabel("% of DJNW articles tagged")
    ax.set_title("Share of articles carrying wire codes (1979--2025)")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, which="major", alpha=0.3)
    ax.set_ylim(bottom=0, top=105)
    _add_event_lines(ax, y_label_top=100)
    ax.legend(loc="center right", fontsize=7, frameon=True, framealpha=0.9)
    plt.tight_layout()
    out = SLIDE_DIR / "crude_tags_wire_percentage.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  wrote {out}")

    # 5. Heatmap at key events
    heatmap_events = [
        ("1979-11", "Iran crisis (Nov '79)"),
        ("1986-02", "Oil crash (Feb '86)"),
        ("1990-08", "Iraq invades Kuwait (Aug '90)"),
        ("1991-01", "Gulf War (Jan '91)"),
        ("1998-12", "Oil trough (Dec '98)"),
        ("2003-03", "Iraq War (Mar '03)"),
        ("2008-07", "$147 peak (Jul '08)"),
        ("2008-10", "Oil crash (Oct '08)"),
        ("2014-11", "OPEC refuses cut (Nov '14)"),
        ("2020-04", "WTI negative (Apr '20)"),
        ("2022-02", "Russia-Ukraine (Feb '22)"),
        ("2022-03", "Oil spikes (Mar '22)"),
    ]
    codes_in_heatmap = CONTENT_CODES_IN_DATA + list(WIRE_CODES)
    import numpy as np
    data = np.zeros((len(heatmap_events), len(codes_in_heatmap)))
    labels = []
    for i, (m, lbl) in enumerate(heatmap_events):
        if m not in rows:
            labels.append(lbl + " (no data)")
            continue
        labels.append(lbl)
        t = rows[m]["total_articles"]
        for j, code in enumerate(codes_in_heatmap):
            data[i, j] = 100 * rows[m][code] / t if t else 0
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=8)
    ax.set_xticks(range(len(codes_in_heatmap)))
    ax.set_xticklabels(codes_in_heatmap, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("% of articles that month")
    ax.set_title("DJNW crude-oil tag coverage during key oil events")
    for i in range(len(heatmap_events)):
        for j in range(len(codes_in_heatmap)):
            if data[i, j] > 0.5:
                ax.text(j, i, f"{data[i, j]:.1f}",
                        ha="center", va="center", fontsize=6, color="black")
    plt.tight_layout()
    out = SLIDE_DIR / "crude_tags_heatmap.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# missing — articles about oil that lack every oil code
# ---------------------------------------------------------------------------

OUT_MISSING = RESULTS_DIR / "missing_oil_tags_v3.txt"

TOPIC_PATTERNS = [
    r"crude oil", r"oil price", r"oil output", r"oil production",
    r"oil supply", r"oil demand", r"oil market", r"oil futures",
    r"oil settlem", r"oil reserve", r"oil export", r"oil import",
    r"oil revenue", r"oil embargo", r"oil crisis", r"oil glut",
    r"oil shock", r"oil surge", r"oil tumble", r"oil plunge",
    r"oil spike", r"oil rally", r"oil slump", r"oil decline",
    r"oil drop", r"oil fall", r"oil rise", r"oil jump",
    r"oil gain", r"oil sell", r"oil inventori", r"oil stockpile",
    r"crude price", r"crude future", r"crude settlem", r"crude market",
    r"crude output", r"crude production",
    r"nymex crude", r"brent crude", r"wti crude",
    r"barrel.{0,5}oil", r"oil.{0,10}barrel",
    r"\boil.{0,5}bbl\b", r"\bbbl\b.{0,5}oil",
    r"opec.{0,20}(oil|crude|cut|output|product|quota|supply|barrel)",
    r"(oil|crude).{0,20}opec",
    r"petroleum price", r"petroleum product", r"petroleum export",
    r"petroleum import",
    r"\$\d+.{0,3}/barrel", r"\$\d+.{0,3}/bbl",
    r"\$\d+.{0,3} a barrel", r"per barrel", r"a barrel", r"/bbl",
    r"oil.{0,10}(higher|lower|up|down|rose|fell|climb|sank|soar|plummet|dip)",
    r"(higher|lower|up|down|rose|fell|climb|sank|soar|plummet|dip).{0,10}oil",
    r"saudi.{0,15}(oil|crude|production|cut|increase|quota|barrels)",
    r"(uae|iran|iraq|russia|venezuela|nigeria|libya).{0,15}(oil|crude|production|export)",
    r"opec",
]
TOPIC_RE = re.compile("|".join(TOPIC_PATTERNS), re.IGNORECASE)

EXCLUDE_PATTERNS = [
    r"petroleum\s+\d+[QH]\s",
    r"petroleum\s+(year|qtr|quarter)\s",
    r"petroleum\s+(names|appoints|elects)",
    r"petroleum\s+(completes|closes|signs)",
    r"petroleum\s+(shares|stock|common)",
    r"petroleum\s+(provides|announces|reports)\s.*\b(update|offering|split)\b",
    r"\(.*\)\s*(resumed|halted|ind:|no mkt)",
    r"imbalance",
    r"^Dir\s+\w+\s+(buys|sells)",
    r"^hot stocks",
    r"palm oil", r"soyoil", r"soybean oil", r"vegetable oil", r"olive oil",
    r"cooking oil",
    r"cme group agricultural",
    r"cbot delivery",
]
EXCLUDE_RE = re.compile("|".join(EXCLUDE_PATTERNS), re.IGNORECASE)

DIRECT_LEAD_PATTERNS = [
    r"^(saudi|opec|iran|iraq|russia|venezuela|uae|nigeria|libya|kuwait|qatar)",
    r"^(crude|oil|petroleum|nymex|brent|wti|ice brent)",
    r"^(refinery|refiner|refining)",
    r"^(gasoline|heating oil|distillate|diesel)",
    r"(oil|crude).{0,20}(price|output|production|supply|export|import|cut|quota)",
    r"(oil|petroleum).{0,20}(policy|agreement|deal|sanction|embargo)",
]
DIRECT_RE = re.compile("|".join(DIRECT_LEAD_PATTERNS), re.IGNORECASE)

CROSS_LEAD_PATTERNS = [
    r"^(ftse|dow jones|s&p|nasdaq|nikkei|stoxx|dax|cac)",
    r"^(canadian dollar|aussie|yen|euro|sterling|pound|loonie|ruble|rouble)",
    r"^(treasur|bund|gilt|jgb|bond)",
    r"^(gold|silver|copper|aluminum|wheat|sugar|cocoa|coffee)",
    r"^(stocks|stock|equities|equity|shares|markets)",
    r"^(us |u\.s\. |european |asian |asia |world |global )",
    r"(stocks|stock|equities|bonds|currency|treasury).{0,20}(oil|crude)",
    r"^(toronto|london|tokyo|shanghai|hong kong) (stocks|market)",
    r"^airline|airlines",
    r"^news highlights",
    r"^market wraps",
    r"^morning briefing|^north american morning",
    r"^top .*news of the day",
]
CROSS_RE = re.compile("|".join(CROSS_LEAD_PATTERNS), re.IGNORECASE)

EVENT_MONTHS_MISSING: list[tuple[str, str]] = [
    ("1998-12", "Oil price trough"),
    ("2003-03", "Iraq War begins"),
    ("2003-04", "Iraq War early phase"),
    ("2008-07", "Oil hits $147 peak"),
    ("2008-10", "Oil crash (financial crisis)"),
    ("2014-11", "OPEC refuses to cut"),
    ("2014-12", "Oil crash continues"),
    ("2015-01", "Oil crash continues"),
    ("2020-03", "COVID + Saudi-Russia price war"),
    ("2020-04", "WTI goes negative"),
    ("2022-02", "Russia invades Ukraine"),
    ("2022-03", "Oil spikes on war"),
    ("2016-09", "OPEC Algiers agreement (cut)"),
    ("2016-11", "OPEC Vienna deal"),
    ("2018-05", "US withdraws Iran deal"),
    ("2018-11", "Iran sanctions take effect"),
    ("2019-09", "Saudi Aramco Abqaiq drone attack"),
    ("2020-01", "Soleimani assassination"),
    ("2023-10", "Israel-Hamas war"),
    ("2024-06", "OPEC+ extension decision"),
]


def _classify_missing(headline: str) -> str:
    if DIRECT_RE.search(headline):
        return "DIRECT"
    if CROSS_RE.search(headline):
        return "CROSS_ASSET"
    return "AMBIGUOUS"


def cmd_missing() -> None:
    OUT_MISSING.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines += [
        "=" * 100,
        "DJNW ARTICLES ABOUT CRUDE OIL MISSING ALL 19 OIL CODES",
        "=" * 100,
        "",
        "Classification: DIRECT = article is primarily about crude oil.",
        "                CROSS_ASSET = other asset (FX, stocks, bonds) reacting to oil.",
        "                AMBIGUOUS = neither pattern clearly matched.",
        "",
        f"Checked against {len(ALL_OIL_CODES)} codes: {', '.join(sorted(ALL_OIL_CODES))}",
        "",
    ]

    by_cat: dict[str, list[dict]] = {"DIRECT": [], "CROSS_ASSET": [], "AMBIGUOUS": []}
    summary_rows: list[tuple] = []

    for month_prefix, event_name in EVENT_MONTHS_MISSING:
        total = topic = 0
        by_cat_event = {"DIRECT": 0, "CROSS_ASSET": 0, "AMBIGUOUS": 0}
        files = sorted(RAW_DIR.glob(f"{month_prefix}*_clean.jsonl"))
        for _fp, art in iter_jsonl(files):
            total += 1
            headline = art.get("headline", "") or ""
            if not TOPIC_RE.search(headline):
                text_start = (art.get("text", "") or "")[:200]
                if not TOPIC_RE.search(headline + " " + text_start):
                    continue
            if EXCLUDE_RE.search(headline):
                continue
            topic += 1
            all_codes = collect_codes(art)
            if all_codes & ALL_OIL_CODES:
                continue
            cat = _classify_missing(headline)
            by_cat_event[cat] += 1
            by_cat[cat].append({
                "an": art.get("accession_number", ""),
                "date": art.get("docdate", ""),
                "event": event_name,
                "headline": headline,
                "codes": sorted(all_codes),
                "text_snippet": (art.get("text", "") or "")[:300],
            })
        missing_n = sum(by_cat_event.values())
        summary_rows.append((month_prefix, event_name, total, topic, missing_n, by_cat_event))

    # summary table
    lines += [
        "SUMMARY STATISTICS",
        "-" * 100,
        f"{'Month':<10s} {'Event':<35s} {'Total':>8s} {'Topic':>8s} {'Miss':>6s} "
        f"{'Direct':>7s} {'Cross':>7s} {'Ambig':>7s}",
        "-" * 100,
    ]
    grand = {"topic": 0, "miss": 0, "DIRECT": 0, "CROSS_ASSET": 0, "AMBIGUOUS": 0}
    for month, event, total, topic, miss, cats in summary_rows:
        lines.append(
            f"{month:<10s} {event:<35s} {total:>8,} {topic:>8,} {miss:>6,} "
            f"{cats['DIRECT']:>7,} {cats['CROSS_ASSET']:>7,} {cats['AMBIGUOUS']:>7,}"
        )
        grand["topic"] += topic; grand["miss"] += miss
        for k in ("DIRECT", "CROSS_ASSET", "AMBIGUOUS"):
            grand[k] += cats[k]
    lines += [
        "-" * 100,
        f"{'TOTAL':<10s} {'':<35s} {'':<8s} {grand['topic']:>8,} {grand['miss']:>6,} "
        f"{grand['DIRECT']:>7,} {grand['CROSS_ASSET']:>7,} {grand['AMBIGUOUS']:>7,}",
        "",
    ]

    for cat in ("DIRECT", "CROSS_ASSET", "AMBIGUOUS"):
        lines += ["", "=" * 100, f"CATEGORY: {cat} ({len(by_cat[cat])} articles)", "=" * 100]
        by_event: dict[str, list] = {}
        for r in by_cat[cat]:
            by_event.setdefault(r["event"], []).append(r)
        for _, event_name in EVENT_MONTHS_MISSING:
            if event_name not in by_event:
                continue
            rs = by_event[event_name]
            lines += ["", f"  --- {event_name} ({len(rs)} articles) ---"]
            for r in rs[:15]:
                snippet = r["text_snippet"].replace("\n", " ")[:180]
                lines += [
                    "",
                    f"  AN: {r['an']}  |  {r['date']}",
                    f"  HEADLINE: {r['headline']}",
                    f"  Text: {snippet}...",
                ]
            if len(rs) > 15:
                lines.append(f"  ... and {len(rs) - 15} more")

    OUT_MISSING.write_text("\n".join(lines))
    print(f"[missing] wrote {OUT_MISSING}")
    for s in lines[:6]:
        print(s)


# ---------------------------------------------------------------------------
# intersection — N/PET vs N/CMKT
# ---------------------------------------------------------------------------

OUT_INTERSECT_CSV = RESULTS_DIR / "npet_ncmkt_intersection.csv"
OUT_PET_NOT_CMKT = RESULTS_DIR / "npet_not_ncmkt_sample.txt"
INTERSECT_TARGET_MONTH = "2022-03"
INTERSECT_N_SAMPLES = 50
INTERSECT_SEED = 42


def cmd_intersection() -> None:
    OUT_INTERSECT_CSV.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "npet": 0, "ncmkt": 0, "both": 0,
                 "pet_only": 0, "cmkt_only": 0}
    )
    pet_not_cmkt: list[dict] = []

    files = sorted(RAW_DIR.glob("*_clean.jsonl"))
    print(f"[intersection] scanning {len(files)} files ...")
    for _fp, art in iter_jsonl(files):
        m = month_key(art)
        if m is None:
            continue
        # Only subject + industry are where these codes can appear.
        codes = collect_codes(art, fields=("subject", "industry"))
        pet = "N/PET" in codes
        cmkt = "N/CMKT" in codes
        c = counts[m]
        c["total"] += 1
        if pet: c["npet"] += 1
        if cmkt: c["ncmkt"] += 1
        if pet and cmkt:
            c["both"] += 1
        elif pet:
            c["pet_only"] += 1
        elif cmkt:
            c["cmkt_only"] += 1
        if m == INTERSECT_TARGET_MONTH and pet and not cmkt:
            pet_not_cmkt.append(art)

    months = sorted(counts)
    with open(OUT_INTERSECT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "month", "total_articles", "n_pet", "n_cmkt", "both",
            "pet_only", "cmkt_only", "pct_cmkt_in_pet", "pct_pet_in_cmkt",
        ])
        for m in months:
            c = counts[m]
            pct_cmkt_in_pet = c["both"] / c["npet"] * 100 if c["npet"] else 0
            pct_pet_in_cmkt = c["both"] / c["ncmkt"] * 100 if c["ncmkt"] else 0
            w.writerow([m, c["total"], c["npet"], c["ncmkt"], c["both"],
                        c["pet_only"], c["cmkt_only"],
                        f"{pct_cmkt_in_pet:.2f}", f"{pct_pet_in_cmkt:.2f}"])
    print(f"[intersection] wrote {OUT_INTERSECT_CSV} ({len(months)} months)")

    any_cmkt_not_pet = any(counts[m]["cmkt_only"] > 0 for m in months)
    tot_cmkt = sum(counts[m]["ncmkt"] for m in months)
    tot_cmkt_not_pet = sum(counts[m]["cmkt_only"] for m in months)
    print("  Is N/CMKT a strict subset of N/PET across the archive?")
    if not any_cmkt_not_pet:
        print("    YES: N/CMKT ⊆ N/PET every month")
    else:
        print(f"    NO: {tot_cmkt_not_pet} / {tot_cmkt} ({tot_cmkt_not_pet/tot_cmkt*100:.2f}%) "
              f"N/CMKT articles lack N/PET")

    # Sample from target month
    print(f"[intersection] sampling {INTERSECT_N_SAMPLES} pet-not-cmkt from {INTERSECT_TARGET_MONTH}")
    rng = random.Random(INTERSECT_SEED)
    sample = rng.sample(pet_not_cmkt, min(INTERSECT_N_SAMPLES, len(pet_not_cmkt)))
    _write_precision_sample(
        OUT_PET_NOT_CMKT,
        f"{INTERSECT_N_SAMPLES} random articles tagged N/PET but NOT N/CMKT ({INTERSECT_TARGET_MONTH})",
        f"Population: {len(pet_not_cmkt):,} articles in {INTERSECT_TARGET_MONTH} with N/PET and no N/CMKT",
        sample,
    )


# ---------------------------------------------------------------------------
# precision — sample articles tagged with a given code (or the wire union)
# ---------------------------------------------------------------------------

PRECISION_TARGET_MONTH = "2022-03"
PRECISION_N_SAMPLES = 50
PRECISION_SEED = 42

PRECISION_CONFIGS = {
    "npet":       {"codes": {"N/PET"},  "desc": "N/PET — Crude Oil & Petroleum Products",
                   "out": RESULTS_DIR / "npet_precision_sample.txt"},
    "ncmkt":      {"codes": {"N/CMKT"}, "desc": "N/CMKT — Crude Spot Market Commentary",
                   "out": RESULTS_DIR / "ncmkt_precision_sample.txt"},
    "ioil":       {"codes": {"I/OIL"},  "desc": "I/OIL — Major Oil & Gas",
                   "out": RESULTS_DIR / "ioil_precision_sample.txt"},
    "wire_union": {"codes": set(WIRE_CODES), "desc": "N/IPR ∪ N/NRG ∪ N/DOI (wire union)",
                   "out": RESULTS_DIR / "wire_union_precision_sample.txt"},
}


def _write_precision_sample(out_path: Path, title: str, pop_line: str, sample: list[dict]) -> None:
    lines: list[str] = [
        "=" * 100,
        title,
        "=" * 100,
        "",
        pop_line,
        f"Sample size: {len(sample)}, random seed: {PRECISION_SEED}",
        "",
    ]
    for i, art in enumerate(sample, 1):
        all_codes = sorted(collect_codes(art))
        oil = [c for c in all_codes if c in ALL_OIL_CODES]
        text = ((art.get("text", "") or "")[:500]).replace("\n", " ")
        lines += [
            "-" * 100,
            f"[{i:2d}] AN: {art.get('accession_number', '')}  |  Date: {art.get('docdate', '')}",
            f"     HEADLINE: {art.get('headline', '')}",
            f"     Oil-related codes: {', '.join(oil)}",
            f"     TEXT: {text}...",
            "",
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"  wrote {out_path}")


def cmd_precision(which: str) -> None:
    cfg = PRECISION_CONFIGS[which]
    codes_wanted: set[str] = cfg["codes"]
    print(f"[precision:{which}] sampling from {PRECISION_TARGET_MONTH} on {cfg['desc']}")

    matching: list[dict] = []
    files = sorted(RAW_DIR.glob(f"{PRECISION_TARGET_MONTH}*_clean.jsonl"))
    for _fp, art in iter_jsonl(files):
        if collect_codes(art) & codes_wanted:
            matching.append(art)

    print(f"  population: {len(matching):,}")
    rng = random.Random(PRECISION_SEED)
    sample = rng.sample(matching, min(PRECISION_N_SAMPLES, len(matching)))
    _write_precision_sample(
        cfg["out"],
        f"PRECISION SAMPLE ({cfg['desc']}) -- {PRECISION_TARGET_MONTH}",
        f"Population: {len(matching):,} articles with any of {sorted(codes_wanted)} "
        f"in {PRECISION_TARGET_MONTH}",
        sample,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("counts", help="monthly oil-code counts -> CSV")
    sub.add_parser("plots", help="5 PDFs from the counts CSV")
    sub.add_parser("missing", help="articles about oil that lack every oil code")
    sub.add_parser("intersection", help="N/PET vs N/CMKT overlap")
    p_prec = sub.add_parser("precision", help="precision sample for a single code / wire union")
    p_prec.add_argument("which", choices=sorted(PRECISION_CONFIGS))
    sub.add_parser("all", help="run counts -> plots -> missing -> intersection -> all 4 precision samples")

    args = p.parse_args()

    if args.cmd == "counts":
        cmd_counts()
    elif args.cmd == "plots":
        cmd_plots()
    elif args.cmd == "missing":
        cmd_missing()
    elif args.cmd == "intersection":
        cmd_intersection()
    elif args.cmd == "precision":
        cmd_precision(args.which)
    elif args.cmd == "all":
        cmd_counts()
        cmd_plots()
        cmd_missing()
        cmd_intersection()
        for w in sorted(PRECISION_CONFIGS):
            cmd_precision(w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
