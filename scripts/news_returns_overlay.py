"""News-annotated price path (exploratory research instrument).

Cumulative log return of an asset group's future(s), overlaid with the dated,
signed, typed KG facts that AFFECT it (the fact's object is in the group's
bubble). Markers are placed at the article's exact `display_date` release time,
colored by sign and shaped by statement_type. Point it at any group/window.

    module load Python/3.12.3-GCCcore-13.3.0
    source .venv/bin/activate
    PYTHONPATH=src python scripts/news_returns_overlay.py \
        --group crude_oil --ticker CL --start 2014-05-01 --end 2014-05-31 \
        --out results/kg/gamma/crude_news_returns.html

Light-lane tool — not a production pipeline; we iterate on it visually.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plotly.graph_objects as go  # noqa: E402
from kg.article_meta import load_display_dates, parse_display_dt, shard_path  # noqa: E402

# signed-directional relation -> sign bucket. Direct (RAISES/DECREASES) and
# causal (CAUSES_RISE_IN/CAUSES_FALL_IN) both push the OBJECT up/down.
SIGN = {
    "RAISES": "up", "CAUSES_RISE_IN": "up",
    "DECREASES": "down", "CAUSES_FALL_IN": "down",
    "LEAVES_UNCHANGED": "flat", "IMPACT": "affect",
}
SIGN_COLOR = {"up": "#2ca02c", "down": "#d62728", "flat": "#7f7f7f", "affect": "#1f77b4"}
SYMBOL = {"FACT": "circle", "PREDICTION": "diamond-open", "OPINION": "circle-open"}


def load_cumret(csv_path: Path, ticker: str, start: str, end: str):
    """Trading dates + cumulative log return for `ticker` over [start, end]."""
    dates, cum, acc = [], [], 0.0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            d = row["date"]
            if start <= d <= end and row.get(ticker):
                try:
                    acc += float(row[ticker])
                except ValueError:
                    continue
                dates.append(d)
                cum.append(acc)
    return dates, cum


def collect_facts(disambig: Path, entity_groups: Path, group_key: str,
                  dd_map: dict[str, datetime], start: str, end: str) -> list[dict]:
    eg = json.loads(entity_groups.read_text())
    targets = {cf for cf, rec in eg.items()
               if any(g["key"] == group_key for g in rec["groups"])}
    facts = []
    for line in disambig.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ts = dd_map.get(row["article_id"])
        if ts is None or not (start <= ts.strftime("%Y-%m-%d") <= end):
            continue
        for ev in row.get("events", []):
            st, tt = ev.get("statement_type", ""), ev.get("temporal_type", "")
            for t in ev.get("triplets", []):
                rel = t["relation"]
                if rel in SIGN and t["object"].casefold() in targets:
                    facts.append({
                        "ts": ts, "day": ts.strftime("%Y-%m-%d"), "sign": SIGN[rel],
                        "stype": st, "ttype": tt, "rel": rel,
                        "subj": t["subject"], "obj": t["object"],
                        "stmt": ev.get("statement", ""),
                    })
    return facts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", required=True, help="asset group key, e.g. crude_oil")
    p.add_argument("--ticker", required=True, help="sync_daily column, e.g. CL")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--disambig", type=Path, default=Path("results/kg/gamma/2014-05.disambig.jsonl"))
    p.add_argument("--entity-groups", type=Path, default=Path("results/kg/gamma/2014-05.entity_groups.json"))
    p.add_argument("--returns", type=Path, default=Path("datasets/sync_daily.csv"))
    p.add_argument("--shards", nargs="+", default=["2014-05a", "2014-05b", "2014-05c"])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    dates, cum = load_cumret(args.returns, args.ticker, args.start, args.end)
    if not dates:
        sys.exit(f"No returns for ticker {args.ticker!r} in [{args.start},{args.end}]")
    cum_by_day = dict(zip(dates, cum))
    sorted_days = sorted(dates)

    dd_map = {aid: parse_display_dt(dd)
              for aid, dd in load_display_dates([shard_path(s) for s in args.shards]).items()}
    facts = collect_facts(args.disambig, args.entity_groups, args.group, dd_map,
                          args.start, args.end)

    # marker y = cumret of the last trading day <= the news day (point-in-time).
    def y_at(day: str):
        prior = [d for d in sorted_days if d <= day]
        return cum_by_day[prior[-1]] if prior else None

    for fct in facts:
        fct["y"] = y_at(fct["day"])
    facts = [f for f in facts if f["y"] is not None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=cum, mode="lines", name=f"{args.ticker} cum. log return",
        line=dict(color="#222", width=1.5), hoverinfo="x+y"))
    fig.add_trace(go.Scatter(
        x=[f["ts"] for f in facts], y=[f["y"] for f in facts], mode="markers",
        name="news facts",
        marker=dict(
            size=10,
            color=[SIGN_COLOR[f["sign"]] for f in facts],
            symbol=[SYMBOL.get(f["stype"], "circle") for f in facts],
            line=dict(width=1, color="#333")),
        customdata=[[f["subj"], f["rel"], f["obj"], f["stype"], f["ttype"], f["stmt"]]
                    for f in facts],
        hovertemplate=("<b>%{customdata[0]} —[%{customdata[1]}]→ %{customdata[2]}</b>"
                       "<br>%{customdata[3]} / %{customdata[4]} · %{x}"
                       "<br>%{customdata[5]}<extra></extra>")))
    n_up = sum(f["sign"] == "up" for f in facts)
    n_dn = sum(f["sign"] == "down" for f in facts)
    fig.update_layout(
        title=(f"{args.group} ({args.ticker}) — cumulative log return + KG news that "
               f"affects it · {args.start}→{args.end}<br>"
               f"<sub>{len(facts)} facts ({n_up}▲ {n_dn}▼) · green=↑ red=↓ grey=unchanged "
               f"blue=impact · circle=Fact diamond=Prediction open=Opinion</sub>"),
        xaxis_title="date", yaxis_title="cumulative log return",
        template="plotly_white", height=640, hovermode="closest")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.out))
    print(f"facts on {args.group}: {len(facts)} ({n_up}▲ / {n_dn}▼ / "
          f"{sum(f['sign']=='flat' for f in facts)}– / "
          f"{sum(f['sign']=='affect' for f in facts)}•) -> {args.out}")


if __name__ == "__main__":
    main()
