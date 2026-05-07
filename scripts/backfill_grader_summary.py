"""Recompute grader .summary.json files from the existing sidecar .jsonl.

Older grader runs (groups-v2/.../groups-v8) wrote summary.json without the
`by_asset_class` and `by_group` breakdowns. This script reads each sidecar
JSONL, replays the same aggregation logic the runner now uses, and rewrites
the matching .summary.json so all historical runs are comparable.

Idempotent: running it on an already-backfilled summary just rewrites the
same content.

Usage:
    python scripts/backfill_grader_summary.py results/groups-v8-grader/gold.jsonl
    python scripts/backfill_grader_summary.py results/prod/groups-v8-grader/2006-03c.jsonl
    python scripts/backfill_grader_summary.py results/**/*-grader/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def aggregate(sidecar_path: Path) -> dict:
    """Replay write_sidecar's aggregation over an existing JSONL."""
    n_correct = n_incorrect = n_inconsistent = 0
    n_graded = n_skipped = 0
    skip_reasons: dict[str, int] = {}
    by_class: dict[str, dict[str, int]] = {}
    by_group: dict[str, dict[str, int]] = {}

    def _bump(d: dict, key: str, verdict: str, flag: bool) -> None:
        bucket = d.setdefault(
            key, {"total": 0, "correct": 0, "incorrect": 0, "self_inconsistent": 0}
        )
        bucket["total"] += 1
        bucket[verdict] += 1
        if flag:
            bucket["self_inconsistent"] += 1

    with open(sidecar_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["grader_verdict"] == "skipped":
                n_skipped += 1
                reason = row.get("skip_reason") or "unknown"
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue
            n_graded += 1
            verdict = row["grader_verdict"]
            flag = bool(row.get("self_consistency_flag"))
            if verdict == "correct":
                n_correct += 1
            else:
                n_incorrect += 1
            if flag:
                n_inconsistent += 1
            _bump(by_class, row["asset_class"], verdict, flag)
            _bump(by_group, row["group_name"], verdict, flag)

    def _with_pct(bucket: dict[str, int]) -> dict[str, float | int]:
        total = bucket["total"]
        return {
            **bucket,
            "correct_pct": round(100 * bucket["correct"] / total, 2) if total else 0.0,
        }

    return {
        "total_mappings": n_graded + n_skipped,
        "graded": n_graded,
        "correct": n_correct,
        "incorrect": n_incorrect,
        "self_inconsistent": n_inconsistent,
        "skipped": n_skipped,
        "skip_reasons": skip_reasons,
        "by_asset_class": {k: _with_pct(v) for k, v in sorted(by_class.items())},
        "by_group": {k: _with_pct(v) for k, v in sorted(by_group.items())},
    }


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", type=Path,
                   help="Grader sidecar JSONL paths; one .summary.json written per input")
    p.add_argument("--dry-run", action="store_true",
                   help="Print summaries to stdout without writing files")
    args = p.parse_args(argv)

    for sidecar in args.paths:
        if not sidecar.exists():
            print(f"SKIP missing: {sidecar}", file=sys.stderr)
            continue
        summary = aggregate(sidecar)
        out = sidecar.with_suffix(".summary.json")
        if args.dry_run:
            print(f"=== {sidecar} (would write -> {out}) ===")
            print(json.dumps(summary, indent=2))
        else:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(
                f"wrote {out}  "
                f"graded={summary['graded']} correct={summary['correct']} "
                f"incorrect={summary['incorrect']} "
                f"classes={len(summary['by_asset_class'])} "
                f"groups={len(summary['by_group'])}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
