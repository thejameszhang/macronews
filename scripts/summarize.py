"""Roll up per-shard *.summary.json sidecars into a combined aggregate.

Usage:
    python scripts/summarize.py 'results/prod/run-1/2015-06*.summary.json'
    python scripts/summarize.py 'results/prod/run-1/2015-*.summary.json'
    python scripts/summarize.py 'results/prod/run-1/*.summary.json'
    python scripts/summarize.py results/imperative/djnw_march_2022.summary.json

Pass one or more glob patterns or paths. Globs are quoted to prevent the shell
from expanding them. Prints a JSON summary to stdout; pass --output to write
to a file.
"""

import argparse
import glob
import json
import sys
from pathlib import Path


def merge(into: dict, other: dict) -> None:
    """Sum scalars, dict-merge sum nested dicts."""
    for k, v in other.items():
        if isinstance(v, dict):
            sub = into.setdefault(k, {})
            for sk, sv in v.items():
                sub[sk] = sub.get(sk, 0) + sv
        elif isinstance(v, (int, float)):
            into[k] = into.get(k, 0) + v
        else:
            into[k] = v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="+", help="Glob patterns or file paths")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON to this path instead of stdout")
    args = parser.parse_args()

    paths: list[Path] = []
    for p in args.patterns:
        matched = sorted(glob.glob(p))
        if not matched:
            print(f"WARN: no files match {p!r}", file=sys.stderr)
            continue
        paths.extend(Path(m) for m in matched)

    if not paths:
        print("ERROR: no files matched any pattern", file=sys.stderr)
        sys.exit(1)

    combined: dict = {}
    for path in paths:
        with open(path) as f:
            shard_agg = json.load(f)
        merge(combined, shard_agg)
    combined["n_shards"] = len(paths)

    out_text = json.dumps(combined, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(out_text + "\n")
        print(f"Wrote {args.output} ({len(paths)} shards)", file=sys.stderr)
    else:
        print(out_text)


if __name__ == "__main__":
    main()
