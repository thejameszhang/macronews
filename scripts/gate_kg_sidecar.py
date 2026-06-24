"""Ex-post 0-mapper gate for a KG event sidecar.

Keeps only the articles the mapper tagged with >=1 asset group (relevance_score
> 0.5), dropping every article the mapper left untagged. This SIMULATES, after
the fact, what a gated production extraction would have produced (the gate
normally runs at extraction time in runner.py via --gate-zero-mappings). The
mapper output is unchanged and the input sidecar is left intact — we only write
a filtered copy.

    PYTHONPATH=src .venv/bin/python scripts/gate_kg_sidecar.py \
        results/kg/temporal/2014-05c-full.jsonl \
        --mapper results/mapper/prod/v1/2014-05c.jsonl \
        --output results/kg/gamma/2014-05c-full.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kg.link_groups import build_mapper_index  # noqa: E402


def gate_sidecar(kg_path: Path, mapper_path: Path, out_path: Path) -> tuple[int, int]:
    """Write `out_path` keeping only KG rows whose article the mapper tagged
    (>0.5). Returns (kept, total) article counts."""
    tagged = set(build_mapper_index(mapper_path))   # article_ids with a >0.5 tag
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = total = 0
    with open(kg_path) as f, open(out_path, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            total += 1
            if json.loads(line)["article_id"] in tagged:
                w.write(line)
                kept += 1
    return kept, total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kg", type=Path, help="KG event sidecar JSONL (one row per article)")
    p.add_argument("--mapper", type=Path, required=True, help="mapper sidecar JSONL")
    p.add_argument("--output", type=Path, required=True, help="gated sidecar out")
    args = p.parse_args()
    kept, total = gate_sidecar(args.kg, args.mapper, args.output)
    pct = 100 * kept / total if total else 0.0
    print(f"{args.kg.name}: kept {kept}/{total} articles ({pct:.1f}%) -> {args.output}")


if __name__ == "__main__":
    main()
