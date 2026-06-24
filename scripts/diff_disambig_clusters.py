"""Diff two disambiguate cluster sidecars ({type: {canonical: [variants...]}}).

Reports clusters present in one side but not the other, per entity type. Exit 0
always — this is a human-read no-regression report, not a CI gate.

    PYTHONPATH=src python scripts/diff_disambig_clusters.py OLD.json NEW.json
"""
import json
import sys
from pathlib import Path


def load(p):
    d = json.loads(Path(p).read_text())
    return {typ: {k: frozenset(v) for k, v in canon_map.items()}
            for typ, canon_map in d.items()}


def main():
    old = load(sys.argv[1])
    new = load(sys.argv[2])
    types = sorted(set(old) | set(new))
    n_diff = 0
    n_old_clusters = sum(len(v) for v in old.values())
    n_new_clusters = sum(len(v) for v in new.values())
    for typ in types:
        o_sets = set(old.get(typ, {}).values())
        n_sets = set(new.get(typ, {}).values())
        only_old = o_sets - n_sets
        only_new = n_sets - o_sets
        if only_old or only_new:
            print(f"\n=== {typ} ===")
            for s in sorted(only_old, key=lambda x: sorted(x)[0]):
                print(f"  OLD-only: {sorted(s)}")
            for s in sorted(only_new, key=lambda x: sorted(x)[0]):
                print(f"  NEW-only: {sorted(s)}")
            n_diff += len(only_old) + len(only_new)
    print(f"\nOLD clusters: {n_old_clusters} | NEW clusters: {n_new_clusters}")
    print(f"Total differing clusters: {n_diff}")


if __name__ == "__main__":
    main()
