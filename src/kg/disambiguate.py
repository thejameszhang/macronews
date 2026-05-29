"""Post-hoc entity disambiguation for the KG v2 sidecar JSONL.

Collapses surface-form variants of the same real-world entity into one
canonical name using sentence embeddings + type-constrained
complete-linkage clustering. Input and output share the same KG v2
schema (the runner's output format), so the visualizer can be applied
to either raw or disambiguated JSONL.

Algorithm:
  1. Iterate facts, count each unique (name, type) pair.
  2. Type-block: group entities by their entity_type.
  3. Embed every unique entity name once with a Sentence-BERT model
     (default Alibaba-NLP/gte-large-en-v1.5, CPU-friendly, 1024-dim).
  4. Within each type, build cosine-similarity matrix and complete-linkage
     cluster above the threshold (default 0.9). Cross-type merging is
     impossible by construction.
  5. Per cluster, pick the canonical name = most-frequent surface form,
     ties broken by length (prefer more specific names).
  6. Rewrite the JSONL with subject/object strings replaced by their
     cluster canonical.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# BAAI/bge-large-en-v1.5 over Alibaba-NLP/gte-large-en-v1.5: the latter
# ships custom modeling.py via trust_remote_code that hits a position-
# embedding IndexError with our pinned transformers 5.5.0 (which we hold
# for vLLM Gemma 4 compatibility). BGE-large uses standard transformer
# code, same 1024-dim output, comparable short-text quality on MTEB.
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_THRESHOLD = 0.9


def collect_entity_counts(rows: Iterable[dict]) -> dict[tuple[str, str], int]:
    """Count appearances of each unique (entity_name, entity_type) pair
    across all facts in all rows."""
    freq: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        for fact in row.get("facts", []):
            freq[(fact["subject"], fact["subject_type"])] += 1
            freq[(fact["object"], fact["object_type"])] += 1
    return dict(freq)


def cluster_within_type(
    names: list[str],
    embeddings: np.ndarray,
    threshold: float,
) -> list[list[str]]:
    """Complete-linkage cluster names by cosine similarity above `threshold`.

    Assumes embeddings are L2-normalized (unit length), so the inner
    product is cosine similarity. Returns a list of clusters; each cluster
    is a list of names. Single-element clusters represent un-merged entities.

    Complete-linkage means: a cluster contains a set of names where ALL
    pairwise similarities are >= threshold. This prevents transitive
    over-merging (A~B + B~C does NOT imply A,B,C cluster unless A~C too).
    """
    if len(names) == 0:
        return []
    if len(names) == 1:
        return [[names[0]]]

    sim = embeddings @ embeddings.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    # Clamp tiny negative floats from numerical error so squareform accepts.
    np.clip(dist, 0.0, None, out=dist)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="complete")
    cluster_ids = fcluster(Z, t=1.0 - threshold, criterion="distance")

    clusters: dict[int, list[str]] = defaultdict(list)
    for name, cid in zip(names, cluster_ids):
        clusters[cid].append(name)
    return list(clusters.values())


def select_canonical(cluster: list[str], freq: dict[str, int]) -> str:
    """Pick the canonical name from a cluster: most frequent surface form,
    with length as the tiebreaker (prefer longer / more specific)."""
    return max(cluster, key=lambda n: (freq[n], len(n)))


def _display(name: str) -> str:
    """Acronym-preserving title-case for the chosen canonical name.

    All-lowercase words get their first letter capitalized; words with any
    uppercase letter (acronyms like 'GDP', 'S&P', 'U.S.') are left alone.
    Mirrors build_graph._display intentionally — kept independent here so
    importing this module doesn't pull in build_graph's import-time
    logging.basicConfig side effect. The two are simple enough that drift
    risk is negligible.
    """
    return " ".join(
        w.capitalize() if w.islower() else w
        for w in name.split(" ")
    )


def disambiguate(
    input_jsonl: Path,
    output_jsonl: Path,
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    clusters_sidecar: Path | None = None,
) -> dict:
    """Run the disambiguation pipeline on a KG v2 sidecar JSONL.

    Writes `output_jsonl` with the same schema, `subject`/`object` strings
    rewritten to the cluster canonical for their (name, type) pair. If
    `clusters_sidecar` is given, also writes a JSON file mapping canonical
    name -> list of cluster members (for paper / audit).

    Returns: {total_entities, clusters_n, merges_n}.
    """
    rows = [
        json.loads(line)
        for line in Path(input_jsonl).read_text().splitlines()
        if line.strip()
    ]

    counts = collect_entity_counts(rows)

    by_type: dict[str, set[str]] = defaultdict(set)
    for (name, typ) in counts:
        by_type[typ].add(name)

    # 3. Embed every unique entity name ONCE, regardless of type.
    model = SentenceTransformer(model_name, device="cpu")
    all_names = sorted({name for name, _ in counts})
    logger.info(
        "Embedding %d unique entity names with %s",
        len(all_names), model_name,
    )
    embs = model.encode(
        all_names,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    embs = np.asarray(embs)
    emb_by_name = dict(zip(all_names, embs))

    # Cluster within each type and pick canonical per cluster.
    canonical_map: dict[tuple[str, str], str] = {}
    # Nested by type so identical canonicals in different types stay distinct.
    clusters_record: dict[str, dict[str, list[str]]] = defaultdict(dict)

    for typ, name_set in by_type.items():
        names_sorted = sorted(name_set)
        type_embs = np.array([emb_by_name[n] for n in names_sorted])
        clusters = cluster_within_type(names_sorted, type_embs, threshold)
        type_freq = {n: counts[(n, typ)] for n in names_sorted}
        for cluster in clusters:
            # Title-case the canonical so the stored name matches what the
            # graph renders (build_graph applies the same casing). Without
            # this, a lowercase surface form winning on frequency yields an
            # ugly canonical like "middle east conflict" in the JSONL.
            canonical = _display(select_canonical(cluster, type_freq))
            clusters_record[typ][canonical] = sorted(cluster)
            for member in cluster:
                canonical_map[(member, typ)] = canonical

    n_merges = sum(
        len(c) - 1
        for type_clusters in clusters_record.values()
        for c in type_clusters.values()
        if len(c) > 1
    )
    n_clusters = sum(len(t) for t in clusters_record.values())
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as out:
        for row in rows:
            new_facts = []
            for fact in row.get("facts", []):
                new_facts.append({
                    **fact,
                    "subject": canonical_map[(fact["subject"], fact["subject_type"])],
                    "object":  canonical_map[(fact["object"],  fact["object_type"])],
                })
            new_row = {**row, "facts": new_facts}
            out.write(json.dumps(new_row, ensure_ascii=False) + "\n")

    if clusters_sidecar is not None:
        clusters_sidecar = Path(clusters_sidecar)
        clusters_sidecar.parent.mkdir(parents=True, exist_ok=True)
        # Per-type, sort canonicals by cluster size descending.
        sorted_sidecar = {
            typ: dict(sorted(
                type_clusters.items(),
                key=lambda kv: (-len(kv[1]), kv[0]),
            ))
            for typ, type_clusters in sorted(clusters_record.items())
        }
        clusters_sidecar.write_text(
            json.dumps(sorted_sidecar, indent=2, ensure_ascii=False)
        )

    summary = {
        "total_entities": len(counts),
        "clusters_n": n_clusters,
        "merges_n": n_merges,
    }
    logger.info(
        "Disambiguated %d unique (name,type) pairs -> %d clusters (%d merges)",
        summary["total_entities"], summary["clusters_n"], summary["merges_n"],
    )
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path,
                   help="KG v2 sidecar JSONL to disambiguate")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSONL path (same schema as input)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help=f"Sentence-Transformers model (default: {DEFAULT_MODEL})")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"Cosine similarity merge threshold (default: {DEFAULT_THRESHOLD})")
    p.add_argument("--clusters", type=Path, default=None,
                   help="Optional path to write {canonical: [variants...]} JSON sidecar")
    args = p.parse_args()

    summary = disambiguate(
        input_jsonl=args.input,
        output_jsonl=args.output,
        model_name=args.model,
        threshold=args.threshold,
        clusters_sidecar=args.clusters,
    )
    print(f"Done: {summary}")


if __name__ == "__main__":
    main()
