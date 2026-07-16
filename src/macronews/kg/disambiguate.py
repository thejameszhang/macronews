"""Post-hoc entity disambiguation for the KG v2 sidecar JSONL.

Collapses surface-form variants of the same real-world entity into one canonical
name: type-block, embed each unique name once (Sentence-BERT, default
BAAI/bge-large-en-v1.5), then cluster within each type by cosine similarity via the
FAISS-backed `kg.clustering.cluster_by_cosine` (connected components then
complete-linkage at threshold=0.9). Cross-type merging is impossible by construction.
Canonical = most-frequent surface form, ties broken by length. Output keeps the same
KG v2 schema, so the visualizer works on either raw or disambiguated JSONL.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from macronews.config.paths import EMBED_MODEL
from macronews.kg.clustering import cluster_by_cosine
from macronews.kg.self_reference import filter_event, SELF_REF_COSINE_THRESHOLD

logger = logging.getLogger(__name__)

# BAAI/bge-large-en-v1.5 over Alibaba-NLP/gte-large-en-v1.5: the latter
# ships custom modeling.py via trust_remote_code that hits a position-
# embedding IndexError with our pinned transformers 5.5.0 (which we hold
# for vLLM Gemma 4 compatibility). BGE-large uses standard transformer
# code, same 1024-dim output, comparable short-text quality on MTEB.
DEFAULT_THRESHOLD = 0.9


def _default_self_ref_rejected(output_jsonl):
    p = Path(output_jsonl)
    # strip a trailing .jsonl, append the audit suffix
    stem = p.name[:-6] if p.name.endswith(".jsonl") else p.stem
    return p.with_name(f"{stem}.self_ref_rejected.jsonl")


def collect_entity_counts(rows: Iterable[dict]) -> dict[tuple[str, str], int]:
    """Count appearances of each unique (entity_name, entity_type) pair
    across all event triplets in all rows."""
    freq: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        for ev in row.get("events", []):
            for t in ev.get("triplets", []):
                freq[(t["subject"], t["subject_type"])] += 1
                freq[(t["object"], t["object_type"])] += 1
    return dict(freq)


def _norm_key(name: str) -> str:
    """Rule A normalization: lowercase, hyphens -> spaces, collapse whitespace.

    Names equal under this (e.g. 'Natural-gas' == 'Natural Gas') are pre-merged
    BEFORE embedding clustering, so the cosine threshold can't fragment trivial
    punctuation/case variants (BGE puts 'Natural-gas' at 0.866 from 'Natural Gas',
    below the 0.90 cutoff). Hyphen -> SPACE (never removal), so 'X-Y' merges with
    'X Y' but NEVER with 'XY' — it can't fuse unrelated tokens.
    """
    return re.sub(r"\s+", " ", name.lower().replace("-", " ")).strip()


def cluster_within_type(
    names: list[str],
    embeddings: np.ndarray,
    threshold: float,
) -> list[list[str]]:
    """Complete-linkage cluster names by cosine similarity above `threshold`.

    Assumes embeddings are L2-normalized (unit length), so the inner
    product is cosine similarity. Returns a list of clusters; each cluster
    is a list of names. Single-element clusters represent un-merged entities.

    Names sharing a Rule A normalized key (see `_norm_key`) are pre-collapsed
    into one unit FIRST (guaranteed merged); the units are then embed-clustered.
    Complete-linkage means: a cluster contains a set of units where ALL pairwise
    similarities are >= threshold. This prevents transitive over-merging (A~B +
    B~C does NOT imply A,B,C cluster unless A~C too); pre-collapsing units adds
    no chaining, since the embedding clustering still runs over distinct keys.
    """
    if len(names) == 0:
        return []

    # Rule A pre-collapse: group names by normalized key; cluster one
    # representative embedding per key, then expand each cluster to all members.
    units: dict[str, list[str]] = defaultdict(list)
    rep: dict[str, int] = {}
    for i, name in enumerate(names):
        key = _norm_key(name)
        if key not in rep:
            rep[key] = i
        units[key].append(name)
    keys = list(units)
    if len(keys) == 1:
        return [list(names)]

    rep_embs = np.array([embeddings[rep[k]] for k in keys])
    # Shared-module complete-linkage, equivalent to the old global scipy version.
    # Block (FAISS) at a slightly looser floor than `threshold`, then enforce
    # complete-linkage at `threshold`. The 0.05 gap matters: FAISS range_search on
    # IndexFlatIP returns inner product STRICTLY > the radius, so blocking exactly at
    # `threshold` could drop a pair sitting at cosine == threshold — which within-block
    # complete-linkage (inclusive, cutoff 1-threshold) WOULD merge. Any
    # search_threshold < threshold is correct; 0.05 is conservative headroom.
    key_clusters = cluster_by_cosine(
        keys, rep_embs,
        search_threshold=max(0.0, threshold - 0.05),
        complete_linkage=True,
        enforce_threshold=threshold,
    )
    # Expand representative-key clusters back to all member names.
    return [[name for key in kc for name in units[key]] for kc in key_clusters]


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
    model_name: str = EMBED_MODEL,
    threshold: float = DEFAULT_THRESHOLD,
    clusters_sidecar: Path | None = None,
    self_ref_threshold: float | None = None,
    self_ref_rejected_jsonl: Path | None = None,
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
    # Use the GPU when one is present (a B200 cuts the encode from ~30 min to
    # ~1-2 min); fall back to CPU on login/interactive nodes and in tests.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    all_names = sorted({name for name, _ in counts})
    logger.info(
        "Embedding %d unique entity names with %s on %s",
        len(all_names), model_name, device,
    )
    embs = model.encode(
        all_names,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=256 if device == "cuda" else 64,
    )
    embs = np.asarray(embs)
    emb_by_name = dict(zip(all_names, embs))

    # Cluster within each type and pick canonical per cluster.
    canonical_map: dict[tuple[str, str], str] = {}
    canonical_emb: dict[str, np.ndarray] = {}
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
            raw_canon = select_canonical(cluster, type_freq)
            canonical = _display(raw_canon)
            canonical_emb[canonical] = emb_by_name[raw_canon]
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
    self_ref_rejected = []
    with output_jsonl.open("w", encoding="utf-8") as out:
        for row in rows:
            new_events = []
            for ev in row.get("events", []):
                new_trips = [{**t,
                              "subject": canonical_map[(t["subject"], t["subject_type"])],
                              "object":  canonical_map[(t["object"],  t["object_type"])]}
                             for t in ev.get("triplets", [])]
                ev_canon = {**ev, "triplets": new_trips}
                if self_ref_threshold is not None:
                    kept, rej = filter_event(ev_canon, canonical_emb, self_ref_threshold)
                    ev_canon = {**ev_canon, "triplets": kept}
                    self_ref_rejected.extend(rej)
                new_events.append(ev_canon)
            out.write(json.dumps({**row, "events": new_events}, ensure_ascii=False) + "\n")

    if self_ref_rejected_jsonl is not None and self_ref_threshold is not None:
        rej_path = Path(self_ref_rejected_jsonl)
        rej_path.parent.mkdir(parents=True, exist_ok=True)
        with rej_path.open("w", encoding="utf-8") as rf:
            for r in self_ref_rejected:
                rf.write(json.dumps(r, ensure_ascii=False) + "\n")

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
        "self_ref_dropped": len(self_ref_rejected),
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
    p.add_argument("--model", type=str, default=EMBED_MODEL,
                   help=f"Sentence-Transformers model (default: {EMBED_MODEL})")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"Cosine similarity merge threshold (default: {DEFAULT_THRESHOLD})")
    p.add_argument("--self-ref-threshold", type=float, default=SELF_REF_COSINE_THRESHOLD,
                   help=f"self-reference cosine threshold (default {SELF_REF_COSINE_THRESHOLD}); "
                        f"set > 1.0 to disable Tier-2")
    p.add_argument("--self-ref-rejected", type=str, default=None,
                   help="self-reference rejected-triplet audit log (default: sibling of --output)")
    p.add_argument("--clusters", type=Path, default=None,
                   help="Optional path to write {canonical: [variants...]} JSON sidecar")
    args = p.parse_args()

    summary = disambiguate(
        input_jsonl=args.input,
        output_jsonl=args.output,
        model_name=args.model,
        threshold=args.threshold,
        clusters_sidecar=args.clusters,
        self_ref_threshold=args.self_ref_threshold,
        self_ref_rejected_jsonl=(args.self_ref_rejected
                                 or _default_self_ref_rejected(args.output)),
    )
    print(f"Done: {summary}")


if __name__ == "__main__":
    main()
