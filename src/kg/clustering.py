"""Scalable cosine-similarity clustering, FAISS-backed.

Shared by disambiguate.py (precision: complete_linkage=True over entity-name
representatives, anti-chaining) and invalidate_llm.py (recall: connected
components over statement embeddings). Two-stage: an IndexFlatIP range_search
blocks near-neighbors at a cosine floor, connected components form coarse
clusters, and optional within-block complete-linkage enforces a stricter floor.
Avoids the O(N^2) dense similarity matrix.
"""
from __future__ import annotations

import numpy as np


def cluster_by_cosine(
    ids: list,
    embeddings: np.ndarray,
    search_threshold: float,
    *,
    complete_linkage: bool = False,
    enforce_threshold: float | None = None,
) -> list[list]:
    """Cluster `ids` by cosine similarity of their `embeddings`.

    Embeddings are copied and L2-normalized internally, so inner product == cosine
    regardless of the input scale; the caller's array is never mutated.
    search_threshold: cosine FLOOR for FAISS blocking (larger = stricter). NOT an
        L2 radius — we use IndexFlatIP.
    complete_linkage=False: each connected component is a cluster (recall).
    complete_linkage=True: within each component, run complete-linkage at
        enforce_threshold so every pair in a returned cluster has cosine >=
        enforce_threshold (precision / anti-chaining). If enforce_threshold is
        None it defaults to search_threshold, which makes the pass a no-op
        (equivalent to complete_linkage=False) — pass a stricter value to split.
    Invariant: search_threshold <= enforce_threshold (when the latter is given).
    """
    n = len(ids)
    if n == 0:
        return []
    if n == 1:
        return [list(ids)]
    if enforce_threshold is not None and search_threshold > enforce_threshold:
        raise ValueError("search_threshold must be <= enforce_threshold")

    import faiss

    # Copy (never mutate the caller's array) + L2-normalize so inner product ==
    # cosine. Idempotent on already-unit rows; makes the cosine contract hold for
    # any caller, not just ones that normalized upstream.
    embs = np.array(embeddings, dtype=np.float32)
    embs = np.ascontiguousarray(embs)
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    lims, _dist, idx = index.range_search(embs, float(search_threshold))

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in idx[lims[i]:lims[i + 1]]:
            j = int(j)
            if j != i:
                union(i, j)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    if not complete_linkage:
        return [[ids[i] for i in members] for members in comps.values()]

    thr = enforce_threshold if enforce_threshold is not None else search_threshold
    out: list[list] = []
    for members in comps.values():
        out.extend(_complete_linkage_block(members, embs, thr, ids))
    return out


def _complete_linkage_block(members, embs, threshold, ids):
    """Complete-linkage within one block; mirrors disambiguate's current logic
    (distance cutoff = 1 - threshold, method='complete')."""
    if len(members) == 1:
        return [[ids[members[0]]]]
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    sub = embs[members]
    dist = 1.0 - (sub @ sub.T)
    np.fill_diagonal(dist, 0.0)
    np.clip(dist, 0.0, None, out=dist)
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="complete")
    cluster_ids = fcluster(z, t=1.0 - threshold, criterion="distance")
    groups: dict[int, list] = {}
    for local_idx, cid in enumerate(cluster_ids):
        groups.setdefault(int(cid), []).append(ids[members[local_idx]])
    return list(groups.values())
