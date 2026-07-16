import numpy as np
from macronews.kg.clustering import cluster_by_cosine


def _unit(rows):
    a = np.asarray(rows, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _as_sets(clusters):
    return sorted([frozenset(c) for c in clusters], key=lambda s: sorted(s)[0])


def test_empty_and_singleton():
    assert cluster_by_cosine([], np.zeros((0, 3), np.float32), 0.8) == []
    assert cluster_by_cosine(["a"], _unit([[1, 0, 0]]), 0.8) == [["a"]]


def test_connected_components_groups_near_separates_far():
    embs = _unit([[1, 0, 0], [0.99, 0.14, 0], [0, 0, 1]])
    out = _as_sets(cluster_by_cosine(["a", "b", "c"], embs, search_threshold=0.85))
    assert out == [frozenset({"a", "b"}), frozenset({"c"})]


def test_complete_linkage_blocks_chaining():
    # Coplanar unit vectors: cos(a,b)=0.97, cos(b,c)=0.953, cos(a,c)=0.85.
    # single-linkage at 0.80 connects all three; complete-linkage at 0.90 -> {a,b},{c}.
    embs = _unit([[1.0, 0.0, 0.0],
                  [0.97, 0.2431, 0.0],
                  [0.85, 0.5268, 0.0]])
    cc = _as_sets(cluster_by_cosine(["a", "b", "c"], embs, search_threshold=0.80))
    assert cc == [frozenset({"a", "b", "c"})]
    cl = _as_sets(cluster_by_cosine(
        ["a", "b", "c"], embs, search_threshold=0.80,
        complete_linkage=True, enforce_threshold=0.90))
    assert cl == [frozenset({"a", "b"}), frozenset({"c"})]


def test_invariant_rejects_search_above_enforce():
    import pytest
    embs = _unit([[1, 0], [1, 0]])
    with pytest.raises(ValueError):
        cluster_by_cosine(["a", "b"], embs, 0.95,
                          complete_linkage=True, enforce_threshold=0.90)


def test_faiss_matches_bruteforce_components():
    rng = np.random.default_rng(0)
    embs = _unit(rng.standard_normal((40, 16)))
    tau = 0.5
    got = _as_sets(cluster_by_cosine(list(range(40)), embs, search_threshold=tau))
    sim = embs @ embs.T
    parent = list(range(40))
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for i in range(40):
        for j in range(i + 1, 40):
            if sim[i, j] >= tau:
                parent[find(i)] = find(j)
    ref = {}
    for i in range(40):
        ref.setdefault(find(i), []).append(i)
    ref_sets = sorted([frozenset(v) for v in ref.values()], key=lambda s: sorted(s)[0])
    assert got == ref_sets


def test_two_member_complete_linkage_block():
    # Smallest non-trivial complete-linkage block (scipy condensed matrix size 1).
    embs = _unit([[1.0, 0.0], [0.99, 0.141]])   # cos ~0.99
    out = _as_sets(cluster_by_cosine(
        ["a", "b"], embs, search_threshold=0.80,
        complete_linkage=True, enforce_threshold=0.90))
    assert out == [frozenset({"a", "b"})]


def test_unnormalized_input_is_normalized_internally():
    # Same directions as the connected-components test but NON-unit lengths.
    # Internal L2-normalization must make this cluster identically.
    raw = np.asarray([[5.0, 0.0, 0.0],          # |.|=5
                      [4.95, 0.70, 0.0],         # ~same direction, |.|~5
                      [0.0, 0.0, 0.3]],          # orthogonal, |.|=0.3
                     dtype=np.float32)
    out = _as_sets(cluster_by_cosine(["a", "b", "c"], raw, search_threshold=0.85))
    assert out == [frozenset({"a", "b"}), frozenset({"c"})]
