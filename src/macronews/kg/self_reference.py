"""Self-reference filter primitives for the macro KG.

Tier 1 (exact self-loop) drops a triplet whose subject and object are the same entity, in
ANY relation. Tier 2 (cosine) drops a directional triplet whose subject and object are the
same asset (embedding cosine >= threshold). The type-signature gate cannot catch these
because they are type-legal. See
docs/superpowers/specs/2026-06-24-kg-self-reference-filter-design.md.
"""
import numpy as np

from macronews.kg.type_signatures import DIRECTIONAL_RELATIONS

SELF_REF_COSINE_THRESHOLD = 0.85


def _norm(s):
    return " ".join((s or "").split()).casefold()


def is_self_loop(subject, object_):
    """True if subject and object are the same entity (case/whitespace-folded, non-empty)."""
    n = _norm(subject)
    return bool(n) and n == _norm(object_)


def is_same_asset(subj_emb, obj_emb, threshold):
    """True if cosine(subj_emb, obj_emb) >= threshold. Embeddings must be L2-normalized
    (so cosine == dot product)."""
    return float(np.dot(subj_emb, obj_emb)) >= threshold


def filter_event(event, canonical_emb, threshold):
    """Split one event's triplets into (kept, rejected_rows). Tier-1 exact self-loop (all
    relations); Tier-2 cosine same-asset (directional relations only). Non-mutating.
    Raises RuntimeError if a directional triplet's canonical has no embedding."""
    aid, eid, stmt = event.get("article_id"), event.get("id"), event.get("statement")
    kept, rejected = [], []
    for t in (event.get("triplets") or []):
        subj, obj = t.get("subject"), t.get("object")
        rel = (t.get("relation") or "").upper()
        reason, cosine = None, None
        if is_self_loop(subj, obj):
            reason = "self_loop"
        elif rel in DIRECTIONAL_RELATIONS:
            try:
                se, oe = canonical_emb[subj], canonical_emb[obj]
            except KeyError as e:
                raise RuntimeError(
                    f"self-reference filter: no embedding for canonical {e.args[0]!r}")
            c = float(np.dot(se, oe))
            if c >= threshold:
                reason, cosine = "same_asset_cosine", c
        if reason is None:
            kept.append(t)
        else:
            rejected.append({
                "article_id": aid, "event_id": eid, "statement": stmt,
                "subject": subj, "subject_type": t.get("subject_type"),
                "relation": t.get("relation"),
                "object": obj, "object_type": t.get("object_type"),
                "value": t.get("value"),
                "self_ref_reason": reason, "cosine": cosine,
            })
    return kept, rejected
