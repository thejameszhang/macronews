"""LLM Invalidation Agent — cookbook-faithful supersession over statement events.

Reads a disambiguated event sidecar, embeds each event's statement, and uses a
per-event funnel (entity-share → temporal-overlap → embedding top-K) to select
candidate (primary, secondary) pairs for LLM judgement. Asks a local LLM whether
the earlier eligible primary event in a chronological pair is superseded by the
later secondary. Non-lossy: only invalid_at / invalidated_by / expired_at /
invalidation_method ever change.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import compat  # noqa: F401
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

TOP_K = 10       # embedding top-K per primary
SIM_FLOOR = 0.5  # minimum cosine similarity to qualify as a candidate


class InvalidationVerdict(BaseModel):
    """Cookbook True/False — is the primary invalidated by the secondary?"""
    invalidated: bool = False


INVALIDATION_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "invalidation.txt"
_PROMPT_TEMPLATE = INVALIDATION_PROMPT_PATH.read_text()


def _triplet_line(ev: dict) -> str:
    """Cookbook `{% if primary_triplet %}Triplet: ...{% endif %}` — emit a 'Triplet: ' line ONLY
    when the event has triplets, else nothing (no bare label). Each triplet is the cookbook's
    (subject, relation, object); multiples joined by '; ' (our event is statement-grained, so it
    can carry several triplets, unlike the cookbook's one)."""
    trips = "; ".join(f"({t.get('subject')}, {t.get('relation')}, {t.get('object')})"
                      for t in (ev.get("triplets") or []))
    return f"Triplet: {trips}\n" if trips else ""


def render_pair_prompt(primary: dict, secondary: dict) -> str:
    return _PROMPT_TEMPLATE.format(
        primary_statement=primary["statement"],
        primary_triplet_line=_triplet_line(primary),
        primary_valid_at=primary.get("valid_at"),
        primary_invalid_at=primary.get("invalid_at"),
        secondary_statement=secondary["statement"],
        secondary_triplet_line=_triplet_line(secondary),
        secondary_valid_at=secondary.get("valid_at"),
        secondary_invalid_at=secondary.get("invalid_at"),
    )

# Eligible: has a temporal position, is not ATEMPORAL, and is FACT or PREDICTION.
# OPINION never participates in invalidation.
ELIGIBLE_TYPES = {"FACT", "PREDICTION"}


def load_asset_linked_keys(path: Path | str) -> set[str]:
    """Read the ex-post linker's entity_groups.json (canonical casefold entity name ->
    {type, groups}) and return the set of asset-linked entity keys. Lets us scope
    invalidation to facts that touch one of the ~50 asset groups (see link_groups.py)."""
    data = json.loads(Path(path).read_text())
    return {k.casefold() for k in data}


def _event_is_asset_linked(ev: dict, asset_keys: set[str]) -> bool:
    """True if any of the event's triplet entities (subject/object) is in an asset group."""
    return any(_ent_key(n) in asset_keys
               for t in ev.get("triplets", [])
               for n in (t.get("subject"), t.get("object")) if n)


def load_eligible_events(disambig_path: Path | str, asset_keys: set[str] | None = None):
    """Return (eligible_events, all_rows). Eligible = valid_at set AND not ATEMPORAL
    AND statement_type in FACT/PREDICTION; when `asset_keys` is given (Option B scoping),
    ALSO require >=1 triplet entity to be asset-group-linked. all_rows is every source row,
    kept verbatim for non-lossy rewrite (non-asset events are still written, just never judged)."""
    rows = []
    eligible = []
    for line in Path(disambig_path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        for ev in row.get("events", []):
            if (ev.get("valid_at") is not None
                    and ev.get("temporal_type") != "ATEMPORAL"
                    and ev.get("statement_type") in ELIGIBLE_TYPES
                    and (asset_keys is None or _event_is_asset_linked(ev, asset_keys))):
                eligible.append(ev)
    return eligible, rows


def save_embeddings(path: Path | str, ids: list[str], vecs: np.ndarray) -> None:
    """One named array per event id (UUID string) — keyed lookup at load."""
    p = str(path)
    if not p.endswith(".npz"):
        p += ".npz"   # np.savez auto-appends .npz; match it so load_embeddings finds it
    np.savez(p, **{str(i): v for i, v in zip(ids, vecs)})


def load_embeddings(path: Path | str) -> dict:
    p = str(path)
    if not p.endswith(".npz"):
        p += ".npz"   # mirror save_embeddings' .npz normalization
    data = np.load(p)
    return {k: data[k] for k in data.files}


def embed_statements(events: list[dict], model_name: str = "BAAI/bge-large-en-v1.5"):
    """Encode each event['statement'] -> (ids, L2-normalized vectors)."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    ids = [e["id"] for e in events]
    vecs = model.encode([e["statement"] for e in events],
                        normalize_embeddings=True, show_progress_bar=True,
                        batch_size=256 if device == "cuda" else 64)
    return ids, np.asarray(vecs, dtype=np.float32)


def _dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"Naive datetime not allowed (expected UTC '...Z'): {s!r}")
    return dt


def temporal_bounds(ev: dict):
    """Return (start, end) datetimes for the event's temporal window, or None if ATEMPORAL
    or missing valid_at. For STATIC (or DYNAMIC without invalid_at): start == end."""
    if ev.get("temporal_type") == "ATEMPORAL" or ev.get("valid_at") is None:
        return None
    start = _dt(ev["valid_at"]); end = start
    if ev["temporal_type"] == "DYNAMIC" and ev.get("invalid_at"):
        end = _dt(ev["invalid_at"])
    return start, end


def _overlaps_dynamic(cand: dict, start, end) -> bool:
    """True if a DYNAMIC candidate's interval overlaps [start, end]."""
    if cand.get("temporal_type") != "DYNAMIC":
        return False
    if not cand.get("valid_at"):
        return False
    es = _dt(cand["valid_at"]); ee = _dt(cand["invalid_at"]) if cand.get("invalid_at") else None
    if ee is not None and es <= start <= ee:
        return True
    if ee is None and es <= start:
        return True
    if start <= es <= end:
        return True
    return False


def select_temporally_relevant(reference: dict, candidates: list[dict]) -> list[dict]:
    """Cookbook temporal filter, from `reference` (the secondary/invalidator) event's
    window. A DYNAMIC candidate qualifies when its interval overlaps; a STATIC candidate
    when its point falls at or before the window end. The cookbook's STATIC test is
    `start <= valid_at <= end`, but our deviation makes any earlier PREDICTION an
    invalidatable primary — so we drop the lower bound and let build_pairs' strict-chronology
    guard (`valid_at < secondary.start`) do the "earlier than" enforcement. STATIC FACTs that
    slip through here are still excluded downstream by is_primary_eligible."""
    b = temporal_bounds(reference)
    if b is None:
        return []
    start, end = b
    statics = [c for c in candidates if c.get("temporal_type") == "STATIC"
               and c.get("valid_at") and _dt(c["valid_at"]) <= end]
    dynamics = [c for c in candidates if _overlaps_dynamic(c, start, end)]
    return statics + dynamics


def _ent_key(s: str) -> str:
    return s.casefold().strip()


def build_entity_index(events: list[dict]) -> dict:
    """Map canonical entity name → {event_id: event} for fast entity-share lookup."""
    idx = defaultdict(dict)
    for ev in events:
        for t in ev.get("triplets", []):
            for name in (t.get("subject"), t.get("object")):
                if name:
                    idx[_ent_key(name)][ev["id"]] = ev
    return idx


def gather_entity_shared(primary: dict, entity_index: dict) -> list[dict]:
    """Return all events (excluding primary itself) that share at least one entity."""
    out = {}
    names = {_ent_key(n) for t in primary.get("triplets", [])
             for n in (t.get("subject"), t.get("object")) if n}
    for name in names:
        for eid, ev in entity_index.get(name, {}).items():
            if eid != primary["id"]:
                out[eid] = ev
    return list(out.values())


def filter_by_embedding_similarity(reference, candidates, emb_by_id,
                                   top_k: int = TOP_K, floor: float = SIM_FLOOR) -> list[dict]:
    """Keep candidates with cosine >= floor to `reference`; sort desc; return top_k.
    Vectorized: one `ref @ M.T` matmul over all candidates (a hot entity yields
    thousands of candidates, so the per-candidate np.dot loop was the bottleneck)."""
    ref = emb_by_id[reference["id"]]
    cands = [c for c in candidates if c["id"] in emb_by_id]
    if not cands:
        return []
    mat = np.stack([emb_by_id[c["id"]] for c in cands])   # (n, d)
    sims = mat @ ref                                       # (n,)  == ref @ mat.T
    keep = np.nonzero(sims >= floor)[0]
    order = keep[np.argsort(sims[keep])[::-1]]             # cosine desc
    return [cands[i] for i in order[:top_k]]


def type_pair_allowed(primary: dict, secondary: dict) -> bool:
    """A FACT is invalidated only by a later FACT; a PREDICTION by a later FACT
    (resolved) or PREDICTION (revised). OPINION never participates."""
    p, s = primary.get("statement_type"), secondary.get("statement_type")
    if p == "FACT":
        return s == "FACT"
    if p == "PREDICTION":
        return s in ("FACT", "PREDICTION")
    return False


def is_primary_eligible(ev: dict) -> bool:
    """Can be invalidated: a DYNAMIC fact, or any prediction (predictions are
    inherently provisional). Must have a temporal position."""
    if ev.get("valid_at") is None or ev.get("temporal_type") == "ATEMPORAL":
        return False
    st = ev.get("statement_type")
    return (st == "FACT" and ev.get("temporal_type") == "DYNAMIC") or st == "PREDICTION"


def build_pairs(events: list[dict], emb_by_id: dict) -> list[tuple]:
    """Cookbook candidate selection. Iterate each event as the SECONDARY (invalidator);
    using the secondary's temporal window, gather entity-sharing candidate PRIMARIES that
    are temporally relevant (an earlier ongoing/overlapping event is found via
    _overlaps_dynamic condition 2), embedding top-K, then keep only EARLIER eligible
    primaries that satisfy the FACT/PREDICTION type rule. Returns (primary_id, secondary_id)."""
    entity_index = build_entity_index(events)
    pairs = []
    for secondary in events:
        sb = temporal_bounds(secondary)
        if sb is None:
            continue
        sstart, _ = sb
        cands = gather_entity_shared(secondary, entity_index)
        cands = select_temporally_relevant(secondary, cands)
        cands = filter_by_embedding_similarity(secondary, cands, emb_by_id)
        for primary in cands:
            if (is_primary_eligible(primary)
                    and _dt(primary["valid_at"]) < sstart
                    and type_pair_allowed(primary, secondary)):
                pairs.append((primary["id"], secondary["id"]))
    return pairs


def dedup_representatives(events: list[dict]):
    """Collapse duplicate events (DJNW wire reruns + same-fact paraphrases) to one representative
    for candidate selection. Two events merge IFF they share the SAME SET of (subject, relation,
    object) triplets AND the same valid_at; an event with NO triplets falls back to
    (statement, valid_at) so distinct triplet-less facts don't merge on date alone. Returns
    (reps, members_by_rep): reps = deduped list (first occurrence kept); members_by_rep[rep_id] =
    [every event id in that group, rep included]. Non-lossy downstream — all original events are
    written; the rep's verdict fans out to its whole group via propagate_updates."""
    def _k(ev):
        trips = ev.get("triplets") or []
        if not trips:
            return ("stmt", ev.get("statement"), ev.get("valid_at"))
        return ("trip", frozenset((t.get("subject"), t.get("relation"), t.get("object")) for t in trips),
                ev.get("valid_at"))
    rep_id_by_key = {}
    members = defaultdict(list)
    reps = []
    for ev in events:
        k = _k(ev)
        if k not in rep_id_by_key:
            rep_id_by_key[k] = ev["id"]
            reps.append(ev)
        members[rep_id_by_key[k]].append(ev["id"])
    return reps, dict(members)


def propagate_updates(updates: dict, members: dict) -> dict:
    """Fan each representative's invalidation update out to every member of its duplicate group,
    so all wire-rerun copies of a closed fact carry the same invalid_at/invalidated_by. Each copy
    gets its own dict (no shared mutable reference). Every rep is a key in `members` by construction."""
    return {mid: dict(upd) for rep_id, upd in updates.items() for mid in members[rep_id]}


def apply_verdicts(events_by_id, pairs, verdicts) -> None:
    """Mutate primaries in place: refine-earlier-only + resolve-duplicates by earliest
    invalid_at (cookbook). Non-lossy elsewhere."""
    invalidators = defaultdict(list)
    for (pid, sid), v in zip(pairs, verdicts):
        if v:
            invalidators[pid].append(sid)
    for pid, sids in invalidators.items():
        primary = events_by_id[pid]
        best = min(sids, key=lambda s: _dt(events_by_id[s]["valid_at"]))
        new_invalid = events_by_id[best]["valid_at"]
        cur = primary.get("invalid_at")
        if cur is None or _dt(new_invalid) < _dt(cur):
            primary["invalid_at"] = new_invalid
            primary["invalidated_by"] = best
            primary["expired_at"] = primary.get("created_at")
            primary["invalidation_method"] = "llm"


class InvalidationAgent:
    """Cookbook pairwise supersession judge on local vLLM (QwQ, batch-invariant)."""

    def __init__(self, model_path: str, max_model_len: int = 8192,
                 tensor_parallel_size: int = 1):
        self.model_path = model_path
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self._llm = None

    def _init_llm(self):
        if self._llm is None:
            from vllm import LLM
            from vllm.config.attention import AttentionConfig
            logger.info("Initializing vLLM invalidation judge: %s", self.model_path)
            # TRITON_ATTN for Qwen2-arch (QwQ) batch-invariance — FA2 is sm_90-only,
            # crashes on B200. Same pin as src/kg/grading/llm.py.
            self._llm = LLM(model=self.model_path, max_model_len=self.max_model_len,
                            tensor_parallel_size=self.tensor_parallel_size,
                            gpu_memory_utilization=0.95,
                            attention_config=AttentionConfig(backend="TRITON_ATTN"))

    def judge_batch(self, pairs, events_by_id, max_tokens: int = 1024) -> list[bool]:
        # QwQ reasons in a <think> block BEFORE the structured JSON (same as the KG
        # grader, which budgets 2048). Too small a cap truncates the reasoning and the
        # JSON never lands -> _parse silently defaults invalidated=False. 1024 is ample
        # for a binary supersession decision over two short statements.
        if not pairs:
            return []
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        self._init_llm()
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                            structured_outputs=StructuredOutputsParams(
                                json=InvalidationVerdict.model_json_schema()))
        convs = [[{"role": "user",
                   "content": render_pair_prompt(events_by_id[p], events_by_id[s])}]
                 for p, s in pairs]
        logger.info("[vLLM] Judging %d candidate pairs...", len(convs))
        outputs = self._llm.chat(convs, sp)
        return [self._parse(o.outputs[0].text).invalidated for o in outputs]

    @staticmethod
    def _parse(raw: str) -> InvalidationVerdict:
        try:
            return InvalidationVerdict.model_validate_json(raw)
        except Exception:
            start = raw.find("{")
            if start != -1:
                try:
                    return InvalidationVerdict.model_validate_json(raw[start:raw.rfind("}") + 1])
                except Exception:
                    pass
            logger.warning("Unparseable verdict: %s", raw[:120])
            return InvalidationVerdict()


def write_output(rows, updates: dict, out_path: Path | str) -> None:
    """Rewrite every source row; apply per-event `updates`; drop `embedding`."""
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            new_events = []
            for ev in row.get("events", []):
                ev = {k: v for k, v in ev.items() if k != "embedding"}
                if ev.get("id") in updates:
                    ev.update(updates[ev["id"]])
                new_events.append(ev)
            f.write(json.dumps({**row, "events": new_events}, ensure_ascii=False) + "\n")


def inspect_candidates(pairs, events_by_id, out_path) -> dict:
    """Human-readable dump grouped by primary: the primary statement + each
    candidate secondary (relation, valid_at). The no-LLM gate artifact."""
    by_primary = defaultdict(list)
    for pid, sid in pairs:
        by_primary[pid].append(sid)
    lines = [f"# primaries_with_candidates={len(by_primary)} total_pairs={len(pairs)}\n"]
    for pid, sids in sorted(by_primary.items(), key=lambda kv: -len(kv[1])):
        p = events_by_id[pid]
        prel = p["triplets"][0]["relation"] if p.get("triplets") else "-"
        lines.append(f"\n=== PRIMARY [{p.get('valid_at')}] ({p.get('statement_type')}/{prel}) {p['statement']}")
        for sid in sorted(sids, key=lambda i: events_by_id[i].get("valid_at") or ""):
            s = events_by_id[sid]
            srel = s["triplets"][0]["relation"] if s.get("triplets") else "-"
            lines.append(f"    -> [{s.get('valid_at')}] ({s.get('statement_type')}/{srel}) {s['statement']}")
    Path(out_path).write_text("\n".join(lines))
    return {"primaries_with_candidates": len(by_primary), "total_pairs": len(pairs)}


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--disambig", type=Path, required=True)
    p.add_argument("--inspect-candidates", dest="inspect_candidates", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--model", type=str, default=None)     # QwQ judge model path
    p.add_argument("--emb-out", dest="emb_out", type=Path, default=None)
    p.add_argument("--asset-groups", dest="asset_groups", type=Path, default=None,
                   help="entity_groups.json from link_groups; scope invalidation to facts "
                        "whose entities belong to an asset group (Option B). Omit = broad mode.")
    args = p.parse_args()

    # Fail fast BEFORE the (expensive) embedding if the judge run is under-specified.
    if not args.inspect_candidates and (not args.model or not args.out):
        raise SystemExit("Judge run needs --model and --out (or use --inspect-candidates).")

    asset_keys = load_asset_linked_keys(args.asset_groups) if args.asset_groups else None
    eligible, rows = load_eligible_events(args.disambig, asset_keys)
    if asset_keys is not None:
        logger.info("Asset-scoped (Option B): %d asset-linked entity keys -> %d eligible events",
                    len(asset_keys), len(eligible))
    work, members = dedup_representatives(eligible)   # always on: collapse wire reruns + same-fact paraphrases
    logger.info("Dedup (triplet-set + date): %d eligible -> %d representatives (%d duplicate copies collapsed)",
                len(eligible), len(work), len(eligible) - len(work))
    ids, vecs = embed_statements(work)
    emb_by_id = dict(zip(ids, vecs))
    if args.emb_out:
        save_embeddings(args.emb_out, ids, vecs)
    by_id = {e["id"]: e for e in work}
    pairs = build_pairs(work, emb_by_id)
    logger.info("Built %d candidate pairs from %d events", len(pairs), len(work))
    if args.inspect_candidates:                    # no-LLM gate: dump candidate pairs and stop
        stats = inspect_candidates(pairs, by_id, args.inspect_candidates)
        logger.info("Inspect-only (no LLM): %s", stats)
        return
    agent = InvalidationAgent(model_path=args.model)
    verdicts = agent.judge_batch(pairs, by_id)
    apply_verdicts(by_id, pairs, verdicts)
    updates = {eid: {k: by_id[eid][k] for k in
                     ("invalid_at", "invalidated_by", "expired_at", "invalidation_method")}
               for eid in by_id if by_id[eid].get("invalidation_method") == "llm"}
    facts_closed = len(updates)                     # unique representative facts the judge closed
    updates = propagate_updates(updates, members)   # fan rep verdicts out to all duplicate copies
    write_output(rows, updates, args.out)
    summary = {"eligible": len(eligible), "representatives": len(work), "pairs": len(pairs),
               "facts_closed": facts_closed, "events_closed": len(updates)}
    Path(str(args.out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Done: %s", summary)


if __name__ == "__main__":
    main()
