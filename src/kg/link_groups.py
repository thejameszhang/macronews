"""Ex-post asset-group entity linking (v1 named-asset branch).

Phase 1 is deterministic: exact-match + mapper-join; the unconfirmed residual is
logged and DROPPED. Phase 2 adds an A4B LLM judge over the residual.
See docs/superpowers/specs/2026-06-09-kg-asset-group-linking-design.md (v3).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# asset_class -> entity types that can legitimately BE that asset (the type-gate).
ASSET_CLASS_TYPES: dict[str, set[str]] = {
    "commodity": {"COMMODITY", "ASSET_METRIC"},
    "equity index": {"EQUITY_INDEX"},
    "US equity sector": {"US_GICS_SECTOR"},
    "currency": {"CURRENCY"},
    "rates": {"GOV_BOND", "INTEREST_RATE"},
    "volatility": {"FIN_INSTRUMENT", "ASSET_METRIC"},
}
ALL_ASSET_TYPES: set[str] = set().union(*ASSET_CLASS_TYPES.values())


@dataclass
class GroupConfig:
    key: str
    name: str
    asset_class: str
    types: set[str]               # type-gate
    alias_rx: re.Pattern          # word-boundary over name+members+short_names+keywords
    exact_set: set[str]           # casefolded name+member names+short_names (fast path)


def build_group_configs(universe: dict) -> list[GroupConfig]:
    configs: list[GroupConfig] = []
    for key, gv in universe.items():
        exact_terms = {gv["name"]}
        for m in gv["members"]:
            exact_terms.add(m["name"])
            exact_terms.add(m["short_name"])
        alias_terms = {t.casefold() for t in exact_terms} | {k.casefold() for k in gv["keywords"]}
        rx = re.compile(r"\b(" + "|".join(re.escape(t) for t in
                                          sorted(alias_terms, key=len, reverse=True)) + r")\b")
        configs.append(GroupConfig(
            key=key, name=gv["name"], asset_class=gv["asset_class"],
            types=ASSET_CLASS_TYPES[gv["asset_class"]],
            alias_rx=rx,
            exact_set={t.casefold() for t in exact_terms},
        ))
    return configs


def candidates(name: str, etype: str, configs: list[GroupConfig]) -> list[GroupConfig]:
    """Groups an (entity name, type) is a candidate for: type-gate ∧ alias match."""
    if etype not in ALL_ASSET_TYPES:
        return []
    cf = name.casefold()
    return [g for g in configs if etype in g.types and g.alias_rx.search(cf)]


def build_mapper_index(mapper_path: Path | str) -> dict[str, dict[str, set[int]]]:
    """article_id -> {group_NAME: set(evidence_paragraphs)} for mappings score>0.5.
    Keyed by NAME because the mapper sidecar emits group names, not keys."""
    idx: dict[str, dict[str, set[int]]] = {}
    for line in Path(mapper_path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tags = {m["group"]: set(m.get("evidence_paragraphs", []))
                for m in r.get("mappings", []) if m["relevance_score"] > 0.5}
        if tags:
            idx[r["article_id"]] = tags
    return idx


@dataclass
class EntityLinks:
    name: str                                    # first-seen surface form (display)
    etype: str
    exact: set[str] = field(default_factory=set)            # group keys
    confirmed: dict[str, str] = field(default_factory=dict)  # key -> "mapper-para"|"mapper-article"
    residual_keys: set[str] = field(default_factory=set)    # group keys -> judge
    residual_rows: list[dict] = field(default_factory=list)  # audit/judge-input rows
    statements: list[str] = field(default_factory=list)     # judge context (capped)


_MAX_STMTS = 3


def accumulate_links(disambig_path: Path | str, configs: list[GroupConfig],
                     mapper_index: dict[str, dict[str, set[int]]],
                     para_strict: bool = True) -> dict[str, EntityLinks]:
    """Walk the disambig sidecar; per canonical (casefold name) accumulate exact /
    mapper-confirmed / residual group keys + residual audit rows + context statements."""
    by_key: dict[str, EntityLinks] = {}
    for line in Path(disambig_path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        aid = row["article_id"]
        tags = mapper_index.get(aid, {})
        for ev in row.get("events", []):
            pstmt = set(ev.get("evidence_paragraphs") or [])
            stmt = ev.get("statement", "")
            for t in ev.get("triplets", []):
                for nm, ty in ((t["subject"], t["subject_type"]),
                               (t["object"], t["object_type"])):
                    cands = candidates(nm, ty, configs)
                    if not cands:
                        continue
                    cf = nm.casefold()
                    el = by_key.setdefault(cf, EntityLinks(name=nm, etype=ty))
                    if stmt and len(el.statements) < _MAX_STMTS and stmt not in el.statements:
                        el.statements.append(stmt)
                    for g in cands:
                        mp = tags.get(g.name)                 # mapper tags by NAME
                        tagged = mp is not None
                        overlap = tagged and bool(pstmt & mp)
                        confirmed = overlap or (tagged and not para_strict)
                        if cf in g.exact_set:
                            el.exact.add(g.key)
                        elif confirmed:
                            # a later article-level hit must not downgrade a para hit
                            if g.key not in el.confirmed or overlap:
                                el.confirmed[g.key] = "mapper-para" if overlap else "mapper-article"
                        else:
                            el.residual_keys.add(g.key)
                            el.residual_rows.append({
                                "entity": nm, "type": ty, "group_key": g.key,
                                "article_id": aid, "evidence_paragraphs": sorted(pstmt),
                                "mapper_article_tagged": tagged,
                                "mapper_evidence_paragraphs": sorted(mp or []),
                            })
    # Exact/confirmed in ANY occurrence wins over residual; drop now-redundant residuals.
    for el in by_key.values():
        el.residual_keys -= (el.exact | set(el.confirmed))
        el.residual_rows = [r for r in el.residual_rows if r["group_key"] in el.residual_keys]
    return by_key


def _write_residual(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _judge_inputs(by_key: dict[str, EntityLinks], cfg_by_key: dict[str, GroupConfig],
                  universe: dict) -> tuple[list, list[str]]:
    """One GroupJudgeInput per entity that has residual candidates (Phase 2)."""
    items, entity_keys = [], []
    for cf, el in by_key.items():
        if not el.residual_keys:
            continue
        from kg.group_judge import GroupJudgeInput      # lazy: Phase 2 only (fires only when residual exists)
        cands = []
        for gk in sorted(el.residual_keys):
            g = cfg_by_key[gk]
            members = [m["short_name"] for m in universe[gk]["members"]]
            cands.append((gk, g.name, g.asset_class, members))
        items.append(GroupJudgeInput(
            entity=el.name, etype=el.etype, statements=el.statements,
            candidates=cands, mapper_prior_keys=sorted(el.confirmed)))
        entity_keys.append(cf)
    return items, entity_keys


def link_entities(disambig_path: Path | str, mapper_path: Path | str | None,
                  output_path: Path | str, residual_path: Path | str,
                  group_members_path: Path | str, use_llm: bool = True,
                  para_strict: bool = True, model_path: str | None = None) -> dict:
    """Run the ex-post linker. Writes entity_groups.json (entity->groups, machine),
    group_members.json (group->entities, human audit — mirrors clusters.json), and
    residual.jsonl. Returns a summary. The judge (use_llm=True) needs a GPU and
    group_judge.py (Phase 2)."""
    from utils.groups import load_group_universe
    universe = load_group_universe()
    configs = build_group_configs(universe)
    cfg_by_key = {g.key: g for g in configs}
    mapper_index = build_mapper_index(mapper_path) if mapper_path else {}

    by_key = accumulate_links(disambig_path, configs, mapper_index, para_strict)

    residual_rows = [r for el in by_key.values() for r in el.residual_rows]
    _write_residual(residual_rows, Path(residual_path))

    llm_links: dict[str, set[str]] = {}
    if use_llm:
        items, entity_keys = _judge_inputs(by_key, cfg_by_key, universe)
        if items:
            from kg.group_judge import LLMGroupJudge   # lazy: Phase 2 only
            judge = LLMGroupJudge(model_path=model_path)
            verdicts = judge.judge_batch(items)
            for cf, v in zip(entity_keys, verdicts):
                allowed = by_key[cf].residual_keys
                llm_links[cf] = {k for k in v.group_keys if k in allowed}

    entity_groups: dict[str, dict] = {}
    for cf, el in by_key.items():
        groups = ([{"key": k, "method": "exact"} for k in sorted(el.exact)]
                  + [{"key": k, "method": el.confirmed[k]} for k in sorted(el.confirmed)]
                  + [{"key": k, "method": "llm"} for k in sorted(llm_links.get(cf, set()))])
        if groups:
            entity_groups[cf] = {"type": el.etype, "groups": groups}

    entity_groups = dict(sorted(entity_groups.items()))   # deterministic key order
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(entity_groups, ensure_ascii=False, indent=2))

    # Inverse, human-auditable view: asset group -> the entities pulled under its
    # umbrella (mirrors disambiguate.py's clusters.json). Keyed by display name;
    # members ordered by method (exact < mapper-para < mapper-article < llm) then name.
    name_by_key = {g.key: g.name for g in configs}
    _ord = {"exact": 0, "mapper-para": 1, "mapper-article": 2, "llm": 3}
    group_members: dict[str, list] = {}
    for cf, rec in entity_groups.items():
        for grp in rec["groups"]:
            group_members.setdefault(name_by_key[grp["key"]], []).append(
                {"entity": by_key[cf].name, "type": rec["type"], "method": grp["method"]})
    for gname in group_members:
        group_members[gname].sort(key=lambda m: (_ord.get(m["method"], 9), m["entity"].casefold()))
    group_members = dict(sorted(group_members.items()))
    Path(group_members_path).parent.mkdir(parents=True, exist_ok=True)
    Path(group_members_path).write_text(json.dumps(group_members, ensure_ascii=False, indent=2))

    summary = {"linked_entities": len(entity_groups),
               "groups_with_members": len(group_members),
               "residual_rows": len(residual_rows),
               "residual_entities": sum(1 for el in by_key.values() if el.residual_keys)}
    logger.info("Linked %d entities; residual %d rows / %d entities%s",
                summary["linked_entities"], summary["residual_rows"],
                summary["residual_entities"], "" if use_llm else " (judge skipped)")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("disambig", type=Path, help="disambiguated KG sidecar JSONL")
    p.add_argument("--mapper", type=Path, default=None, help="mapper sidecar JSONL")
    p.add_argument("--output", type=Path, required=True, help="entity_groups.json out")
    p.add_argument("--residual", type=Path, required=True, help="residual.jsonl out")
    p.add_argument("--group-members", type=Path, required=True,
                   help="group_members.json out (group -> entities, human audit)")
    p.add_argument("--model", type=str,
                   default="/nfs/roberts/scratch/pi_btk22/jyz32/gemma-4-26b-a4b-it",
                   help="A4B judge model path (Phase 2)")
    p.add_argument("--no-llm", action="store_true",
                   help="skip the residual judge (join + exact only; CPU) — Phase 1 default")
    p.add_argument("--article-level", action="store_true",
                   help="PARA_STRICT=False (count mapper-article hits as confirmed)")
    args = p.parse_args()
    link_entities(args.disambig, args.mapper, args.output, args.residual,
                  args.group_members, use_llm=not args.no_llm,
                  para_strict=not args.article_level, model_path=args.model)


if __name__ == "__main__":
    main()
