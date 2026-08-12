"""
Macronews ArticleMapper pipeline for article-to-group tagging.

Per-group article-level mapping: one LLM call per (article, group) pair.
Each group's verdict (relevance, evidence_paragraphs, score) fans out to
its member assets in the output JSONL. Asset-class-specific rules are
injected into the prompt per group asset_class.
"""

import json
import logging
from pathlib import Path

from collections import defaultdict

from djnw import runtime as invariants
from macronews.config.paths import PROMPTS_DIR
from macronews.config.runconfig import MapperConfig
from macronews.loaders import load_articles
from macronews.mapping.llm import (
    ASSET_CLASS_DISQUALIFIERS_PLACEHOLDER,
    ASSET_CLASS_POSITIVES_PLACEHOLDER,
    LLMMapper,
)
from macronews.mapping.gate import compile_gate, gate_text
from macronews.mapping.schemas import SingleAssetResult
from macronews.utils.groups import load_group_universe, group_keys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ARTICLE_PROMPT = (PROMPTS_DIR / "mapper.txt").read_text()

# Group universe loaded at import time. Each group has a name,
# asset_class, and a list of members (each {name, ticker_symbol}). The
# mapper iterates groups (not individual assets) and the output fans out
# to per-member entries.
_GROUP_UNIVERSE = load_group_universe()
ALL_GROUPS = group_keys(_GROUP_UNIVERSE)


def _group_label(group_key: str) -> str:
    """Build a human-readable group label for the LLM, listing constituents
    so the model anchors on the actual underlying contracts rather than
    free-associating from the group name (e.g. so "Asia Pacific Equities"
    is not mistaken for Latin American markets, and "Eurozone Rates" is
    not confused with Eurodollar/SOFR which lives under US Rates)."""
    g = _GROUP_UNIVERSE[group_key]
    members = ", ".join(m["name"] for m in g["members"])
    return f"{g['name']} | {g['asset_class']} — constituents: {members}"


# ---------------------------------------------------------------------------
# Helpers

# ---------------------------------------------------------------------------
# Per-asset-class batching helper
# ---------------------------------------------------------------------------

def _run_per_class(
    mapper: LLMMapper,
    prompt_template: str,
    keys: list[str],
    texts: list[str],
    call,
) -> list:
    """Group (key, text) pairs by the group's asset_class, substitute
    class-specific rules into ``prompt_template``, and invoke
    ``call(mapper, batch_texts)`` once per class. Returns results in the
    original input order. ``keys`` are group_keys from group_universe.yaml.

    ``call`` is a callable taking (mapper, batch_texts) and returning a list
    of results parallel to batch_texts.
    """
    if not texts:
        return []
    for placeholder in (
        ASSET_CLASS_DISQUALIFIERS_PLACEHOLDER,
        ASSET_CLASS_POSITIVES_PLACEHOLDER,
    ):
        if placeholder not in prompt_template:
            raise ValueError(
                f"Prompt template is missing {placeholder} placeholder"
            )
    if len(keys) != len(texts):
        raise ValueError(f"keys/texts length mismatch: {len(keys)} vs {len(texts)}")

    by_class: dict[str, list[int]] = defaultdict(list)
    for idx, gk in enumerate(keys):
        ac = _GROUP_UNIVERSE[gk]["asset_class"]
        by_class[ac].append(idx)

    results: list = [None] * len(texts)
    for ac, indices in by_class.items():
        disqualifiers, positives = mapper.asset_class_rules(ac)
        mapper.system_prompt = prompt_template.replace(
            ASSET_CLASS_DISQUALIFIERS_PLACEHOLDER, disqualifiers
        ).replace(
            ASSET_CLASS_POSITIVES_PLACEHOLDER, positives
        )
        batch_texts = [texts[i] for i in indices]
        logger.info("  [%s] %d calls", ac, len(batch_texts))
        batch_results = call(mapper, batch_texts)
        for orig_idx, result in zip(indices, batch_results):
            results[orig_idx] = result
    return results


# ---------------------------------------------------------------------------
# ArticleMapper pipeline (one call per article × group)
# ---------------------------------------------------------------------------

def run_pipeline(
    mapper: LLMMapper,
    articles: list[dict],
    keyword_gate: bool,
) -> tuple[list[dict[str, SingleAssetResult]], dict]:
    """Per-group article-level mapping.

    Returns (results, gate_stats). ``results`` is one entry per article,
    {group_key: SingleAssetResult} for relevant groups only; output fan-out to
    per-member assets happens in save_results_jsonl. ``gate_stats`` records
    whether the gate ran and how many calls it skipped, so a gated run and an
    ungated one are distinguishable from their artifacts alone.

    With ``keyword_gate`` on, an (article, group) pair whose article contains
    none of that group's keywords is skipped without an LLM call. It is
    required, not defaulted: MapperConfig (config/runconfig.py) decides the
    default, based on what the dataset is for.
    """
    gate = compile_gate(_GROUP_UNIVERSE) if keyword_gate else None
    n_gated = 0

    tasks: list[tuple[int, str]] = []
    texts: list[str] = []
    for art_idx, a in enumerate(articles):
        # Pre-LLM filtered articles (DJNW denylist, unembeddable, etc.) produce
        # no mapping tasks; they are still emitted in the final summary with
        # their filtered_reasons recorded.
        if a.get("filtered_reasons"):
            continue
        # Indexed paragraphs so the model can reference them by [N] in evidence_paragraphs.
        article_text = (
            f"[HEADLINE] {a['headline']}\n\n[ARTICLE]\n"
            + "\n\n".join(f"[{i}] {p}" for i, p in enumerate(a["paragraphs"]))
            + "\n[/ARTICLE]"
        )
        gated_on = gate_text(a["headline"], a["paragraphs"]) if gate is not None else ""
        for gk in ALL_GROUPS:
            if gate is not None and not gate[gk].search(gated_on):
                n_gated += 1
                continue
            # Group-last layout so sys_prompt + article body form a shared
            # prefix across siblings in a class batch (prefix-cache reuse).
            # mapper.txt YOUR TASK block instructs the model to read the
            # [ASSET_GROUP] block first despite its physical position at
            # the end.
            text = f"{article_text}\n\n[ASSET_GROUP] {_group_label(gk)}"
            tasks.append((art_idx, gk))
            texts.append(text)

    attempted = len(texts) + n_gated
    gate_stats = {
        "keyword_gate": gate is not None,
        "calls_made": len(texts),
        "calls_skipped": n_gated,
        "calls_saved_pct": (100 * n_gated / attempted) if attempted else 0.0,
    }
    if gate is not None:
        logger.info(
            "=== keyword gate: %d of %d pairs skipped (%.1f%% of calls saved) ===",
            n_gated, attempted, gate_stats["calls_saved_pct"],
        )
    logger.info("=== ArticleMapper: %d calls (groups) ===", len(texts))
    results = _run_per_class(
        mapper,
        ARTICLE_PROMPT,
        keys=[gk for _, gk in tasks],
        texts=texts,
        call=lambda m, ts: m.map_single_asset(ts, max_tokens=512),
    )

    by_article: list[dict[str, SingleAssetResult]] = [{} for _ in articles]
    for (art_idx, gk), sar in zip(tasks, results):
        if sar.relevant:
            by_article[art_idx][gk] = sar

    for art_idx, a in enumerate(articles):
        logger.info("  %s: %d groups flagged by ArticleMapper", a["id"], len(by_article[art_idx]))

    return by_article, gate_stats


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _refuse_mixed_gate(out_path: Path, gate_stats: dict) -> None:
    """Fail if a sibling shard in this directory ran with the other gate setting.

    The array launcher skips shards whose output already exists, so re-running a
    directory after flipping the gate would leave the old shards ungated and gate
    only the new ones — a corpus that is half one thing and half another.
    """
    mine = gate_stats["keyword_gate"]
    for sibling in sorted(out_path.parent.glob("*.summary.json")):
        if sibling.name == out_path.with_suffix(".summary.json").name:
            continue
        try:
            summary = json.loads(sibling.read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(
                f"{sibling} is not valid JSON ({e}). A damaged or unreadable summary "
                f"must not read as \"no sibling\" -- this guard exists precisely to "
                f"catch a half-gated corpus, and a directory with a damaged summary is "
                f"exactly the one that needs it most. Repair or remove {sibling.name} "
                f"before re-running."
            ) from e
        # A summary with no `gate` block was written before the gate existed,
        # which means that shard is ungated. Absence is False, not unknown --
        # resuming the pre-gate prod run with gated code is exactly the mistake
        # this guard is for, and every shard of it looks like this.
        theirs = summary.get("gate", {}).get("keyword_gate", False)
        if theirs != mine:
            raise ValueError(
                f"{out_path.parent} already holds shards with keyword_gate={theirs} "
                f"(e.g. {sibling.name}), but this run has keyword_gate={mine}. "
                f"Mixing them makes the directory a half-gated corpus. Write to a "
                f"different --output-dir, or re-run the existing shards to match."
            )


def save_results_jsonl(
    articles: list[dict],
    article_results: list[dict[str, SingleAssetResult]],
    out_path: Path,
    gate_stats: dict,
    run_record: dict,
) -> None:
    """Write one JSONL record per article + a sidecar `<basename>.summary.json`.

    ``gate_stats`` is required, not defaulted, and is checked against the
    shards already in the output directory: a run is a corpus, and half of it
    gated is a silently corrupt one.

    Each `mappings` entry corresponds to one LLM group decision: it carries
    the group name, its asset_class, the relevance_score, and the
    evidence_paragraphs. Per-asset details are not duplicated into mappings
    — consumers can join group_name -> members via group_universe.yaml, or
    use the top-level `assets` field which lists every member of every
    fired group as `{name, ticker_symbol}` pairs.

    Counts in the summary sidecar are per-group-decision (not per-member);
    `by_asset_class` counts group decisions per class. summarize.py rolls
    these up across shards.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _refuse_mixed_gate(out_path, gate_stats)

    total_mappings = 0
    by_asset_class_count: dict[str, int] = {}
    by_group_count: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    n_filtered = 0
    n_written = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for art_idx, art in enumerate(articles):
            am_map = article_results[art_idx]
            mappings = []
            assets = []
            referenced_paras: set[int] = set()
            # Stable ordering: group_key alphabetical; within a group, member
            # order is YAML insertion order (preserved by safe_load).
            for gk in sorted(am_map.keys()):
                sar = am_map[gk]
                evidence = list(sar.evidence_paragraphs)
                gv = _GROUP_UNIVERSE[gk]
                ac = gv["asset_class"]
                gname = gv["name"]
                mappings.append({
                    "group": gname,
                    "asset_class": ac,
                    "relevance_score": sar.relevance_score,
                    "evidence_paragraphs": evidence,
                })
                for member in gv["members"]:
                    assets.append({
                        "name": member["name"],
                        "ticker_symbol": member["ticker_symbol"],
                    })
                referenced_paras.update(evidence)
                by_asset_class_count[ac] = by_asset_class_count.get(ac, 0) + 1
                by_group_count[gname] = by_group_count.get(gname, 0) + 1
            total_mappings += len(mappings)

            reasons = art.get("filtered_reasons") or []
            if reasons:
                n_filtered += 1
                for r in reasons:
                    reason_counts[r] = reason_counts.get(r, 0) + 1

            paragraphs = art["paragraphs"]
            para_dict = {
                str(i): paragraphs[i]
                for i in sorted(referenced_paras)
                if i < len(paragraphs)
            }

            record = {
                "article_id": art["id"],
                "headline": art["headline"],
                "filtered_reasons": reasons,
                "groups": [m["group"] for m in mappings],
                "assets": assets,
                "paragraphs": para_dict,
                "mappings": mappings,
            }
            url = art.get("url")
            if url:
                record["url"] = url
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    aggregate = {
        "total_articles": len(articles),
        "filtered_articles": n_filtered,
        "filtered_by_reason": reason_counts,
        "run": run_record,
        "gate": gate_stats,
        "total_mappings": total_mappings,
        "by_asset_class": {ac: by_asset_class_count[ac] for ac in sorted(by_asset_class_count)},
        "by_group": {g: by_group_count[g] for g in sorted(by_group_count)},
    }
    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {n_written} articles to {out_path} (+ {summary_path.name})")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_experiment(cfg: MapperConfig) -> None:
    """Run the mapper for one resolved config.

    max_article_tokens comes from the config rather than being recomputed here: it
    follows from max_model_len, and having it in two places is how a serving knob
    silently changed which articles were mapped.
    """
    articles = load_articles(
        cfg.dataset,
        cfg.sample_dir,
        max_articles=cfg.max_articles,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        random_seed=cfg.random_seed,
        max_tokens=cfg.max_article_tokens,
        tokenizer_path=str(cfg.model),
        chars_per_token=2.0,
        input_file=cfg.input_file,
    )

    mapper = LLMMapper(
        model_path=str(cfg.model),
        max_model_len=cfg.max_model_len,
        tensor_parallel_size=invariants.TENSOR_PARALLEL_SIZE,   # invariant, not a flag
    )

    article_results, gate_stats = run_pipeline(
        mapper, articles, keyword_gate=cfg.keyword_gate
    )
    save_results_jsonl(articles, article_results, cfg.output_path, gate_stats,
                       run_record=cfg.record())
