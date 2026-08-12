"""Grader runner — joins mapper output to source articles, batches grader calls,
writes sidecar JSONL. QwQ-32B verification judge; invoked via `macronews mapper
grade` (see --help for flags). One folder per production run, mapper and grader
side by side: results/mapping/prod/<run>/{mapper,grader}/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from djnw import runtime as invariants
from macronews.config.runconfig import GraderConfig
from macronews.mapping.grading.llm import GraderInput, LLMGrader
from macronews.mapping.grading.schemas import GraderResult, is_self_inconsistent
from macronews.loaders import load_articles
from macronews.utils.groups import load_group_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


SKIP_GROUP_NOT_IN_UNIVERSE = "group_not_in_universe"
SKIP_ARTICLE_NOT_IN_SOURCE = "article_not_in_source"


def build_tasks(
    mapper_rows: list[dict],
    source_by_id: dict[str, dict],
    group_by_name: dict[str, dict],
) -> tuple[list[GraderInput], list[dict], list[dict]]:
    """Join mapper output to source articles.

    Returns three parallel lists:
      - ``inputs``: ``GraderInput`` objects for mappings that will be graded.
      - ``meta``: per-mapping metadata (article_id, group_name, asset_class,
        mapper_score, mapper_evidence_paragraphs) parallel to ``inputs``.
      - ``skipped``: rows for mappings that could NOT be graded (source
        article missing, or group renamed/removed in group_universe.yaml).
        Each carries a ``skip_reason`` and the available metadata so the
        sidecar surfaces these rather than silently dropping them.

    Each skip is also logged at WARN level with article_id + group so it
    appears in SLURM logs.
    """
    inputs: list[GraderInput] = []
    meta: list[dict] = []
    skipped: list[dict] = []

    for row in mapper_rows:
        aid = row["article_id"]
        src = source_by_id.get(aid)
        if src is None:
            for m in row.get("mappings", []):
                skipped.append(
                    {
                        "article_id": aid,
                        "group_name": m.get("group", ""),
                        "asset_class": m.get("asset_class", ""),
                        "mapper_score": m.get("relevance_score"),
                        "mapper_evidence_paragraphs": list(m.get("evidence_paragraphs", [])),
                        "skip_reason": SKIP_ARTICLE_NOT_IN_SOURCE,
                    }
                )
                logger.warning(
                    "SKIP article=%s group=%r reason=%s",
                    aid, m.get("group", ""), SKIP_ARTICLE_NOT_IN_SOURCE,
                )
            continue
        headline = row.get("headline") or src.get("headline", "")
        paragraphs = src["paragraphs"]
        for m in row.get("mappings", []):
            gname = m["group"]
            gv = group_by_name.get(gname)
            if gv is None:
                skipped.append(
                    {
                        "article_id": aid,
                        "group_name": gname,
                        "asset_class": m.get("asset_class", ""),
                        "mapper_score": m.get("relevance_score"),
                        "mapper_evidence_paragraphs": list(m.get("evidence_paragraphs", [])),
                        "skip_reason": SKIP_GROUP_NOT_IN_UNIVERSE,
                    }
                )
                logger.warning(
                    "SKIP article=%s group=%r reason=%s",
                    aid, gname, SKIP_GROUP_NOT_IN_UNIVERSE,
                )
                continue
            inputs.append(
                GraderInput(
                    article_id=aid,
                    headline=headline,
                    paragraphs=paragraphs,
                    group_name=gname,
                    asset_class=m["asset_class"],
                    member_names=[mem["name"] for mem in gv["members"]],
                    mapper_score=m["relevance_score"],
                    mapper_evidence_paragraphs=list(m.get("evidence_paragraphs", [])),
                )
            )
            meta.append(
                {
                    "article_id": aid,
                    "group_name": gname,
                    "asset_class": m["asset_class"],
                    "mapper_score": m["relevance_score"],
                    "mapper_evidence_paragraphs": list(m.get("evidence_paragraphs", [])),
                }
            )

    return inputs, meta, skipped


def write_sidecar(
    out_path: Path,
    meta: list[dict],
    results: list[GraderResult],
    run_record: dict,
    skipped: list[dict] | None = None,
) -> dict:
    """Write the sidecar JSONL and return a small summary dict.

    Graded rows have ``grader_verdict`` ∈ {correct, borderline, incorrect}
    and ``skip_reason`` = null. Skipped rows have ``grader_verdict`` =
    "skipped", null grader scores/evidence, and a non-null
    ``skip_reason`` naming the cause. This keeps the sidecar
    self-describing — any consumer can group by ``skip_reason`` to
    audit grading coverage without a separate file.
    """
    if len(meta) != len(results):
        raise ValueError(
            f"meta/results length mismatch: {len(meta)} vs {len(results)}"
        )
    skipped = skipped or []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_correct = n_incorrect = n_inconsistent = 0
    skip_reasons: dict[str, int] = {}
    by_class: dict[str, dict[str, int]] = {}
    by_group: dict[str, dict[str, int]] = {}

    def _bump(d: dict[str, dict[str, int]], key: str, verdict: str, flag: bool) -> None:
        bucket = d.setdefault(
            key, {"total": 0, "correct": 0, "incorrect": 0, "self_inconsistent": 0}
        )
        bucket["total"] += 1
        bucket[verdict] += 1
        if flag:
            bucket["self_inconsistent"] += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for m, r in zip(meta, results):
            flag = is_self_inconsistent(
                m["mapper_score"], r.relevance_score, r.verdict
            )
            row = {
                "article_id": m["article_id"],
                "group_name": m["group_name"],
                "asset_class": m["asset_class"],
                "mapper_score": m["mapper_score"],
                "mapper_evidence_paragraphs": m["mapper_evidence_paragraphs"],
                "grader_evidence_paragraphs": list(r.evidence_paragraphs),
                "grader_score": r.relevance_score,
                "grader_verdict": r.verdict,
                "self_consistency_flag": flag,
                "skip_reason": None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if r.verdict == "correct":
                n_correct += 1
            else:
                n_incorrect += 1
            if flag:
                n_inconsistent += 1
            _bump(by_class, m["asset_class"], r.verdict, flag)
            _bump(by_group, m["group_name"], r.verdict, flag)
        for s in skipped:
            row = {
                "article_id": s["article_id"],
                "group_name": s["group_name"],
                "asset_class": s.get("asset_class", ""),
                "mapper_score": s.get("mapper_score"),
                "mapper_evidence_paragraphs": s.get("mapper_evidence_paragraphs", []),
                "grader_evidence_paragraphs": [],
                "grader_score": None,
                "grader_verdict": "skipped",
                "self_consistency_flag": False,
                "skip_reason": s["skip_reason"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            skip_reasons[s["skip_reason"]] = skip_reasons.get(s["skip_reason"], 0) + 1

    n_graded = len(meta)
    n_skipped = len(skipped)

    def _with_pct(bucket: dict[str, int]) -> dict[str, float | int]:
        total = bucket["total"]
        return {
            **bucket,
            "correct_pct": round(100 * bucket["correct"] / total, 2) if total else 0.0,
        }

    summary = {
        "run": run_record,
        "total_mappings": n_graded + n_skipped,
        "graded": n_graded,
        "correct": n_correct,
        "incorrect": n_incorrect,
        "self_inconsistent": n_inconsistent,
        "skipped": n_skipped,
        "skip_reasons": skip_reasons,
        "by_asset_class": {k: _with_pct(v) for k, v in sorted(by_class.items())},
        "by_group": {k: _with_pct(v) for k, v in sorted(by_group.items())},
    }
    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Wrote %d graded + %d skipped rows to %s (+ %s); skip_reasons=%s",
        n_graded, n_skipped, out_path, summary_path.name, skip_reasons,
    )
    return summary


def grade(cfg: GraderConfig) -> None:
    """Verify one mapper output with QwQ.

    max_article_tokens comes from the config, which reserves 3000 tokens for this
    stage (the mapper reserves 2000) because the grader's generation budget is 1024 to
    the mapper's 512. That difference decides which articles load.
    """
    if cfg.dataset == "gold":
        articles = load_articles(dataset="gold", sample_dir=cfg.sample_dir)
    else:
        articles = load_articles(
            dataset="djnw",
            sample_dir=cfg.input_file.parent,
            input_file=cfg.input_file,
            max_tokens=cfg.max_article_tokens,
            tokenizer_path=str(cfg.model),
        )
    source_by_id = {a["id"]: a for a in articles}
    logger.info("Loaded %d source articles", len(source_by_id))

    with open(cfg.mapper_output, encoding="utf-8") as f:
        mapper_rows = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d mapper rows from %s", len(mapper_rows), cfg.mapper_output)

    group_universe = load_group_universe()
    group_by_name = {gv["name"]: gv for gv in group_universe.values()}
    inputs, meta, skipped = build_tasks(mapper_rows, source_by_id, group_by_name)
    logger.info("Built %d grader inputs (%d skipped — see WARN logs)",
                len(inputs), len(skipped))

    grader = LLMGrader(
        model_path=str(cfg.model),
        max_model_len=cfg.max_model_len,
        tensor_parallel_size=invariants.TENSOR_PARALLEL_SIZE,   # invariant, not a flag
    )
    results = grader.grade_batch(inputs, max_tokens=cfg.max_tokens)

    summary = write_sidecar(cfg.output, meta, results, skipped=skipped,
                            run_record=cfg.record())
    logger.info("Done. Summary: %s", summary)
