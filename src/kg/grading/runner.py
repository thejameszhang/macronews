"""KG grader runner — joins extracted facts to full-text source articles,
grades one fact per call, writes a per-fact sidecar JSONL.

  python src/kg/grading/runner.py \\
      --kg-output results/kg/dev/march_2022_dev.v2.4.jsonl \\
      --dataset djnw --data-dir <shards> --start-date 2022-03 --end-date 2022-03 \\
      --output results/kg/dev/march_2022_dev.v2.4.grader.jsonl \\
      --model /nfs/roberts/scratch/pi_btk22/jyz32/qwq-32b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from kg.grading.llm import KGGraderInput, LLMKGGrader  # noqa: E402
from pipeline import load_articles  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SKIP_ARTICLE_NOT_IN_SOURCE = "article_not_in_source"
SKIP_PARAGRAPH_MISALIGNMENT = "paragraph_misalignment"


def _fact_meta(fct: dict) -> dict:
    return {
        "subject": fct["subject"], "subject_type": fct["subject_type"],
        "relation": fct["relation"], "object": fct["object"],
        "object_type": fct["object_type"],
        "evidence_paragraphs": list(fct.get("evidence_paragraphs", [])),
    }


def build_fact_tasks(
    kg_rows: list[dict],
    source_by_id: dict[str, dict],
) -> tuple[list[KGGraderInput], list[dict], list[dict]]:
    """One KGGraderInput per fact, joined to its full-text source article.

    Records skips (rather than crashing) when the source article is missing or
    when the sidecar's sparse paragraph text disagrees with the reloaded source
    at the same index (the index-alignment guard).
    """
    inputs: list[KGGraderInput] = []
    meta: list[dict] = []
    skipped: list[dict] = []

    for row in kg_rows:
        facts = row.get("facts") or []
        if not facts:
            continue
        aid = row["article_id"]
        src = source_by_id.get(aid)
        if src is None:
            for fct in facts:
                skipped.append({"article_id": aid, "skip_reason": SKIP_ARTICLE_NOT_IN_SOURCE,
                                **_fact_meta(fct)})
            logger.warning("SKIP article=%s reason=%s (%d facts)",
                           aid, SKIP_ARTICLE_NOT_IN_SOURCE, len(facts))
            continue
        full = src["paragraphs"]
        sparse = row.get("paragraphs") or {}
        # Sidecar keys are str(int) paragraph indices (src/kg/runner.py writes
        # {str(i): text}), so int(k) cannot raise; compare each stored
        # paragraph against the reloaded full article at the same index.
        mis = next(
            (k for k, txt in sparse.items()
             if not (0 <= int(k) < len(full)) or full[int(k)] != txt),
            None,
        )
        if mis is not None:
            for fct in facts:
                skipped.append({"article_id": aid, "skip_reason": SKIP_PARAGRAPH_MISALIGNMENT,
                                **_fact_meta(fct)})
            logger.warning("SKIP article=%s reason=%s key=%s", aid,
                           SKIP_PARAGRAPH_MISALIGNMENT, mis)
            continue
        headline = row.get("headline") or src.get("headline", "")
        for fct in facts:
            inputs.append(KGGraderInput(
                article_id=aid, headline=headline, paragraphs=full,
                subject=fct["subject"], subject_type=fct["subject_type"],
                relation=fct["relation"], object=fct["object"],
                object_type=fct["object_type"],
                evidence_paragraphs=list(fct.get("evidence_paragraphs", [])),
            ))
            meta.append({"article_id": aid, "date": row.get("date", ""), **_fact_meta(fct)})

    return inputs, meta, skipped


# The two fail-able boolean axes (False = a problem with the fact).
_BOOL_AXES = ["macro_relevant", "correct"]
# Free-text "better label" slots (non-blank = the judge would re-label it). Mapping
# a non-blank suggestion against the extractor's code list offline recovers both
# mis-pick (suggestion is a listed code) and schema-gap (out-of-schema).
_SUGGESTION_SLOTS = {
    "subject_type": "subject_type_suggestion",
    "relation": "relation_suggestion",
    "object_type": "object_type_suggestion",
}


def write_sidecar(out_path, meta, verdicts, skipped=None) -> dict:
    """Write one JSONL row per graded fact (+ skipped rows); return a summary."""
    if len(meta) != len(verdicts):
        raise ValueError(f"meta/verdicts mismatch: {len(meta)} vs {len(verdicts)}")
    skipped = skipped or []
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    axis_fail = {ax: 0 for ax in _BOOL_AXES}
    suggestion_counts = {slot: 0 for slot in _SUGGESTION_SLOTS}
    # Per-relation / per-type incorrect (correct=False) tallies for the summary.
    _GROUPS = ("relation", "subject_type", "object_type")
    tot = {k: Counter() for k in _GROUPS}
    bad = {k: Counter() for k in _GROUPS}

    with open(out_path, "w", encoding="utf-8") as f:
        for m, v in zip(meta, verdicts):
            row = {
                **m,
                "grader_verdict_present": True,
                "grader_evidence_paragraphs": list(v.evidence_paragraphs),
                "macro_relevant": v.macro_relevant,
                "subject_type_suggestion": v.subject_type_suggestion,
                "relation_suggestion": v.relation_suggestion,
                "object_type_suggestion": v.object_type_suggestion,
                "correct": v.correct,
                "skip_reason": None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            for ax in _BOOL_AXES:
                if getattr(v, ax) is False:
                    axis_fail[ax] += 1
            for slot, field in _SUGGESTION_SLOTS.items():
                if (getattr(v, field) or "").strip():
                    suggestion_counts[slot] += 1
            for k in _GROUPS:
                tot[k][m[k]] += 1
                if v.correct is False:
                    bad[k][m[k]] += 1
        for s in skipped:
            row = {**s, "grader_verdict_present": False}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _incorrect_by(k):
        # {label: {n incorrect, of total, rate}} sorted by # incorrect desc.
        return {key: {"n": bad[k][key], "of": tot[k][key],
                      "rate": round(bad[k][key] / tot[k][key], 2)}
                for key in sorted(tot[k], key=lambda x: -bad[k][x])}

    summary = {
        "graded": len(meta),
        "skipped": len(skipped),
        "axis_fail_counts": axis_fail,
        "suggestion_counts": suggestion_counts,
        "incorrect_by_relation": _incorrect_by("relation"),
        "incorrect_by_subject_type": _incorrect_by("subject_type"),
        "incorrect_by_object_type": _incorrect_by("object_type"),
    }
    with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %d graded + %d skipped to %s; fails=%s suggestions=%s",
                len(meta), len(skipped), out_path, axis_fail, suggestion_counts)
    return summary


def _load_source(args) -> dict[str, dict]:
    if args.dataset == "gold":
        arts = load_articles(dataset="gold", sample_dir=args.sample_dir)
    elif args.dataset == "sports":
        arts = load_articles(dataset="sports", sample_dir=args.sports_dir,
                             max_articles=args.max_articles)
    else:  # djnw — load the date range WITHOUT token filtering so every
           # fact-bearing article is present to join (max_tokens=None).
        arts = load_articles(
            dataset="djnw", sample_dir=args.data_dir,
            start_date=args.start_date, end_date=args.end_date,
            max_tokens=None,
        )
    return {a["id"]: a for a in arts}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kg-output", required=True, type=Path,
                   help="KG extractor sidecar JSONL (facts to grade)")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", required=True, type=str)
    p.add_argument("--dataset", choices=["gold", "djnw", "sports"], required=True)
    p.add_argument("--sample-dir", type=Path, default=None)   # gold
    p.add_argument("--sports-dir", type=Path, default=None)   # sports
    p.add_argument("--data-dir", type=Path, default=None)     # djnw
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--max-articles", type=int, default=None)  # sports cap
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=2048)
    args = p.parse_args()

    # Each dataset needs its source directory; fail with a clear CLI message
    # rather than an obscure error deep inside load_articles.
    required_dir = {"gold": "sample-dir", "sports": "sports-dir", "djnw": "data-dir"}[args.dataset]
    if getattr(args, required_dir.replace("-", "_")) is None:
        p.error(f"--dataset {args.dataset} requires --{required_dir}")
    if not args.kg_output.exists():
        p.error(f"--kg-output not found: {args.kg_output}")

    source_by_id = _load_source(args)
    logger.info("Loaded %d source articles", len(source_by_id))
    with open(args.kg_output, encoding="utf-8") as f:
        kg_rows = [json.loads(line) for line in f if line.strip()]
    inputs, meta, skipped = build_fact_tasks(kg_rows, source_by_id)
    logger.info("Built %d fact-grading tasks (%d skipped)", len(inputs), len(skipped))

    grader = LLMKGGrader(model_path=args.model, max_model_len=args.max_model_len,
                         tensor_parallel_size=args.tensor_parallel_size)
    verdicts = grader.grade_batch(inputs, max_tokens=args.max_tokens)
    summary = write_sidecar(args.output, meta, verdicts, skipped=skipped)
    logger.info("Done. Summary: %s", summary)


if __name__ == "__main__":
    main()
