"""KG grader runner — joins extracted statements to full-text source articles,
grades one statement (with all its triplets) per call, writes a per-statement
sidecar JSONL.

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
from loaders import load_articles  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SKIP_ARTICLE_NOT_IN_SOURCE = "article_not_in_source"


def build_statement_tasks(
    kg_rows: list[dict],
    source_by_id: dict[str, dict],
) -> tuple[list[KGGraderInput], list[dict], list[dict]]:
    """One KGGraderInput per event/statement, joined to its full-text source.

    Records a skip (rather than crashing) when the source article is missing.
    Tasks are emitted sorted by article_id so all statements of one article are
    contiguous — vLLM's prefix cache then reuses the article KV across them.
    """
    inputs: list[KGGraderInput] = []
    meta: list[dict] = []
    skipped: list[dict] = []

    for row in sorted(kg_rows, key=lambda r: r.get("article_id", "")):
        events = row.get("events") or []
        if not events:
            continue
        aid = row["article_id"]
        src = source_by_id.get(aid)
        if src is None:
            for ev in events:
                skipped.append({"article_id": aid, "event_id": ev.get("id"),
                                "statement": ev.get("statement", ""),
                                "skip_reason": SKIP_ARTICLE_NOT_IN_SOURCE})
            logger.warning("SKIP article=%s reason=%s (%d events)",
                           aid, SKIP_ARTICLE_NOT_IN_SOURCE, len(events))
            continue
        full = src["paragraphs"]
        headline = src.get("headline", row.get("headline", ""))
        for ev in events:
            triplets = ev.get("triplets") or []
            inputs.append(KGGraderInput(
                article_id=aid, headline=headline, paragraphs=full,
                statement=ev.get("statement", ""),
                statement_type=ev.get("statement_type", ""),
                triplets=triplets,
                evidence_paragraphs=list(ev.get("evidence_paragraphs", [])),
            ))
            meta.append({"article_id": aid, "date": row.get("date", ""),
                         "event_id": ev.get("id"),
                         "statement": ev.get("statement", ""),
                         "statement_type": ev.get("statement_type", ""),
                         "temporal_type": ev.get("temporal_type", ""),
                         "triplets": triplets,
                         "evidence_paragraphs": list(ev.get("evidence_paragraphs", []))})

    return inputs, meta, skipped


# Free-text "better label" slots; a non-blank value = the judge would re-label
# that triplet (mapped against the extractor's codes offline to find mis-picks).
_SUGGESTION_SLOTS = ("relation_suggestion", "subject_type_suggestion",
                     "object_type_suggestion")


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def write_sidecar(out_path, meta, verdicts, skipped=None) -> dict:
    """Write one JSONL row per statement (+ skipped rows); return a summary.

    `faithful_rate_directional` is the headline A/B metric: faithful rate over
    triplets whose statement the judge flagged `asserts_direction` — keyed on the
    judge's flag, NOT the extractor's relation code, so 26B's REPORTS/IMPACT moves
    are comparable to 31B's CAUSES_*.
    """
    if len(meta) != len(verdicts):
        raise ValueError(f"meta/verdicts mismatch: {len(meta)} vs {len(verdicts)}")
    skipped = skipped or []
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_macro = n_supported = n_asserts = 0
    trip_judged = trip_faithful = 0
    dir_judged = dir_faithful = 0
    mismatch = zero_triplet = 0
    sugg = {s: 0 for s in _SUGGESTION_SLOTS}
    rel_tot, rel_faith = Counter(), Counter()

    with open(out_path, "w", encoding="utf-8") as f:
        for m, v in zip(meta, verdicts):
            in_triplets = m.get("triplets") or []
            n_in = len(in_triplets)
            if n_in == 0:
                # A statement the triplet pass left undecomposed (a unary level/
                # move fact — "S&P 500 rose 0.6%" — or non-macro micro). It still
                # earns a statement-level verdict but has no triplet to judge; the
                # judge's phantom triplet verdict is dropped and this is NOT a
                # count mismatch (0 in -> 0 expected).
                zero_triplet += 1
                bad_len = False
                aligned = False
            else:
                bad_len = len(v.triplets) != n_in
                aligned = not bad_len
                if bad_len:
                    mismatch += 1
            row = {
                **m,
                "grader_verdict_present": True,
                "grader_evidence_paragraphs": list(v.evidence_paragraphs),
                "macro_relevant": v.macro_relevant,
                "supported": v.supported,
                "asserts_direction": v.asserts_direction,
                "triplet_verdicts": (
                    [tv.model_dump() for tv in v.triplets] if aligned else []),
                "triplet_count_mismatch": bad_len,
                "skip_reason": None,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            n_macro += v.macro_relevant
            n_supported += v.supported
            n_asserts += v.asserts_direction
            if not aligned:
                continue
            for in_t, tv in zip(in_triplets, v.triplets):
                trip_judged += 1
                trip_faithful += tv.faithful
                rel = in_t.get("relation", "")
                rel_tot[rel] += 1
                rel_faith[rel] += tv.faithful
                if v.asserts_direction:
                    dir_judged += 1
                    dir_faithful += tv.faithful
                for s in _SUGGESTION_SLOTS:
                    if (getattr(tv, s) or "").strip():
                        sugg[s] += 1
        for s in skipped:
            f.write(json.dumps({**s, "grader_verdict_present": False},
                               ensure_ascii=False) + "\n")

    graded = len(meta)
    faithful_by_relation = {
        rel: {"faithful": rel_faith[rel], "of": rel_tot[rel],
              "rate": _rate(rel_faith[rel], rel_tot[rel])}
        for rel in sorted(rel_tot, key=lambda r: -rel_tot[r])
    }
    summary = {
        "statements_graded": graded,
        "statements_skipped": len(skipped),
        "skipped_by_reason": dict(Counter(s.get("skip_reason") for s in skipped)),
        "zero_triplet_statements": zero_triplet,
        "triplets_judged": trip_judged,
        "macro_relevant_rate": _rate(n_macro, graded),
        "supported_rate": _rate(n_supported, graded),
        "asserts_direction_rate": _rate(n_asserts, graded),
        "faithful_rate": _rate(trip_faithful, trip_judged),
        "faithful_rate_directional": _rate(dir_faithful, dir_judged),
        "faithful_by_relation": faithful_by_relation,
        "suggestion_counts": sugg,
        "triplet_count_mismatch": mismatch,
    }
    with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %d graded + %d skipped to %s; faithful=%s directional=%s",
                graded, len(skipped), out_path,
                summary["faithful_rate"], summary["faithful_rate_directional"])
    return summary


def _load_source(args) -> dict[str, dict]:
    if args.dataset == "gold":
        arts = load_articles(dataset="gold", sample_dir=args.sample_dir)
    elif args.dataset == "sports":
        arts = load_articles(dataset="sports", sample_dir=args.sports_dir,
                             max_articles=args.max_articles)
    else:  # djnw — pre-filter articles whose body alone would push the rendered
           # conversation past max_model_len (mirrors src/mapping/grading/
           # runner.py). An over-long article's statements then skip as
           # article_not_in_source. Buffer = max_tokens (generation) + 3000 for
           # the system prompt (~450) + paragraph numbering + the statement, its
           # triplets, and the evidence line. Measured prompt overhead over the
           # raw body is ~1-1.5k even for the longest articles, so 3000 is safe.
        max_article_tokens = max(1024, args.max_model_len - args.max_tokens - 3000)
        arts = load_articles(
            dataset="djnw", sample_dir=args.data_dir,
            start_date=args.start_date, end_date=args.end_date,
            max_tokens=max_article_tokens, tokenizer_path=args.model,
        )
    return {a["id"]: a for a in arts}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kg-output", required=True, type=Path,
                   help="KG extractor sidecar JSONL (statements to grade)")
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
    inputs, meta, skipped = build_statement_tasks(kg_rows, source_by_id)
    logger.info("Built %d statement-grading tasks (%d skipped)", len(inputs), len(skipped))

    grader = LLMKGGrader(model_path=args.model, max_model_len=args.max_model_len,
                         tensor_parallel_size=args.tensor_parallel_size)
    verdicts = grader.grade_batch(inputs, max_tokens=args.max_tokens)
    summary = write_sidecar(args.output, meta, verdicts, skipped=skipped)
    logger.info("Done. Summary: %s", summary)


if __name__ == "__main__":
    main()
