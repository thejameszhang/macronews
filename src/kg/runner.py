"""KG fact-extraction runner.

Loads articles via pipeline.load_articles, runs LLMExtractor, writes
sidecar JSONL + summary. CLI mirrors src/mapping/grading/runner.py.

Gold usage:
  python src/kg/runner.py \\
      --dataset gold \\
      --sample-dir data/articles_sample \\
      --output results/kg/gold.jsonl \\
      --model ~/models/gemma-4-31b-it

DJNW usage (wired but not the v1 acceptance):
  python src/kg/runner.py \\
      --dataset djnw \\
      --input-file <path>/2014-05c_clean.jsonl \\
      --output results/kg/2014-05c.jsonl \\
      --model ~/models/gemma-4-31b-it
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from kg.llm import LLMExtractor  # noqa: E402
from kg.schemas import (  # noqa: E402
    ENTITY_TYPES_TUPLE,
    KGArticleResult,
    KGFact,
)
from pipeline import load_articles  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Runner-side cleanup applied post-generation:
#   1. Drop facts whose subject/object string is a literal entity-type
#      code (schema leak — the prompt rule occasionally misses these).
#   2. Drop facts where subject == object (self-referential).
#   3. Dedup by (subject, relation, object), unioning evidence_paragraphs.
_TYPE_NAME_SET = frozenset(ENTITY_TYPES_TUPLE)


def _postprocess_facts(facts: list[KGFact]) -> list[KGFact]:
    """Strip type-code leaks and self-references, then merge duplicate
    (s, r, o) triples.

    Preserves first-occurrence ordering of unique triples; evidence
    paragraph lists are sorted-unique-merged. Type tags on the
    surviving merged fact come from the first occurrence.
    """
    cleaned: list[KGFact] = [
        f for f in facts
        if f.subject not in _TYPE_NAME_SET
        and f.object not in _TYPE_NAME_SET
        and f.subject != f.object
    ]
    merged: dict[tuple[str, str, str], KGFact] = {}
    for f in cleaned:
        key = (f.subject, f.relation, f.object)
        if key in merged:
            existing = merged[key]
            combined = sorted(
                set(existing.evidence_paragraphs) | set(f.evidence_paragraphs)
            )
            merged[key] = KGFact(
                evidence_paragraphs=combined,
                subject=existing.subject,
                subject_type=existing.subject_type,
                relation=existing.relation,
                object=existing.object,
                object_type=existing.object_type,
            )
        else:
            merged[key] = f
    return list(merged.values())


def _article_date(article: dict) -> str:
    """Extract the article date in ISO YYYY-MM-DD form.

    Gold articles carry `date` directly. DJNW articles may carry `date`
    or `publication_date`; fall back to empty string if neither is
    present so the JSONL row is still well-formed.
    """
    return article.get("date") or article.get("publication_date") or ""


def write_sidecar(
    out_path: Path,
    articles: list[dict],
    results: list[KGArticleResult],
) -> dict:
    """Write the sidecar JSONL and return a small summary dict.

    JSONL row shape (v2):
      {"article_id", "date", "headline", "facts": [...],
       "paragraphs": {str(idx): text}}

    Facts come before paragraphs (priority-ordered for readability).
    `paragraphs` is a dict containing only the indices that appear in
    some fact's evidence_paragraphs — un-referenced paragraphs would
    just bloat the file (same pattern as src/pipeline.py).

    Summary shape (v2):
      {"total_articles", "total_facts",
       "raw_facts_before_postprocess",
       "facts_removed_by_postprocess",
       "by_relation": {...},
       "by_subject_type": {...},
       "by_object_type": {...}}
    """
    if len(articles) != len(results):
        raise ValueError(
            f"articles/results length mismatch: {len(articles)} vs {len(results)}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_relation: Counter = Counter()
    by_subject_type: Counter = Counter()
    by_object_type: Counter = Counter()
    total_facts = 0
    raw_fact_total = 0    # before postprocess, for cleanup logging

    with open(out_path, "w", encoding="utf-8") as f:
        for art, res in zip(articles, results):
            raw_fact_total += len(res.facts)
            clean_facts = _postprocess_facts(res.facts)
            paragraphs = art.get("paragraphs", [])
            referenced = sorted({
                i for fct in clean_facts for i in fct.evidence_paragraphs
                if 0 <= i < len(paragraphs)
            })
            para_dict = {str(i): paragraphs[i] for i in referenced}
            row = {
                "article_id": art["id"],
                "date": _article_date(art),
                "headline": art.get("headline", ""),
                "facts": [fct.model_dump() for fct in clean_facts],
                "paragraphs": para_dict,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_facts += len(clean_facts)
            by_relation.update(fct.relation for fct in clean_facts)
            by_subject_type.update(fct.subject_type for fct in clean_facts)
            by_object_type.update(fct.object_type for fct in clean_facts)

    summary = {
        "total_articles": len(articles),
        "total_facts": total_facts,
        "raw_facts_before_postprocess": raw_fact_total,
        "facts_removed_by_postprocess": raw_fact_total - total_facts,
        "by_relation": dict(sorted(by_relation.items())),
        "by_subject_type": dict(sorted(by_subject_type.items())),
        "by_object_type": dict(sorted(by_object_type.items())),
    }
    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(
        "Wrote %d rows to %s (+ %s). facts=%d (removed %d)",
        len(articles), out_path, summary_path.name,
        total_facts, raw_fact_total - total_facts,
    )
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["gold", "djnw", "sports"], required=True)
    p.add_argument("--sample-dir", type=Path, default=None,
                   help="For --dataset gold: directory containing gold_*.json")
    p.add_argument("--sports-dir", type=Path, default=None,
                   help="For --dataset sports: root of data/sports_news_1994_2000")
    p.add_argument("--input-file", type=Path, default=None,
                   help="For --dataset djnw: path to a single *_clean.jsonl shard. "
                        "Mutually exclusive with --data-dir / --start-date / --end-date.")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="For --dataset djnw: directory of *_clean.jsonl shards. "
                        "Used with --start-date and --end-date to load a date range "
                        "and (optionally) random-sample from it. Mirrors mapper DEV mode.")
    p.add_argument("--start-date", type=str, default=None,
                   help="For --dataset djnw + --data-dir: YYYY-MM lower bound (inclusive).")
    p.add_argument("--end-date", type=str, default=None,
                   help="For --dataset djnw + --data-dir: YYYY-MM upper bound (inclusive).")
    p.add_argument("--max-articles", type=int, default=None,
                   help="Cap on number of articles loaded. For --dataset djnw + "
                        "--data-dir, combine with --random-seed for deterministic "
                        "sub-sampling; also caps --dataset sports.")
    p.add_argument("--random-seed", type=int, default=None,
                   help="For --dataset djnw + --data-dir: random seed for sub-sampling. "
                        "If set, loads ALL matching articles then samples --max-articles.")
    p.add_argument("--output", required=True, type=Path,
                   help="Sidecar JSONL to write")
    p.add_argument("--model", required=True, type=str,
                   help="Path to Gemma 4 31B (or compatible) model directory")
    # Production context window (same as the mapper and run_kg.sh's djnw mode).
    # Keeps the loader's length cap (max_model_len - 2000) generous so articles
    # are rarely dropped for length on direct runner calls (djnw or sports).
    p.add_argument("--max-model-len", type=int, default=65536)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Generation token budget per article. v1 default 2048 (gold). "
             "DJNW deployment may need higher — observe output-token "
             "distribution before increasing; bumping max_tokens raises "
             "vLLM KV memory pressure at scale.",
    )
    args = p.parse_args()

    # 1. Load source articles.
    if args.dataset == "gold":
        if args.sample_dir is None:
            p.error("--dataset gold requires --sample-dir")
        articles = load_articles(dataset="gold", sample_dir=args.sample_dir)
    elif args.dataset == "sports":
        if args.sports_dir is None:
            p.error("--dataset sports requires --sports-dir")
        # Same token budget as the djnw path so a long sports article can't
        # blow the model context (max_model_len - 2000, matching the mapper).
        articles = load_articles(
            dataset="sports", sample_dir=args.sports_dir,
            max_articles=args.max_articles,
            max_tokens=max(1024, args.max_model_len - 2000),
            tokenizer_path=args.model,
        )
    else:  # djnw
        # Two sub-modes, mutually exclusive:
        #   (A) --input-file: process a single *_clean.jsonl shard end-to-end
        #   (B) --data-dir [+ --start-date/--end-date/--max-articles/--random-seed]:
        #       load across multiple shards by date, with deterministic sub-sampling.
        #       Mirrors mapper DEV mode exactly so seed=42 reproduces the same articles.
        if args.input_file is None and args.data_dir is None:
            p.error("--dataset djnw requires --input-file OR --data-dir")
        if args.input_file is not None and args.data_dir is not None:
            p.error("--input-file and --data-dir are mutually exclusive")
        # Pre-filter long articles by token budget. Matches the mapper's
        # formula (src/pipeline.py: max_model_len - 2000) so that loader-
        # level filtering produces the SAME article pool given the same
        # max_model_len + start/end-date + seed inputs. Reproducibility
        # is more important than the larger safety buffer.
        max_article_tokens = max(1024, args.max_model_len - 2000)
        sample_dir = args.data_dir if args.data_dir is not None else args.input_file.parent
        articles = load_articles(
            dataset="djnw",
            sample_dir=sample_dir,
            input_file=args.input_file,
            start_date=args.start_date,
            end_date=args.end_date,
            max_articles=args.max_articles,
            random_seed=args.random_seed,
            max_tokens=max_article_tokens,
            tokenizer_path=args.model,
        )
    # Skip articles already marked by the filter waterfall (tabular_body, etc.).
    eligible = [a for a in articles if not a.get("filtered_reasons")]
    logger.info(
        "Loaded %d articles (%d eligible after filter-waterfall skip)",
        len(articles), len(eligible),
    )

    # 2. Extract.
    extractor = LLMExtractor(
        model_path=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    results = extractor.extract_batch(eligible, max_tokens=args.max_tokens)

    # 3. Write sidecar.
    summary = write_sidecar(args.output, eligible, results)
    logger.info("Done. Summary: %s", summary)


if __name__ == "__main__":
    main()
