"""KG temporal event-extraction runner: articles -> statements + triplets.

Loads articles via loaders.load_articles, runs LLMTemporalExtractor (3-pass), writes
sidecar JSONL + summary. The sidecar carries row["events"]; kg.grading.runner reads
that same key.

  macronews kg extract --dataset gold --sample-dir data/articles_sample \\
      --output results/kg/gold.jsonl

  macronews kg extract --dataset djnw --input-file <shard>_clean.jsonl \\
      --output results/kg/<shard>.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from macronews import invariants
from macronews.config.paths import MAPPER_MODEL
from macronews.kg.llm import render_mapper_context
from macronews.kg.schemas import ENTITY_TYPES_TUPLE
from macronews.kg.temporal_extractor import LLMTemporalExtractor
from macronews.kg.temporal_schemas import RawTriplet, TemporalEvent
from macronews.loaders import load_articles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

_MIN_RELEVANCE = 0.5  # only mapper tags scored strictly above this prime the extractor


def load_mapper_rows(path: Path) -> dict[str, list[str]]:
    """article_id -> flagged group NAMES with relevance_score > _MIN_RELEVANCE.

    The per-group score lives in `rec["mappings"]` (each entry is
    {group, asset_class, relevance_score, evidence_paragraphs}; see
    pipeline.save_results_jsonl) — the top-level `groups` field carries
    no score, so we read `mappings`. An empty list means the mapper
    covered the article but no group cleared the threshold -> the
    no-groups mapper-context form (the article is still extracted).
    """
    rows: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows[rec["article_id"]] = [
                m["group"] for m in rec["mappings"]
                if m["relevance_score"] > _MIN_RELEVANCE
            ]
    return rows


def attach_mapper_context(
    articles: list[dict], mapper_rows: dict[str, list[str]]
) -> list[dict]:
    """Set a["mapper_context"] on every article from its mapper row. No skipping.

    Errors loudly if an article has no matching mapper row (alignment
    failure — distinct from a row that covers it with zero groups).
    """
    for a in articles:
        aid = a["id"]
        if aid not in mapper_rows:
            raise ValueError(
                f"article {aid!r} has no matching mapper row — mapper/KG "
                f"article sets are misaligned (same --max-model-len + shard?)"
            )
        a["mapper_context"] = render_mapper_context(mapper_rows[aid])
    return articles


def gate_zero_mappings(
    articles: list[dict], mapper_rows: dict[str, list[str]]
) -> list[dict]:
    """Keep only articles the mapper tagged with >=1 group > _MIN_RELEVANCE (an empty
    list, or a missing row, = no-mapping = dropped). Applied UNCONDITIONALLY: a
    no-mapping article can't mapper-join to any asset group, so its facts would be
    dropped downstream by the asset gate anyway — gating at extraction just skips the
    wasted LLM work. (If a future broad/network KG ever needs the no-mapping articles
    — foreign-CB / sovereign / geopolitics content — restore the opt-out from git.)"""
    return [a for a in articles if mapper_rows.get(a["id"])]


_TYPE_NAME_SET = frozenset(ENTITY_TYPES_TUPLE)


def _clean_triplets(triplets: list[RawTriplet]) -> list[RawTriplet]:
    """Drop entity-type-code leaks (subject/object is a literal type code) and
    self-referential triplets. No cross-statement merge — the statement is the
    unit of provenance, so duplicate triples across statements are kept."""
    return [
        t for t in triplets
        if t.subject not in _TYPE_NAME_SET
        and t.object not in _TYPE_NAME_SET
        and t.subject != t.object
    ]


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
    events_by_id: dict[str, list[TemporalEvent]],
) -> dict:
    """Write the event sidecar JSONL + summary. Row shape:
      {"article_id", "date", "events": [<TemporalEvent dict, mode=json>, ...]}
    Each event's triplets are cleaned (type-code/self-ref dropped). Every input
    article gets a row (empty events if the extractor produced none)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_relation: Counter = Counter()
    by_statement_type: Counter = Counter()
    by_temporal_type: Counter = Counter()
    total_events = 0
    total_triplets = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for art in articles:
            events = events_by_id.get(art["id"], [])
            for ev in events:
                ev.triplets = _clean_triplets(ev.triplets)
            row = {
                "article_id": art["id"],
                "date": _article_date(art),
                "headline": art.get("headline", ""),
                "events": [ev.model_dump(mode="json") for ev in events],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_events += len(events)
            for ev in events:
                by_statement_type.update([ev.statement_type])
                by_temporal_type.update([ev.temporal_type])
                total_triplets += len(ev.triplets)
                by_relation.update(t.relation for t in ev.triplets)
    summary = {
        "total_articles": len(articles),
        "total_events": total_events,
        "total_triplets": total_triplets,
        "by_relation": dict(sorted(by_relation.items())),
        "by_statement_type": dict(sorted(by_statement_type.items())),
        "by_temporal_type": dict(sorted(by_temporal_type.items())),
    }
    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %d rows to %s (+ %s). events=%d triplets=%d",
                len(articles), out_path, summary_path.name, total_events, total_triplets)
    return summary


def main():
    invariants.apply()   # before vLLM loads; idempotent, cli.main() may have run it already
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
    p.add_argument("--model", type=str, default=str(MAPPER_MODEL),
                   help="Extractor model (default: Gemma 4 26B-A4B)")
    p.add_argument("--mapper-file", type=Path, default=None,
                   help="REQUIRED (every dataset): mapper sidecar JSONL whose "
                        "article_ids match this run. Run the mapper on this dataset "
                        "first; no group-blind fallback.")
    p.add_argument("--mapper-dir", type=Path, default=None,
                   help="Alternative to --mapper-file: a dir of mapper JSONLs; "
                        "the row for each article_id is looked up across them.")
    # Production context window (same as the mapper and run_kg.sh's djnw mode).
    # Keeps the loader's length cap (max_model_len - 2000) generous so articles
    # are rarely dropped for length on direct runner calls (djnw or sports).
    p.add_argument("--max-model-len", type=int, default=65536)
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
        # formula (config.runconfig.MapperConfig: max_model_len - 2000) so that
        # loader-level filtering produces the SAME article pool given the same
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

    if args.mapper_file is None and args.mapper_dir is None:
        p.error("--mapper-file or --mapper-dir is required (run the mapper on "
                "this dataset first; no group-blind fallback)")
    if args.mapper_file is not None and args.mapper_dir is not None:
        p.error("--mapper-file and --mapper-dir are mutually exclusive")
    if args.mapper_file is not None:
        mapper_rows = load_mapper_rows(args.mapper_file)
    else:
        mapper_rows = {}
        for mf in sorted(args.mapper_dir.glob("*.jsonl")):  # excludes *.summary.json
            mapper_rows.update(load_mapper_rows(mf))
    eligible = attach_mapper_context(eligible, mapper_rows)
    n_flagged = sum(1 for a in eligible if mapper_rows[a["id"]])
    logger.info("Primed %d articles (%d flagged >0.5, %d no-groups)",
                len(eligible), n_flagged, len(eligible) - n_flagged)

    before = len(eligible)
    eligible = gate_zero_mappings(eligible, mapper_rows)   # unconditional: no-mapping = non-asset
    logger.info("Zero-mapping gate: %d -> %d articles (dropped %d no-mapping)",
                before, len(eligible), before - len(eligible))

    # 2. Extract.
    extractor = LLMTemporalExtractor(
        model_path=args.model,
        max_model_len=args.max_model_len,
        tensor_parallel_size=invariants.TENSOR_PARALLEL_SIZE,   # invariant, not a flag
    )
    events_by_id = extractor.extract_batch(eligible, max_tokens=args.max_tokens)

    # 3. Write sidecar. Supersession is handled downstream by the LLM invalidation
    # agent (kg/invalidate_llm.py), a separate post-disambiguation stage — the
    # deterministic reconcile() pass was retired (redundant with the agent).
    summary = write_sidecar(args.output, eligible, events_by_id)
    logger.info("Done. Summary: %s", summary)


if __name__ == "__main__":
    main()
