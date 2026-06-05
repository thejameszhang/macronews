"""
Macronews ArticleMapper pipeline for article-to-group tagging.

Per-group article-level mapping: one LLM call per (article, group) pair.
Each group's verdict (relevance, evidence_paragraphs, score) fans out to
its member assets in the output JSONL. Asset-class-specific rules are
injected into the prompt per group asset_class.
"""

import argparse
import json
import logging
from pathlib import Path

from collections import defaultdict

from config.paths import PROMPTS_DIR, DEFAULT_MODEL
from loaders import load_gold_articles, load_sports_articles, load_wikigaming_articles, load_djnw_articles
from mapping.llm import (
    ASSET_CLASS_DISQUALIFIERS_PLACEHOLDER,
    ASSET_CLASS_POSITIVES_PLACEHOLDER,
    LLMMapper,
)
from mapping.schemas import SingleAssetResult
from utils.groups import load_group_universe, group_keys

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

def load_articles(
    dataset: str,
    sample_dir: Path,
    max_articles: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    random_seed: int | None = None,
    max_tokens: int | None = None,
    tokenizer_path: str | None = None,
    chars_per_token: float = 2.0,
    input_file: Path | None = None,
) -> list[dict]:
    """Load articles in the standard schema {id, headline, paragraphs}.

    Parameters
    ----------
    dataset : str
        "gold" for gold_*.json articles, "sports" for sports news.
    sample_dir : Path
        Directory to load from.
    max_articles : int, optional
        Limit number of articles (useful for sports with 5000+ articles).
    start_date, end_date : str, optional
        YYYY-MM bounds for djnw monthly files.
    random_seed : int, optional
        Seed for random sampling within djnw. Default is the first ``max_articles``.
    max_tokens : int, optional
        Skip articles whose text exceeds this token count (djnw only).
    tokenizer_path : str, optional
        Model path used to load the tokenizer for the ``max_tokens`` filter.
    chars_per_token : float, optional
        Conservative lower bound on the tokenizer's char-to-token ratio for the
        fast-path filter. Default 2.0 is safe for Gemma English.
    """
    if dataset == "gold":
        return load_gold_articles(sample_dir)
    elif dataset == "sports":
        return load_sports_articles(
            sample_dir,
            max_articles=max_articles,
            max_tokens=max_tokens,
            tokenizer_path=tokenizer_path,
            chars_per_token=chars_per_token,
        )
    elif dataset == "wikigaming":
        return load_wikigaming_articles(sample_dir, max_articles=max_articles)
    elif dataset == "djnw":
        return load_djnw_articles(
            sample_dir,
            max_articles=max_articles,
            start_date=start_date,
            end_date=end_date,
            random_seed=random_seed,
            max_tokens=max_tokens,
            tokenizer_path=tokenizer_path,
            input_file=input_file,
            chars_per_token=chars_per_token,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")



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
) -> list[dict[str, SingleAssetResult]]:
    """Per-group article-level mapping.

    Returns list (one per article) of {group_key: SingleAssetResult} for
    relevant groups only. Output fan-out to per-member assets happens in
    save_results_jsonl.
    """
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
        for gk in ALL_GROUPS:
            # Group-last layout so sys_prompt + article body form a shared
            # prefix across siblings in a class batch (prefix-cache reuse).
            # mapper.txt YOUR TASK block instructs the model to read the
            # [ASSET_GROUP] block first despite its physical position at
            # the end.
            text = f"{article_text}\n\n[ASSET_GROUP] {_group_label(gk)}"
            tasks.append((art_idx, gk))
            texts.append(text)

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

    return by_article


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results_jsonl(
    articles: list[dict],
    article_results: list[dict[str, SingleAssetResult]],
    out_path: Path,
) -> None:
    """Write one JSONL record per article + a sidecar `<basename>.summary.json`.

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

def run_experiment(
    model_path: str,
    max_model_len: int,
    sample_dir: Path,
    output_path: Path | None = None,
    tensor_parallel_size: int = 1,
    dataset: str = "gold",
    max_articles: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    random_seed: int | None = None,
    input_file: Path | None = None,
) -> None:
    # Leave ~2K tokens headroom for the system prompt and generation budget.
    # djnw loader uses this to skip articles whose text alone would push the
    # prompt past ``max_model_len``.
    max_article_tokens = max(1024, max_model_len - 2000)
    chars_per_token = 2.0
    articles = load_articles(
        dataset,
        sample_dir,
        max_articles=max_articles,
        start_date=start_date,
        end_date=end_date,
        random_seed=random_seed,
        max_tokens=max_article_tokens,
        tokenizer_path=model_path,
        chars_per_token=chars_per_token,
        input_file=input_file,
    )

    mapper = LLMMapper(
        model_path=model_path,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )

    article_results = run_pipeline(mapper, articles)
    if output_path:
        save_results_jsonl(articles, article_results, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="ArticleMapper: per-asset article-level LLM classification"
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=["dev", "prod"],
        help="dev: ad-hoc with sampling/date-range. prod: one shard file -> one JSONL.",
    )
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )

    # DEV-only
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["gold", "sports", "wikigaming", "djnw"],
                        help="[dev] Dataset name")
    parser.add_argument("--sample-dir", type=str, default=None,
                        help="[dev] Data directory (default: auto based on --dataset)")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="[dev] Limit number of articles")
    parser.add_argument("--start-date", type=str, default=None,
                        help="[dev] Earliest monthly file (YYYY-MM, djnw only)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="[dev] Latest monthly file inclusive (YYYY-MM, djnw only)")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="[dev] Seed for random sampling within date range")
    parser.add_argument("--output-file", type=str, default=None,
                        help="[dev] Output JSONL path")

    # PROD-only
    parser.add_argument("--input-file", type=str, default=None,
                        help="[prod] Single *_clean.jsonl shard to process")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="[prod] Output directory; filename derived from input basename")

    args = parser.parse_args()

    dev_args = (args.dataset, args.sample_dir, args.max_articles,
                args.start_date, args.end_date, args.random_seed, args.output_file)
    prod_args = (args.input_file, args.output_dir)

    if args.mode == "dev":
        if any(a is not None for a in prod_args):
            parser.error("--input-file/--output-dir are prod-only")
        if args.dataset is None:
            parser.error("--mode dev requires --dataset")
        if args.output_file is None:
            args.output_file = f"results/{args.dataset}.jsonl"
        if args.sample_dir is None:
            if args.dataset == "gold":
                args.sample_dir = "data/articles_sample"
            elif args.dataset == "sports":
                args.sample_dir = "data/sports_news_1994_2000"
            elif args.dataset == "wikigaming":
                args.sample_dir = "data/WikiGaming.jsonl"
            elif args.dataset == "djnw":
                args.sample_dir = "/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles"
        sample_dir = Path(args.sample_dir)
        output_path = Path(args.output_file)
        input_file = None
        dataset = args.dataset
    else:  # prod
        if any(a is not None for a in dev_args):
            parser.error("dev-only args (--dataset, --sample-dir, --max-articles, "
                         "--start-date, --end-date, --random-seed, --output-file) "
                         "cannot be combined with --mode prod")
        if args.input_file is None or args.output_dir is None:
            parser.error("--mode prod requires --input-file and --output-dir")
        input_file = Path(args.input_file)
        # Strip _clean.jsonl -> .jsonl  (e.g., 2015-06a_clean.jsonl -> 2015-06a.jsonl)
        basename = input_file.name
        if basename.endswith("_clean.jsonl"):
            out_name = basename[: -len("_clean.jsonl")] + ".jsonl"
        else:
            out_name = input_file.stem + ".jsonl"
        output_path = Path(args.output_dir) / out_name
        sample_dir = input_file.parent
        dataset = "djnw"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_experiment(
        model_path=args.model,
        max_model_len=args.max_model_len,
        sample_dir=sample_dir,
        output_path=output_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dataset=dataset,
        max_articles=args.max_articles,
        start_date=args.start_date,
        end_date=args.end_date,
        random_seed=args.random_seed,
        input_file=input_file,
    )



if __name__ == "__main__":
    main()
