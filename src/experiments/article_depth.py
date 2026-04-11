"""
Experiment: three-stage LLM classification for article-to-asset mapping.

Pipeline:
  Stage 1: article-level → themes, regions, assets, macro_summary
  Stage 2: paragraph-level with macro_summary context → assets
  Stage 3: validate each (article, asset) pair + select text for embedding
"""

import argparse
import json
import logging
from pathlib import Path

from config.paths import PROMPTS_DIR, DEFAULT_MODEL
from experiments.loaders import load_gold_articles, load_sports_articles, load_wikigaming_articles
from mapping.llm import LLMMapper
from mapping.schemas import AssetMapping, MappingResult, SingleAssetResult, SummarizeResult, ValidationResult
from utils.config import load_asset_universe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SUMMARIZE_PROMPT = (PROMPTS_DIR / "summarize.txt").read_text()
ARTICLE_PROMPT = (PROMPTS_DIR / "single_asset.txt").read_text()
PARAGRAPH_PROMPT = (PROMPTS_DIR / "single_asset_paragraph.txt").read_text()
VALIDATE_PROMPT = (PROMPTS_DIR / "validate.txt").read_text()

# Asset symbol → human-readable name
_ASSET_UNIVERSE = load_asset_universe()
ASSET_NAMES = {sym: info.get("name", sym) for sym, info in _ASSET_UNIVERSE.items()}
ALL_ASSETS = sorted(_ASSET_UNIVERSE.keys())


def _asset_label(sym: str) -> str:
    """Build a human-readable asset label for the LLM (no ticker symbol)."""
    info = _ASSET_UNIVERSE[sym]  # KeyError = bug, fix the universe
    name = info["name"]
    ac = info["asset_class"]
    exchange = info["exchange_name"]
    return f"{name} | {ac} | {exchange}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_articles(
    dataset: str,
    sample_dir: Path,
    max_articles: int | None = None,
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
    """
    if dataset == "gold":
        return load_gold_articles(sample_dir)
    elif dataset == "sports":
        return load_sports_articles(sample_dir, max_articles=max_articles)
    elif dataset == "wikigaming":
        return load_wikigaming_articles(sample_dir, max_articles=max_articles)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")



# ---------------------------------------------------------------------------
# Stage 0: Summarize (one call per article)
# ---------------------------------------------------------------------------

def run_summarize(
    mapper: LLMMapper, articles: list[dict],
) -> list[SummarizeResult]:
    """Stage 0: macro summary + company-specific check, one call per article."""
    logger.info("=== Stage 0: summarize (%d articles) ===", len(articles))
    mapper.system_prompt = SUMMARIZE_PROMPT
    texts = ["\n\n".join(a["paragraphs"]) for a in articles]
    return mapper.map_summarize(texts, max_tokens=1536)


# ---------------------------------------------------------------------------
# Stage 1: ArticleMapper (one call per article × asset)
# ---------------------------------------------------------------------------

def run_article_level(
    mapper: LLMMapper,
    articles: list[dict],
    skip_indices: set[int] | None = None,
) -> list[dict[str, SingleAssetResult]]:
    """Stage 1: per-asset article-level mapping.

    Returns list (one per article) of {asset_sym: SingleAssetResult} for relevant assets only.
    """
    skip = skip_indices or set()
    mapper.system_prompt = ARTICLE_PROMPT

    # Build all (article, asset) pairs
    tasks: list[tuple[int, str]] = []
    texts: list[str] = []
    for art_idx, a in enumerate(articles):
        if art_idx in skip:
            continue
        article_text = "\n\n".join(a["paragraphs"])
        for sym in ALL_ASSETS:
            text = f"{article_text}\n\n[ASSET] {_asset_label(sym)}"
            tasks.append((art_idx, sym))
            texts.append(text)

    logger.info("=== Stage 1: article-level (%d calls) ===", len(texts))
    results = mapper.map_single_asset(texts, max_tokens=512)

    # Group by article, keep only relevant
    by_article: list[dict[str, SingleAssetResult]] = [{} for _ in articles]
    for (art_idx, sym), sar in zip(tasks, results):
        if sar.relevant:
            by_article[art_idx][sym] = sar

    for art_idx, a in enumerate(articles):
        n = len(by_article[art_idx])
        if art_idx not in skip:
            logger.info("  %s: %d assets flagged by ArticleMapper", a["id"], n)

    return by_article


# ---------------------------------------------------------------------------
# Stage 2: ParagraphMapper (one call per paragraph × asset)
# ---------------------------------------------------------------------------

def run_paragraph_level(
    mapper: LLMMapper,
    articles: list[dict],
    context_per_article: list[str],
    skip_indices: set[int] | None = None,
) -> list[dict[str, list[tuple[int, SingleAssetResult]]]]:
    """Stage 2: per-asset paragraph-level mapping.

    Returns list (one per article) of {asset_sym: [(para_idx, SingleAssetResult), ...]}.
    """
    skip = skip_indices or set()
    mapper.system_prompt = PARAGRAPH_PROMPT

    # Build all (paragraph, asset) pairs
    tasks: list[tuple[int, int, str]] = []
    texts: list[str] = []
    for art_idx, a in enumerate(articles):
        if art_idx in skip:
            continue
        context = context_per_article[art_idx]
        for para_idx, para in enumerate(a["paragraphs"]):
            if context:
                para_text = f"[CONTEXT]\n{context}\n[/CONTEXT]\n\n{para}"
            else:
                para_text = para
            for sym in ALL_ASSETS:
                text = f"{para_text}\n\n[ASSET] {_asset_label(sym)}"
                tasks.append((art_idx, para_idx, sym))
                texts.append(text)

    logger.info("=== Stage 2: paragraph-level (%d calls) ===", len(texts))
    results = mapper.map_single_asset(texts, max_tokens=512)

    # Group by article and asset, keep only relevant
    by_article: list[dict[str, list[tuple[int, SingleAssetResult]]]] = [{} for _ in articles]
    for (art_idx, para_idx, sym), sar in zip(tasks, results):
        if sar.relevant:
            by_article[art_idx].setdefault(sym, []).append((para_idx, sar))

    for art_idx, a in enumerate(articles):
        n = len(by_article[art_idx])
        if art_idx not in skip:
            logger.info("  %s: %d unique assets flagged by ParagraphMapper", a["id"], n)

    return by_article


# ---------------------------------------------------------------------------
# Orchestrate Stages 0-2
# ---------------------------------------------------------------------------

def run_two_stage(
    mapper: LLMMapper, articles: list[dict],
) -> tuple[
    list[SummarizeResult],
    list[dict[str, SingleAssetResult]],
    list[dict[str, list[tuple[int, SingleAssetResult]]]],
]:
    """
    Three-stage pipeline (stages 0-2):
      Stage 0: summarize → company_specific + macro_summary
      Stage 1: article-level per-asset mapping
      Stage 2: paragraph-level per-asset mapping with [CONTEXT]

    Returns (summaries, article_results, para_results).
    """
    # Stage 0: summarize
    summaries = run_summarize(mapper, articles)

    # Identify company-specific articles
    company_specific_indices: set[int] = set()
    for i, (a, s) in enumerate(zip(articles, summaries)):
        if s.company_specific:
            logger.info("  %s flagged as company-specific — skipping Stages 1-2", a["id"])
            company_specific_indices.add(i)
        else:
            logger.info("  %s summary: %s", a["id"], s.macro_summary[:100] if s.macro_summary else "(empty)")

    # Stage 1: article-level per-asset
    article_results = run_article_level(mapper, articles, skip_indices=company_specific_indices)

    # Stage 2: paragraph-level per-asset with context
    context_per_article = [s.macro_summary for s in summaries]
    para_results = run_paragraph_level(
        mapper, articles, context_per_article=context_per_article,
        skip_indices=company_specific_indices,
    )

    return summaries, article_results, para_results


# ---------------------------------------------------------------------------
# Stage 3: validation + text selection
# ---------------------------------------------------------------------------

def _build_validation_input(
    article: dict,
    asset: str,
    article_result: SingleAssetResult | None,
    paragraph_results: list[tuple[int, SingleAssetResult]],
) -> str:
    """Build the user message for a validation call.

    The validator always receives the full article so it can reason across
    paragraphs — finding corroborating or contradicting evidence beyond
    what the mappers flagged.
    """
    parts = []

    parts.append("[ARTICLE]")
    for idx, para in enumerate(article["paragraphs"]):
        parts.append(f"[{idx}] {para}")
    parts.append("[/ARTICLE]")

    # Asset
    parts.append(f"\n[ASSET] {_asset_label(asset)}")

    # Mapper reasoning (includes signal strength)
    parts.append("\n[MAPPER REASONING]")
    if article_result:
        parts.append(
            f"ArticleMapper (read the entire article, signal={article_result.signal}, relevance_score={article_result.relevance_score:.2f}): "
            f"{article_result.reasoning}"
        )
    else:
        parts.append(
            "ArticleMapper (read the entire article): did not flag this asset."
        )

    if paragraph_results:
        for para_idx, sar in paragraph_results:
            if sar.reasoning:
                parts.append(
                    f"ParagraphMapper (paragraph [{para_idx}], signal={sar.signal}, relevance_score={sar.relevance_score:.2f}): "
                    f"{sar.reasoning}"
                )
            else:
                parts.append(
                    f"ParagraphMapper (paragraph [{para_idx}], signal={sar.signal}, relevance_score={sar.relevance_score:.2f}): "
                    "flagged this asset but provided no reasoning."
                )
    else:
        parts.append(
            "ParagraphMapper (reviewed individual paragraphs): "
            "did not flag this asset."
        )
    parts.append("[/MAPPER REASONING]")

    return "\n".join(parts)


def build_selected_text(
    paragraphs: list[str],
    evidence_indices: list[int],
) -> str:
    """Construct the text to embed for a validated (article, asset) pair.

    Only includes the raw evidence paragraphs — no LLM-generated reasoning.
    Direction should come from the downstream ridge regression, not from
    the LLM's interpretation baked into the embedding input.
    """
    return " ".join(
        paragraphs[i] for i in evidence_indices if i < len(paragraphs)
    )


def run_validation(
    mapper: LLMMapper,
    articles: list[dict],
    article_results: list[dict[str, SingleAssetResult]],
    para_results: list[dict[str, list[tuple[int, SingleAssetResult]]]],
) -> list[dict]:
    """
    Stage 3: validate each (article, asset) pair proposed by AM or PM.

    Returns list of dicts with keys:
        article_idx, asset, valid, source, evidence_paragraphs,
        reasoning, selected_text
    """
    logger.info("=== Stage 3: validation + text selection ===")
    mapper.system_prompt = VALIDATE_PROMPT

    # Enumerate all (article, asset) pairs from the union of AM and PM
    tasks: list[dict] = []
    input_texts: list[str] = []

    for art_idx, article in enumerate(articles):
        art_syms = set(article_results[art_idx].keys())
        para_syms = set(para_results[art_idx].keys())
        all_proposed = sorted(art_syms | para_syms)

        for sym in all_proposed:
            in_article = sym in art_syms
            in_paragraph = sym in para_syms

            if in_article and in_paragraph:
                source = "both"
            elif in_article:
                source = "article_only"
            else:
                source = "paragraph_only"

            text = _build_validation_input(
                article,
                sym,
                article_result=article_results[art_idx].get(sym),
                paragraph_results=para_results[art_idx].get(sym, []),
            )
            tasks.append({
                "article_idx": art_idx,
                "asset": sym,
                "source": source,
            })
            input_texts.append(text)

    if not input_texts:
        logger.info("  No (article, asset) pairs to validate")
        return []

    logger.info("  %d (article, asset) pairs to validate", len(input_texts))

    # Batch validation call
    validation_results = mapper.map_validate(input_texts, max_tokens=512)

    # Combine results
    output = []
    for task, vr in zip(tasks, validation_results):
        art_idx = task["article_idx"]
        paragraphs = articles[art_idx]["paragraphs"]

        # Range-check evidence_paragraphs
        evidence = [i for i in vr.evidence_paragraphs if 0 <= i < len(paragraphs)]

        selected_text = ""
        if vr.valid and evidence:
            selected_text = build_selected_text(paragraphs, evidence)

        output.append({
            "article_idx": art_idx,
            "asset": task["asset"],
            "valid": vr.valid,
            "signal": vr.signal,
            "relevance_score": vr.relevance_score,
            "source": task["source"],
            "evidence_paragraphs": evidence,
            "reasoning": vr.reasoning,
            "selected_text": selected_text,
        })

    n_valid = sum(1 for r in output if r["valid"])
    n_invalid = len(output) - n_valid
    logger.info("  Validation complete: %d valid, %d rejected", n_valid, n_invalid)

    return output


def run_three_stage(
    mapper: LLMMapper,
    articles: list[dict],
) -> tuple[
    list[SummarizeResult],
    list[dict[str, SingleAssetResult]],
    list[dict[str, list[tuple[int, SingleAssetResult]]]],
    list[dict],
]:
    """
    Four-stage pipeline:
      Stage 0: summarize → company_specific + macro_summary
      Stage 1: article-level per-asset mapping
      Stage 2: paragraph-level per-asset mapping with [CONTEXT]
      Stage 3: validate each proposed (article, asset) pair

    Returns (summaries, article_results, para_results, validation_output).
    """
    summaries, article_results, para_results = run_two_stage(mapper, articles)

    validation_output = run_validation(
        mapper, articles, article_results, para_results,
    )

    return summaries, article_results, para_results, validation_output



def _asset_display_name(sym: str) -> str:
    """Human-readable asset name for JSON output."""
    return ASSET_NAMES.get(sym, sym)


def save_final_results_json(
    articles: list[dict],
    summaries: list[SummarizeResult],
    article_results: list[dict[str, SingleAssetResult]],
    para_results: list[dict[str, list[tuple[int, SingleAssetResult]]]],
    validation_output: list[dict],
    out_path: Path,
) -> None:
    """Save full experiment results to a JSON file."""
    # Group validation results by article
    validation_by_article: dict[int, list[dict]] = {}
    for v in validation_output:
        validation_by_article.setdefault(v["article_idx"], []).append(v)

    total_accepted = 0
    total_rejected = 0
    final_output = []

    for art_idx, art in enumerate(articles):
        summary = summaries[art_idx]
        s1_map = article_results[art_idx]
        s2_map = para_results[art_idx]
        validations = validation_by_article.get(art_idx, [])

        # Stage 3 pairs: accepted and rejected
        accepted = []
        rejected = []
        referenced_paras: set[int] = set()
        for v in validations:
            evidence = v["evidence_paragraphs"]
            # Backfill from Stage 2 if validator returned empty evidence
            if not evidence:
                pm_entries = s2_map.get(v["asset"], [])
                evidence = [idx for idx, _ in pm_entries]
            # Collect mapper reasoning for diagnostics
            s1_sar = s1_map.get(v["asset"])
            s1_r = s1_sar.reasoning if s1_sar else ""
            s1_signal = s1_sar.signal if s1_sar else ""
            s2_entries = s2_map.get(v["asset"], [])
            s2_r_list = [
                {"paragraph": idx, "signal": sar.signal, "relevance_score": sar.relevance_score, "reasoning": sar.reasoning}
                for idx, sar in s2_entries if sar.reasoning
            ]
            # Relevance scores: AM score, max PM score, and validator's final score
            s1_score = s1_sar.relevance_score if s1_sar else 0.0
            s2_score = max((sar.relevance_score for _, sar in s2_entries), default=0.0)
            validator_score = v.get("relevance_score", 0.0)
            entry = {
                "asset": _asset_display_name(v["asset"]),
                "signal": v.get("signal", "strong"),
                "relevance_score": validator_score,
                "mapper_relevance_scores": {"article_mapper": s1_score, "paragraph_mapper_max": s2_score},
                "source": v["source"],
                "evidence_paragraphs": evidence,
                "mapper_reasoning": {
                    "article_mapper": {"signal": s1_signal, "relevance_score": s1_score, "reasoning": s1_r} if s1_sar else "",
                    "paragraph_mapper": s2_r_list,
                },
                "validator_reasoning": v.get("reasoning", ""),
            }
            referenced_paras.update(evidence)
            if v["valid"]:
                accepted.append(entry)
            else:
                rejected.append(entry)

        n_accepted = len(accepted)
        n_rejected = len(rejected)
        n_total = n_accepted + n_rejected
        total_accepted += n_accepted
        total_rejected += n_rejected

        # Collect all referenced paragraphs
        for sym, entries in s2_map.items():
            for idx, _ in entries:
                referenced_paras.add(idx)

        # Paragraphs dict: only include paragraphs referenced by any stage
        paragraphs = art["paragraphs"]
        para_dict = {
            str(i): paragraphs[i]
            for i in sorted(referenced_paras)
            if i < len(paragraphs)
        }

        article_entry = {
            "article_id": art["id"],
            "headline": art["headline"],
            "url": art.get("url", ""),
            "company_specific": summary.company_specific,
            "macro_summary": summary.macro_summary,
            "false_positive_rate": round(n_rejected / n_total, 3) if n_total else 0.0,
            "accepted_assets": [e["asset"] for e in accepted],
            "rejected_assets": [e["asset"] for e in rejected],
            "paragraphs": para_dict,
            "accepted": accepted,
            "rejected": rejected,
        }
        final_output.append(article_entry)

    # Aggregate stats
    grand_total = total_accepted + total_rejected

    # Per-asset-class, per-source, and per-mapper-signal breakdowns
    ac_accepted: dict[str, int] = {}
    ac_rejected: dict[str, int] = {}
    src_accepted: dict[str, int] = {}
    src_rejected: dict[str, int] = {}
    sig_accepted: dict[str, int] = {}
    sig_rejected: dict[str, int] = {}

    for v in validation_output:
        sym = v["asset"]
        art_idx = v["article_idx"]
        ac = _ASSET_UNIVERSE.get(sym, {}).get("asset_class", "unknown")
        source = v["source"]

        # Look up mapper signals
        am_sar = article_results[art_idx].get(sym)
        pm_entries = para_results[art_idx].get(sym, [])

        am_sig = am_sar.signal if am_sar else None
        pm_sig = None
        if pm_entries:
            pm_sig = "strong" if any(sar.signal == "strong" for _, sar in pm_entries) else "weak"

        bucket = sig_accepted if v["valid"] else sig_rejected

        if v["valid"]:
            ac_accepted[ac] = ac_accepted.get(ac, 0) + 1
            src_accepted[source] = src_accepted.get(source, 0) + 1
        else:
            ac_rejected[ac] = ac_rejected.get(ac, 0) + 1
            src_rejected[source] = src_rejected.get(source, 0) + 1

        # Marginal stats
        if am_sig:
            key = f"am_{am_sig}"
            bucket[key] = bucket.get(key, 0) + 1
        if pm_sig:
            key = f"pm_{pm_sig}"
            bucket[key] = bucket.get(key, 0) + 1

        # Joint stats (only when both mappers tagged)
        if am_sig and pm_sig:
            key = f"am_{am_sig}_pm_{pm_sig}"
            bucket[key] = bucket.get(key, 0) + 1

    # Asset class breakdown with per-class FPR
    all_classes = sorted(set(ac_accepted) | set(ac_rejected))
    by_asset_class = {}
    for ac in all_classes:
        a = ac_accepted.get(ac, 0)
        r = ac_rejected.get(ac, 0)
        t = a + r
        by_asset_class[ac] = {
            "accepted": a,
            "rejected": r,
            "total": t,
            "false_positive_rate": round(r / t, 3) if t else 0.0,
        }

    # Source breakdown with acceptance rate
    all_sources = sorted(set(src_accepted) | set(src_rejected))
    by_source = {}
    for src in all_sources:
        a = src_accepted.get(src, 0)
        r = src_rejected.get(src, 0)
        t = a + r
        by_source[src] = {
            "accepted": a,
            "rejected": r,
            "total": t,
            "acceptance_rate": round(a / t, 3) if t else 0.0,
        }

    aggregate = {
        "total_articles": len(articles),
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "total_pairs": grand_total,
        "false_positive_rate": round(total_rejected / grand_total, 3) if grand_total else 0.0,
        "company_specific_count": sum(
            1 for s in summaries if s.company_specific
        ),
        "by_asset_class": by_asset_class,
        "by_source": by_source,
        "by_mapper_signal": {
            key: {
                "accepted": sig_accepted.get(key, 0),
                "rejected": sig_rejected.get(key, 0),
                "total": sig_accepted.get(key, 0) + sig_rejected.get(key, 0),
                "acceptance_rate": round(
                    sig_accepted.get(key, 0) / (sig_accepted.get(key, 0) + sig_rejected.get(key, 0)), 3
                ) if (sig_accepted.get(key, 0) + sig_rejected.get(key, 0)) else 0.0,
            }
            for key in sorted(set(sig_accepted) | set(sig_rejected))
        },
    }

    output = {
        "aggregate": aggregate,
        "articles": final_output,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results for {len(final_output)} articles to {out_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_experiment(
    model_path: str,
    max_model_len: int,
    sample_dir: Path,
    output_json: Path | None = None,
    tensor_parallel_size: int = 1,
    dataset: str = "gold",
    max_articles: int | None = None,
) -> None:
    articles = load_articles(dataset, sample_dir, max_articles=max_articles)

    mapper = LLMMapper(
        model_path=model_path,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )

    summaries, article_results, para_results, validation_output = run_three_stage(
        mapper, articles,
    )
    if output_json:
        save_final_results_json(
            articles, summaries, article_results,
            para_results, validation_output, output_json,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Three-stage LLM classification for article-to-asset mapping"
    )
    parser.add_argument(
        "--model", type=str,
        default=str(DEFAULT_MODEL),
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--sample-dir", type=str, default=None,
                        help="Data directory (default: auto based on --dataset)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Output JSON path (default: results/<dataset>_summary.json)")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--dataset", type=str, default="gold",
        choices=["gold", "sports", "wikigaming"],
        help="Dataset to run: 'gold', 'sports', or 'wikigaming' (default: gold)",
    )
    parser.add_argument(
        "--max-articles", type=int, default=None,
        help="Limit number of articles to process (useful for large datasets)",
    )
    args = parser.parse_args()

    # Default sample-dir and output-json based on dataset
    if args.sample_dir is None:
        if args.dataset == "gold":
            args.sample_dir = "data/articles_sample"
        elif args.dataset == "sports":
            args.sample_dir = "data/sports_news_1994_2000"
        elif args.dataset == "wikigaming":
            args.sample_dir = "data/WikiGaming.jsonl"
    if args.output_json is None:
        args.output_json = f"results/{args.dataset}_summary.json"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_experiment(
        model_path=args.model,
        max_model_len=args.max_model_len,
        sample_dir=Path(args.sample_dir),
        output_json=Path(args.output_json) if args.output_json else None,
        tensor_parallel_size=args.tensor_parallel_size,
        dataset=args.dataset,
        max_articles=args.max_articles,
    )



if __name__ == "__main__":
    main()
