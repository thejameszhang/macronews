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
from mapping.schemas import AssetMapping, MappingResult, ValidationResult
from utils.config import load_asset_universe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ARTICLE_PROMPT = (PROMPTS_DIR / "article.txt").read_text()
PARAGRAPH_PROMPT = (PROMPTS_DIR / "paragraph.txt").read_text()
VALIDATE_PROMPT = (PROMPTS_DIR / "validate.txt").read_text()

# Asset symbol → human-readable name (for Stage 3 prompt)
_ASSET_UNIVERSE = load_asset_universe()
ASSET_NAMES = {sym: info.get("name", sym) for sym, info in _ASSET_UNIVERSE.items()}


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
# Run functions
# ---------------------------------------------------------------------------

def run_article_level(
    mapper: LLMMapper, articles: list[dict],
) -> list[MappingResult]:
    """Stage 1: single LLM call per article."""
    logger.info("=== Stage 1: article-level (%d calls) ===", len(articles))
    mapper.system_prompt = ARTICLE_PROMPT
    texts = ["\n\n".join(a["paragraphs"]) for a in articles]
    return mapper.map(texts, max_tokens=1536, guided=False)


def run_paragraph_level(
    mapper: LLMMapper,
    articles: list[dict],
    context_per_article: list[str],
    skip_indices: set[int] | None = None,
) -> tuple[list[MappingResult], list[list[tuple[int, MappingResult]]]]:
    """
    Stage 2: one LLM call per paragraph, union across paragraphs.

    Each paragraph's user message is prepended with [CONTEXT] from Stage 1.
    Articles in skip_indices are skipped (e.g., company-specific articles).
    """
    logger.info("=== Stage 2: paragraph-level ===")
    mapper.system_prompt = PARAGRAPH_PROMPT

    skip = skip_indices or set()

    # Flatten all paragraphs with tracking
    all_paras: list[tuple[int, int, str]] = []
    for art_idx, a in enumerate(articles):
        if art_idx in skip:
            logger.info("  Skipping %s (company-specific)", a["id"])
            continue
        for para_idx, para in enumerate(a["paragraphs"]):
            text = para
            if context_per_article[art_idx]:
                text = f"[CONTEXT]\n{context_per_article[art_idx]}\n[/CONTEXT]\n\n{para}"
            all_paras.append((art_idx, para_idx, text))

    logger.info("  %d total paragraphs across %d articles", len(all_paras), len(articles))
    para_raw = mapper.map([text for _, _, text in all_paras], max_tokens=1024)

    # Re-group by article
    para_by_article: dict[int, list[tuple[int, MappingResult]]] = {}
    for (art_idx, para_idx, _), result in zip(all_paras, para_raw):
        para_by_article.setdefault(art_idx, []).append((para_idx, result))

    # Aggregate: unique asset symbols across paragraphs
    # Reasoning is not carried here — the validator builds its own
    # per-paragraph reasoning lookup from `details`.
    union_results: list[MappingResult] = []
    details: list[list[tuple[int, MappingResult]]] = []
    for art_idx in range(len(articles)):
        art_details = para_by_article.get(art_idx, [])
        details.append(art_details)
        seen: set[str] = set()
        for _, r in art_details:
            seen.update(am.asset for am in r.relevant_assets)
        union = [AssetMapping(asset=s) for s in sorted(seen)]
        union_results.append(MappingResult(relevant_assets=union))

    return union_results, details


def run_two_stage(
    mapper: LLMMapper, articles: list[dict],
) -> tuple[list[MappingResult], list[MappingResult], list[list[tuple[int, MappingResult]]]]:
    """
    Two-stage context-augmented pipeline:
      Stage 1: article-level → themes, regions, assets, macro_summary
      Stage 2: paragraph-level with [CONTEXT] macro_summary → assets
      Final: union of Stage 1 and Stage 2 assets

    Returns (final_union_results, article_results, para_details).
    """
    # Stage 1: article-level
    article_results = run_article_level(mapper, articles)

    # Identify company-specific articles to skip in Stage 2
    company_specific_indices: set[int] = set()
    for i, (a, r) in enumerate(zip(articles, article_results)):
        if r.company_specific:
            logger.info("  %s flagged as company-specific — skipping Stages 2-3", a["id"])
            company_specific_indices.add(i)

    # Extract macro summaries for Stage 2 context
    summaries = [r.macro_summary for r in article_results]
    for a, s in zip(articles, summaries):
        logger.info("  %s summary: %s", a["id"], s[:100] if s else "(empty)")

    # Stage 2: paragraph-level with context (skip company-specific)
    para_results, para_details = run_paragraph_level(
        mapper, articles, context_per_article=summaries,
        skip_indices=company_specific_indices,
    )

    # Final: union of article-level and paragraph-level assets
    # Company-specific articles get empty results (no union needed)
    final_results: list[MappingResult] = []
    for art_idx in range(len(articles)):
        if art_idx in company_specific_indices:
            final_results.append(MappingResult(
                relevant_assets=[],
                company_specific=True,
                macro_summary="",
            ))
            continue
        # Merge AssetMappings by symbol, preferring ArticleMapper reasoning
        art_mappings = {am.asset: am for am in article_results[art_idx].relevant_assets}
        para_mappings = {am.asset: am for am in para_results[art_idx].relevant_assets}
        all_syms = sorted(set(art_mappings) | set(para_mappings))
        union = [art_mappings[s] if s in art_mappings else para_mappings[s] for s in all_syms]
        final_results.append(MappingResult(
            relevant_assets=union,
            company_specific=False,
            macro_summary=article_results[art_idx].macro_summary,
        ))

    return final_results, article_results, para_details


# ---------------------------------------------------------------------------
# Stage 3: validation + text selection
# ---------------------------------------------------------------------------

def _build_validation_input(
    article: dict,
    asset: str,
    article_reasoning: str | None,
    paragraph_reasonings: list[tuple[int, str]],
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
    asset_name = ASSET_NAMES.get(asset, asset)
    parts.append(f"\n[ASSET] {asset} ({asset_name})")

    # Mapper reasoning
    parts.append("\n[MAPPER REASONING]")
    if article_reasoning:
        parts.append(
            f"ArticleMapper (read the entire article): {article_reasoning}"
        )
    else:
        parts.append(
            "ArticleMapper (read the entire article): did not flag this asset."
        )

    if paragraph_reasonings:
        for para_idx, reasoning in paragraph_reasonings:
            if reasoning:
                parts.append(
                    f"ParagraphMapper (paragraph [{para_idx}]): {reasoning}"
                )
            else:
                parts.append(
                    f"ParagraphMapper (paragraph [{para_idx}]): flagged this "
                    "asset but provided no reasoning."
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
    article_results: list[MappingResult],
    para_details: list[list[tuple[int, MappingResult]]],
    final_union_results: list[MappingResult],
) -> list[dict]:
    """
    Stage 3: validate each (article, asset) pair and select text.

    Returns list of dicts with keys:
        article_idx, asset, valid, source, evidence_paragraphs,
        reasoning, selected_text
    """
    logger.info("=== Stage 3: validation + text selection ===")
    mapper.system_prompt = VALIDATE_PROMPT

    # Build ArticleMapper reasoning lookup: art_idx -> {symbol: reasoning}
    s1_reasonings: list[dict[str, str]] = []
    for ar in article_results:
        s1_reasonings.append({am.asset: am.reasoning for am in ar.relevant_assets})

    # Build ParagraphMapper reasoning + paragraph index lookup:
    # art_idx -> {symbol: [(para_idx, reasoning), ...]}
    s2_reasonings: list[dict[str, list[tuple[int, str]]]] = []
    for art_idx in range(len(articles)):
        asset_para_reasonings: dict[str, list[tuple[int, str]]] = {}
        for para_idx, r in para_details[art_idx]:
            for am in r.relevant_assets:
                asset_para_reasonings.setdefault(am.asset, []).append(
                    (para_idx, am.reasoning)
                )
        s2_reasonings.append(asset_para_reasonings)

    # Enumerate all (article, asset) pairs and classify cases
    tasks: list[dict] = []
    input_texts: list[str] = []

    for art_idx, final_r in enumerate(final_union_results):
        art_syms = set(s1_reasonings[art_idx])
        para_syms = set(s2_reasonings[art_idx])

        for am in final_r.relevant_assets:
            asset = am.asset
            in_article = asset in art_syms
            in_paragraph = asset in para_syms

            if in_article and in_paragraph:
                source = "both"
            elif in_article:
                source = "article_only"
            else:
                source = "paragraph_only"

            text = _build_validation_input(
                articles[art_idx],
                asset,
                article_reasoning=s1_reasonings[art_idx].get(asset),
                paragraph_reasonings=s2_reasonings[art_idx].get(asset, []),
            )
            tasks.append({
                "article_idx": art_idx,
                "asset": asset,
                "source": source,
            })
            input_texts.append(text)

    if not input_texts:
        logger.info("  No (article, asset) pairs to validate")
        return []

    logger.info("  %d (article, asset) pairs to validate", len(input_texts))

    # Batch validation call
    validation_results = mapper.map_validate(input_texts, max_tokens=256)

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
) -> tuple[list[MappingResult], list[MappingResult], list[list[tuple[int, MappingResult]]], list[dict]]:
    """
    Three-stage pipeline:
      Stage 1: article-level → themes, regions, assets, macro_summary
      Stage 2: paragraph-level with [CONTEXT] macro_summary → assets
      Stage 3: validate each (article, asset) pair + select text

    Returns (final_union_results, article_results, para_details, validation_output).
    """
    final_results, article_results, para_details = run_two_stage(mapper, articles)

    validation_output = run_validation(
        mapper, articles, article_results, para_details, final_results,
    )

    return final_results, article_results, para_details, validation_output



def _asset_label(sym: str) -> str:
    """Human-readable asset name for JSON output."""
    return ASSET_NAMES.get(sym, sym)


def save_final_results_json(
    articles: list[dict],
    article_results: list[MappingResult],
    para_details: list[list[tuple[int, MappingResult]]],
    final_union_results: list[MappingResult],
    validation_output: list[dict],
    out_path: Path,
) -> None:
    """Save full experiment results to a JSON file.

    Includes Stage 1 metadata, Stage 2 per-asset paragraph evidence, all
    (article, asset) pairs with accept/reject decisions, rejection reasoning,
    and per-article + aggregate false positive rates — so the output serves as
    both the final mapping and an evaluation dataset for diagnosing mapper
    weaknesses.
    """
    # Group validation results by article
    validation_by_article: dict[int, list[dict]] = {}
    for v in validation_output:
        validation_by_article.setdefault(v["article_idx"], []).append(v)

    # Build ArticleMapper reasoning lookup per article
    s1_reasoning_by_article: list[dict[str, str]] = []
    for ar in article_results:
        s1_reasoning_by_article.append({am.asset: am.reasoning for am in ar.relevant_assets})

    # Build ParagraphMapper reasoning + paragraph lookup per article
    s2_by_article: list[dict[str, list[tuple[int, str]]]] = []
    para_assets_by_article: list[dict[str, list[int]]] = []
    for art_idx in range(len(articles)):
        asset_para_reasonings: dict[str, list[tuple[int, str]]] = {}
        asset_paras: dict[str, list[int]] = {}
        for para_idx, r in para_details[art_idx]:
            for am in r.relevant_assets:
                asset_para_reasonings.setdefault(am.asset, []).append(
                    (para_idx, am.reasoning)
                )
                asset_paras.setdefault(am.asset, []).append(para_idx)
        s2_by_article.append(asset_para_reasonings)
        para_assets_by_article.append(asset_paras)

    total_accepted = 0
    total_rejected = 0
    final_output = []

    for art_idx, art in enumerate(articles):
        ar = article_results[art_idx]
        validations = validation_by_article.get(art_idx, [])
        s1_reas = s1_reasoning_by_article[art_idx]
        s2_reas = s2_by_article[art_idx]
        para_asset_map = para_assets_by_article[art_idx]

        # Stage 3 pairs: accepted and rejected
        accepted = []
        rejected = []
        referenced_paras: set[int] = set()
        for v in validations:
            evidence = v["evidence_paragraphs"]
            # Backfill from Stage 2 if validator returned empty evidence
            if not evidence:
                evidence = para_asset_map.get(v["asset"], [])
            # Collect mapper reasoning for diagnostics
            s1_r = s1_reas.get(v["asset"], "")
            s2_entries = s2_reas.get(v["asset"], [])
            s2_r_list = [
                {"paragraph": idx, "reasoning": r}
                for idx, r in s2_entries if r
            ]
            entry = {
                "asset": _asset_label(v["asset"]),
                "source": v["source"],
                "evidence_paragraphs": evidence,
                "mapper_reasoning": {
                    "article_mapper": s1_r,
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

        # Collect all referenced paragraphs from accepted/rejected entries
        for sym, paras in para_asset_map.items():
            referenced_paras.update(paras)

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
            "company_specific": ar.company_specific,
            "macro_summary": ar.macro_summary,
            "false_positive_rate": round(n_rejected / n_total, 3) if n_total else 0.0,
            "paragraphs": para_dict,
            "accepted": accepted,
            "rejected": rejected,
        }
        final_output.append(article_entry)

    # Aggregate stats
    grand_total = total_accepted + total_rejected

    # Per-asset-class and per-source breakdowns
    ac_accepted: dict[str, int] = {}
    ac_rejected: dict[str, int] = {}
    src_accepted: dict[str, int] = {}
    src_rejected: dict[str, int] = {}

    for v in validation_output:
        sym = v["asset"]
        ac = _ASSET_UNIVERSE.get(sym, {}).get("asset_class", "unknown")
        source = v["source"]
        if v["valid"]:
            ac_accepted[ac] = ac_accepted.get(ac, 0) + 1
            src_accepted[source] = src_accepted.get(source, 0) + 1
        else:
            ac_rejected[ac] = ac_rejected.get(ac, 0) + 1
            src_rejected[source] = src_rejected.get(source, 0) + 1

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
            1 for ar in article_results if ar.company_specific
        ),
        "by_asset_class": by_asset_class,
        "by_source": by_source,
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

    final_results, article_results, para_details, validation_output = run_three_stage(
        mapper, articles,
    )
    if output_json:
        save_final_results_json(
            articles, article_results, para_details,
            final_results, validation_output, output_json,
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
