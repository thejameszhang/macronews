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
from mapping.schemas import MappingResult, ValidationResult
from utils.config import load_asset_universe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ARTICLE_PROMPT = (PROMPTS_DIR / "article.txt").read_text()
PARAGRAPH_PROMPT = (PROMPTS_DIR / "paragraph.txt").read_text()
VALIDATE_PROMPT = (PROMPTS_DIR / "validate.txt").read_text()
VALIDATE_SUMMARY_PROMPT = (PROMPTS_DIR / "validate_summary.txt").read_text()

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


def _fmt_assets(assets: set[str] | list[str]) -> str:
    s = sorted(assets) if isinstance(assets, set) else list(assets)
    return ",".join(s) if s else "[]"


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
    return mapper.map(texts, max_tokens=1024, guided=False)


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
    para_raw = mapper.map([text for _, _, text in all_paras])

    # Re-group by article
    para_by_article: dict[int, list[tuple[int, MappingResult]]] = {}
    for (art_idx, para_idx, _), result in zip(all_paras, para_raw):
        para_by_article.setdefault(art_idx, []).append((para_idx, result))

    # Aggregate: union of assets across paragraphs
    union_results: list[MappingResult] = []
    details: list[list[tuple[int, MappingResult]]] = []
    for art_idx in range(len(articles)):
        art_details = para_by_article.get(art_idx, [])
        details.append(art_details)
        seen: set[str] = set()
        union: list[str] = []
        for _, r in art_details:
            for sym in r.relevant_assets:
                if sym not in seen:
                    union.append(sym)
                    seen.add(sym)
        union_results.append(MappingResult(relevant_assets=sorted(union)))

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
                macro_themes=article_results[art_idx].macro_themes,
                regions=article_results[art_idx].regions,
                relevant_assets=[],
                company_specific=True,
                macro_summary="",
            ))
            continue
        art_assets = set(article_results[art_idx].relevant_assets)
        para_assets = set(para_results[art_idx].relevant_assets)
        union = sorted(art_assets | para_assets)
        final_results.append(MappingResult(
            macro_themes=article_results[art_idx].macro_themes,
            regions=article_results[art_idx].regions,
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
    macro_summary: str,
    para_indices: list[int] | None,
) -> str:
    """Build the user message for a Stage 3 validation call."""
    parts = []

    # Context block
    if macro_summary:
        parts.append(f"[CONTEXT]\n{macro_summary}\n[/CONTEXT]")

    # Paragraphs block
    parts.append("[PARAGRAPHS]")
    if para_indices is not None:
        for idx in para_indices:
            if idx < len(article["paragraphs"]):
                parts.append(f"[{idx}] {article['paragraphs'][idx]}")
    else:
        # Case 2: full article (no specific paragraphs identified)
        for idx, para in enumerate(article["paragraphs"]):
            parts.append(f"[{idx}] {para}")
    parts.append("[/PARAGRAPHS]")

    # Asset
    asset_name = ASSET_NAMES.get(asset, asset)
    parts.append(f"\n[ASSET] {asset} ({asset_name})")

    return "\n".join(parts)


def build_selected_text(
    paragraphs: list[str],
    evidence_indices: list[int],
) -> str:
    """Construct the text to embed for a validated (article, asset) pair.

    Only includes the raw evidence paragraphs — no LLM-generated rationale.
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
    generate_summary: bool = False,
) -> list[dict]:
    """
    Stage 3: validate each (article, asset) pair and select text.

    If generate_summary=True, the LLM generates a selected_text summary for
    each valid pair (for downstream embedding). If False, selected_text is
    constructed programmatically from evidence paragraphs + rationale.

    Returns list of dicts with keys:
        article_idx, asset, valid, source, evidence_paragraphs,
        rationale, selected_text
    """
    logger.info("=== Stage 3: validation + text selection (generate_summary=%s) ===",
                generate_summary)
    mapper.system_prompt = VALIDATE_SUMMARY_PROMPT if generate_summary else VALIDATE_PROMPT

    # Build paragraph-to-asset index from Stage 2
    para_assets_by_article: list[dict[str, list[int]]] = []
    for art_idx in range(len(articles)):
        asset_paras: dict[str, list[int]] = {}
        for para_idx, r in para_details[art_idx]:
            for sym in r.relevant_assets:
                asset_paras.setdefault(sym, []).append(para_idx)
        para_assets_by_article.append(asset_paras)

    # Enumerate all (article, asset) pairs and classify cases
    tasks: list[dict] = []
    input_texts: list[str] = []

    for art_idx, final_r in enumerate(final_union_results):
        art_assets = set(article_results[art_idx].relevant_assets)
        para_asset_map = para_assets_by_article[art_idx]
        macro_summary = article_results[art_idx].macro_summary

        for asset in final_r.relevant_assets:
            in_article = asset in art_assets
            in_paragraph = asset in para_asset_map

            if in_article and in_paragraph:
                source = "both"
                para_indices = para_asset_map[asset]
            elif in_article and not in_paragraph:
                source = "article_only"
                para_indices = None  # full article
            else:
                source = "paragraph_only"
                para_indices = para_asset_map.get(asset, [])

            text = _build_validation_input(
                articles[art_idx], asset, macro_summary, para_indices,
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
    max_tok = 512 if generate_summary else 256
    validation_results = mapper.map_validate(
        input_texts, max_tokens=max_tok, generate_summary=generate_summary,
    )

    # Combine results
    output = []
    for task, vr in zip(tasks, validation_results):
        art_idx = task["article_idx"]
        paragraphs = articles[art_idx]["paragraphs"]

        # Range-check evidence_paragraphs
        evidence = [i for i in vr.evidence_paragraphs if 0 <= i < len(paragraphs)]

        selected_text = ""
        if vr.valid and evidence:
            if generate_summary and hasattr(vr, "selected_text") and vr.selected_text:
                selected_text = vr.selected_text
            else:
                selected_text = build_selected_text(paragraphs, evidence)

        output.append({
            "article_idx": art_idx,
            "asset": task["asset"],
            "valid": vr.valid,
            "source": task["source"],
            "evidence_paragraphs": evidence,
            "rationale": vr.rationale if vr.valid else "",
            "selected_text": selected_text,
        })

    n_valid = sum(1 for r in output if r["valid"])
    n_invalid = len(output) - n_valid
    logger.info("  Validation complete: %d valid, %d rejected", n_valid, n_invalid)

    return output


def run_three_stage(
    mapper: LLMMapper,
    articles: list[dict],
    generate_summary: bool = False,
) -> tuple[list[MappingResult], list[MappingResult], list[list[tuple[int, MappingResult]]], list[dict]]:
    """
    Three-stage pipeline:
      Stage 1: article-level → themes, regions, assets, macro_summary
      Stage 2: paragraph-level with [CONTEXT] macro_summary → assets
      Stage 3: validate each (article, asset) pair + select text

    If generate_summary=True, Stage 3 asks the LLM to write a summary for
    each valid pair (for downstream embedding). Otherwise, text is constructed
    programmatically from evidence paragraphs + rationale.

    Returns (final_union_results, article_results, para_details, validation_output).
    """
    final_results, article_results, para_details = run_two_stage(mapper, articles)

    validation_output = run_validation(
        mapper, articles, article_results, para_details, final_results,
        generate_summary=generate_summary,
    )

    return final_results, article_results, para_details, validation_output


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

def print_results(
    articles: list[dict],
    final_results: list[MappingResult],
    article_results: list[MappingResult],
    para_details: list[list[tuple[int, MappingResult]]],
    validation_output: list[dict] | None = None,
) -> None:
    """Print results tables."""

    # Dynamic ID width — truncate long IDs for readability
    id_w = max(10, min(40, max(len(a["id"]) for a in articles) + 2))

    def _aid(a: dict) -> str:
        """Article ID, truncated if needed."""
        aid = a["id"]
        return aid[:id_w - 2] + ".." if len(aid) > id_w else aid

    # --- Table 1: Article-level output ---
    print(f"\n{'=' * (id_w + 145)}")
    print("TABLE 1: Article-Level Output (Stage 1)")
    print("=" * (id_w + 145))
    print(f"  {'ID':<{id_w}} {'Themes':<35} {'Regions':<20} {'Assets':<25} {'Co.Spec':<8} {'Summary':<50}")
    print("  " + "-" * (id_w + 138))
    for i, a in enumerate(articles):
        r = article_results[i]
        themes = ",".join(r.macro_themes)
        regions = ",".join(r.regions)
        assets = ",".join(r.relevant_assets) if r.relevant_assets else "[]"
        co_spec = "YES" if r.company_specific else ""
        summary = r.macro_summary[:50] if r.macro_summary else ""
        print(f"  {_aid(a):<{id_w}} {themes:<35} {regions:<20} {assets:<25} {co_spec:<8} {summary:<50}")

    # --- Table 2: Final results (union) ---
    print(f"\n{'=' * (id_w + 60)}")
    print("TABLE 2: Final Results (Stage 1 + Stage 2 Union)")
    print("=" * (id_w + 60))
    print(f"  {'ID':<{id_w}} {'Assets':<50} {'Count':>6}")
    print("  " + "-" * (id_w + 56))
    total = 0
    for i, a in enumerate(articles):
        assets = _fmt_assets(final_results[i].relevant_assets)
        n = len(final_results[i].relevant_assets)
        total += n
        print(f"  {_aid(a):<{id_w}} {assets:<50} {n:>6}")
    print(f"  {'TOTAL':<{id_w}} {'':<50} {total:>6}")
    print(f"  {'AVG':<{id_w}} {'':<50} {total / len(articles):>6.1f}")

    # --- Table 3: Stage 1 vs Stage 2 differences ---
    print(f"\n{'=' * (id_w + 65)}")
    print("TABLE 3: Stage 1 (article) vs Stage 2 (paragraph) Differences")
    print("=" * (id_w + 65))
    print(f"  {'ID':<{id_w}} {'Only in Stage 1':<30} {'Only in Stage 2':<30}")
    print("  " + "-" * (id_w + 60))
    for i, a in enumerate(articles):
        s1 = set(article_results[i].relevant_assets)
        s2_union: set[str] = set()
        for _, r in para_details[i]:
            s2_union.update(r.relevant_assets)
        only_s1 = s1 - s2_union
        only_s2 = s2_union - s1
        if only_s1 or only_s2:
            print(f"  {_aid(a):<{id_w}} {_fmt_assets(only_s1):<30} {_fmt_assets(only_s2):<30}")

    # --- Table 4: Paragraph drill-down ---
    print(f"\n{'=' * 120}")
    print("TABLE 4: Paragraph Drill-Down (Stage 2)")
    print("=" * 120)
    for i, a in enumerate(articles):
        has_any = any(r.relevant_assets for _, r in para_details[i])
        if not has_any:
            continue
        print(f"\n  --- {_aid(a)}: {a['headline'][:70]} ---")
        print(f"  {'Para':<6} {'Themes':<30} {'Regions':<15} {'Assets':<20} {'Text':<50}")
        print("  " + "-" * 121)
        for para_idx, result in para_details[i]:
            if not result.relevant_assets:
                continue
            para_text = a["paragraphs"][para_idx][:50]
            themes = ",".join(result.macro_themes)
            regions = ",".join(result.regions)
            assets = ",".join(result.relevant_assets)
            print(f"  {para_idx:<6} {themes:<30} {regions:<15} {assets:<20} {para_text:<50}")

    # --- Table 5: Macro summaries ---
    if any(r.macro_summary for r in article_results):
        print(f"\n{'=' * 130}")
        print("TABLE 5: Macro Summaries (Stage 1)")
        print("=" * 130)
        for i, a in enumerate(articles):
            r = article_results[i]
            if r.macro_summary:
                print(f"\n  {_aid(a)}: {r.macro_summary}")
            else:
                print(f"\n  {_aid(a)}: (no macro content)")

    # --- Table 6: Validation results (Stage 3) ---
    if validation_output:
        print(f"\n{'=' * (id_w + 130)}")
        print("TABLE 6: Validation Results (Stage 3)")
        print("=" * (id_w + 130))
        print(f"  {'ID':<{id_w}} {'Asset':<8} {'Valid':<7} {'Source':<16} {'Evidence':<12} {'Rationale':<80}")
        print("  " + "-" * (id_w + 123))
        for v in validation_output:
            aid = _aid(articles[v["article_idx"]])
            valid_str = "YES" if v["valid"] else "NO"
            evidence = ",".join(str(i) for i in v["evidence_paragraphs"]) if v["evidence_paragraphs"] else "[]"
            rationale = v["rationale"][:80] if v["rationale"] else ""
            print(f"  {aid:<{id_w}} {v['asset']:<8} {valid_str:<7} {v['source']:<16} {evidence:<12} {rationale}")

        # Summary
        n_valid = sum(1 for v in validation_output if v["valid"])
        n_total = len(validation_output)
        print(f"\n  Stage 3 summary: {n_valid}/{n_total} pairs validated "
              f"({n_total - n_valid} rejected)")

        # Validated asset list per article
        print(f"\n{'=' * (id_w + 65)}")
        print("TABLE 7: Final Validated Assets (after Stage 3)")
        print("=" * (id_w + 65))
        print(f"  {'ID':<{id_w}} {'Assets':<50} {'Count':>6}  {'Pruned':>6}")
        print("  " + "-" * (id_w + 63))
        total_final = 0
        total_pruned = 0
        for i, a in enumerate(articles):
            pre_assets = set(final_results[i].relevant_assets)
            post_assets = {v["asset"] for v in validation_output
                          if v["article_idx"] == i and v["valid"]}
            pruned = pre_assets - post_assets
            assets_str = _fmt_assets(post_assets)
            total_final += len(post_assets)
            total_pruned += len(pruned)
            pruned_str = _fmt_assets(pruned) if pruned else ""
            print(f"  {_aid(a):<{id_w}} {assets_str:<50} {len(post_assets):>6}  {pruned_str}")
        print(f"  {'TOTAL':<{id_w}} {'':<50} {total_final:>6}  ({total_pruned} pruned)")
        print(f"  {'AVG':<{id_w}} {'':<50} {total_final / len(articles):>6.1f}")


def save_final_results_json(
    articles: list[dict],
    validation_output: list[dict],
    out_path: Path,
) -> None:
    """Save the final validated mappings to a JSON file."""
    # Group by article_idx
    valid_mappings_by_article = {}
    for v in validation_output:
        if v["valid"]:
            art_idx = v["article_idx"]
            valid_mappings_by_article.setdefault(art_idx, []).append(v)

    final_output = []
    # Iterate through all input articles to ensure every one is represented
    for art_idx, art in enumerate(articles):
        mappings = valid_mappings_by_article.get(art_idx, [])
        
        article_entry = {
            "article_id": art["id"],
            "headline": art["headline"],
            "url": art.get("url", ""),
            "mappings": []
        }
        for m in mappings:
            asset = m["asset"]
            article_entry["mappings"].append({
                "asset": asset,
                "asset_name": ASSET_NAMES.get(asset, asset),
                "rationale": m.get("rationale", ""),
                "selected_text": m["selected_text"]
            })
        final_output.append(article_entry)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved final results for {len(final_output)} articles to {out_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_experiment(
    model_path: str,
    max_model_len: int,
    sample_dir: Path,
    output_json: Path | None = None,
    tensor_parallel_size: int = 1,
    generate_summary: bool = False,
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
        mapper, articles, generate_summary=generate_summary,
    )
    print_results(articles, final_results, article_results, para_details, validation_output)
    
    if output_json:
        save_final_results_json(articles, validation_output, output_json)


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
        "--generate-summary", action="store_true",
        help="Have the LLM generate selected_text summaries in Stage 3 "
        "(default: construct programmatically from evidence + rationale)",
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
        generate_summary=args.generate_summary,
        dataset=args.dataset,
        max_articles=args.max_articles,
    )



if __name__ == "__main__":
    main()
