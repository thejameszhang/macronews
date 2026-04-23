"""
Macronews two-stage LLM pipeline for article-to-asset tagging (v12).

Stage 1: article-level mapping — per-asset relevance + evidence paragraphs
Stage 2: validation — enforce rules, select final evidence paragraphs
"""

import argparse
import json
import logging
from pathlib import Path

from collections import defaultdict

from config.paths import PROMPTS_DIR, DEFAULT_MODEL
from experiments.loaders import load_gold_articles, load_sports_articles, load_wikigaming_articles, load_djnw_articles
from mapping.llm import (
    ASSET_CLASS_DISQUALIFIERS_PLACEHOLDER,
    ASSET_CLASS_POSITIVES_PLACEHOLDER,
    LLMMapper,
)
from mapping.schemas import SingleAssetResult, ValidationResult
from utils.config import load_asset_universe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ARTICLE_PROMPT = (PROMPTS_DIR / "single_asset.txt").read_text()
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
    start_date: str | None = None,
    end_date: str | None = None,
    random_seed: int | None = None,
    max_tokens: int | None = None,
    tokenizer_path: str | None = None,
    chars_per_token: float = 2.0,
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
        return load_sports_articles(sample_dir, max_articles=max_articles)
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
            chars_per_token=chars_per_token,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")



# ---------------------------------------------------------------------------
# Per-asset-class batching helper
# ---------------------------------------------------------------------------

def _run_per_asset_class(
    mapper: LLMMapper,
    prompt_template: str,
    syms: list[str],
    texts: list[str],
    call,
) -> list:
    """Group (sym, text) pairs by the asset's class, substitute class-specific
    rules into ``prompt_template``, and invoke ``call(mapper, batch_texts)``
    once per class. Returns results in the original input order.

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
    if len(syms) != len(texts):
        raise ValueError(f"syms/texts length mismatch: {len(syms)} vs {len(texts)}")

    by_class: dict[str, list[int]] = defaultdict(list)
    for idx, sym in enumerate(syms):
        ac = _ASSET_UNIVERSE[sym]["asset_class"]
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
# Stage 1: ArticleMapper (one call per article × asset)
# ---------------------------------------------------------------------------

def run_article_level(
    mapper: LLMMapper,
    articles: list[dict],
) -> list[dict[str, SingleAssetResult]]:
    """Stage 1: per-asset article-level mapping.

    Returns list (one per article) of {asset_sym: SingleAssetResult} for relevant assets only.
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
        for sym in ALL_ASSETS:
            # [ASSET] first so the model reads the article with the target in
            # mind (primes attention for localized relevance passages).
            text = f"[ASSET] {_asset_label(sym)}\n\n{article_text}"
            tasks.append((art_idx, sym))
            texts.append(text)

    logger.info("=== Stage 1: article-level (%d calls) ===", len(texts))
    results = _run_per_asset_class(
        mapper,
        ARTICLE_PROMPT,
        syms=[sym for _, sym in tasks],
        texts=texts,
        call=lambda m, ts: m.map_single_asset(ts, max_tokens=512),
    )

    by_article: list[dict[str, SingleAssetResult]] = [{} for _ in articles]
    for (art_idx, sym), sar in zip(tasks, results):
        if sar.relevant:
            by_article[art_idx][sym] = sar

    for art_idx, a in enumerate(articles):
        logger.info("  %s: %d assets flagged by ArticleMapper", a["id"], len(by_article[art_idx]))

    return by_article


# ---------------------------------------------------------------------------
# Stage 2: validation + text selection
# ---------------------------------------------------------------------------

def _build_validation_input(
    article: dict,
    asset: str,
    article_result: SingleAssetResult,
) -> str:
    """Build the user message for a validation call. [ASSET] first for the
    same attention-priming reason as the ArticleMapper input."""
    parts = [
        f"[ASSET] {_asset_label(asset)}",
        "",
        f"[HEADLINE] {article['headline']}",
        "[ARTICLE]",
    ]
    for idx, para in enumerate(article["paragraphs"]):
        parts.append(f"[{idx}] {para}")
    parts.append("[/ARTICLE]")
    parts.append("\n[MAPPER REASONING]")
    parts.append(
        f"ArticleMapper (signal={article_result.signal}, "
        f"relevance_score={article_result.relevance_score:.2f}, "
        f"evidence_paragraphs={article_result.evidence_paragraphs}): "
        f"{article_result.reasoning}"
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
) -> list[dict]:
    """Stage 2: validate each (article, asset) pair proposed by the ArticleMapper.

    Returns list of dicts with keys:
        article_idx, asset, valid, signal, relevance_score,
        evidence_paragraphs, reasoning, selected_text
    """
    logger.info("=== Stage 2: validation + text selection ===")

    tasks: list[dict] = []
    input_texts: list[str] = []

    for art_idx, article in enumerate(articles):
        for sym in sorted(article_results[art_idx].keys()):
            text = _build_validation_input(
                article, sym,
                article_result=article_results[art_idx][sym],
            )
            tasks.append({"article_idx": art_idx, "asset": sym})
            input_texts.append(text)

    if not input_texts:
        logger.info("  No (article, asset) pairs to validate")
        return []

    logger.info("  %d (article, asset) pairs to validate", len(input_texts))

    validation_results = _run_per_asset_class(
        mapper,
        VALIDATE_PROMPT,
        syms=[t["asset"] for t in tasks],
        texts=input_texts,
        call=lambda m, ts: m.map_validate(ts, max_tokens=512),
    )

    output = []
    for task, vr in zip(tasks, validation_results):
        art_idx = task["article_idx"]
        paragraphs = articles[art_idx]["paragraphs"]
        evidence = [i for i in vr.evidence_paragraphs if 0 <= i < len(paragraphs)]
        selected_text = build_selected_text(paragraphs, evidence) if vr.valid and evidence else ""
        output.append({
            "article_idx": art_idx,
            "asset": task["asset"],
            "valid": vr.valid,
            "signal": vr.signal,
            "relevance_score": vr.relevance_score,
            "evidence_paragraphs": evidence,
            "reasoning": vr.reasoning,
            "selected_text": selected_text,
        })

    n_valid = sum(1 for r in output if r["valid"])
    logger.info("  Validation complete: %d valid, %d rejected", n_valid, len(output) - n_valid)
    return output


def _synthesize_validation_from_mapper(
    article_results: list[dict[str, SingleAssetResult]],
) -> list[dict]:
    """Mapper-only mode: every relevant mapping is accepted as-is."""
    out: list[dict] = []
    for idx, am_map in enumerate(article_results):
        for asset, sar in am_map.items():
            if not sar.relevant:
                continue
            out.append({
                "article_idx": idx,
                "asset": asset,
                "valid": True,
                "signal": sar.signal,
                "relevance_score": sar.relevance_score,
                "evidence_paragraphs": list(sar.evidence_paragraphs),
                "reasoning": sar.reasoning,
            })
    return out


def run_pipeline(
    mapper: LLMMapper,
    articles: list[dict],
    *,
    use_validator: bool = True,
) -> tuple[list[dict[str, SingleAssetResult]], list[dict]]:
    """ArticleMapper, optionally followed by the validator.

    Returns (article_results, validation_output). When ``use_validator`` is
    False, validation_output is synthesized directly from the mapper (every
    relevant mapping is accepted), skipping the validator LLM pass.
    """
    article_results = run_article_level(mapper, articles)
    if use_validator:
        validation_output = run_validation(mapper, articles, article_results)
    else:
        logger.info("=== Stage 2: validation SKIPPED (mapper-only mode) ===")
        validation_output = _synthesize_validation_from_mapper(article_results)
        logger.info("  %d mapper mappings accepted as-is", len(validation_output))
    return article_results, validation_output



def _asset_display_name(sym: str) -> str:
    """Human-readable asset name for JSON output."""
    return ASSET_NAMES.get(sym, sym)


def save_final_results_json(
    articles: list[dict],
    article_results: list[dict[str, SingleAssetResult]],
    validation_output: list[dict],
    out_path: Path,
) -> None:
    """Save full experiment results to a JSON file."""
    validation_by_article: dict[int, list[dict]] = {}
    for v in validation_output:
        validation_by_article.setdefault(v["article_idx"], []).append(v)

    total_accepted = 0
    total_rejected = 0
    final_output = []

    for art_idx, art in enumerate(articles):
        am_map = article_results[art_idx]
        validations = validation_by_article.get(art_idx, [])

        accepted = []
        rejected = []
        referenced_paras: set[int] = set()
        for v in validations:
            am_sar = am_map.get(v["asset"])
            am_evidence = list(am_sar.evidence_paragraphs) if am_sar else []
            # If the validator returned no evidence, fall back to the
            # ArticleMapper's evidence_paragraphs.
            evidence = v["evidence_paragraphs"] or am_evidence
            am_score = am_sar.relevance_score if am_sar else 0.0
            am_signal = am_sar.signal if am_sar else ""
            am_reasoning = am_sar.reasoning if am_sar else ""
            entry = {
                "asset": _asset_display_name(v["asset"]),
                "signal": v.get("signal", "strong"),
                "relevance_score": v.get("relevance_score", 0.0),
                "mapper_relevance_score": am_score,
                "evidence_paragraphs": evidence,
                "mapper_reasoning": {
                    "signal": am_signal,
                    "relevance_score": am_score,
                    "evidence_paragraphs": am_evidence,
                    "reasoning": am_reasoning,
                } if am_sar else "",
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

        # Also include any paragraphs the AM flagged but were not surfaced by the validator,
        # so readers can audit what the mapper saw.
        for sar in am_map.values():
            referenced_paras.update(sar.evidence_paragraphs)

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
            "filtered_reasons": art.get("filtered_reasons", []),
            "false_positive_rate": round(n_rejected / n_total, 3) if n_total else 0.0,
            "accepted_assets": [e["asset"] for e in accepted],
            "rejected_assets": [e["asset"] for e in rejected],
            "paragraphs": para_dict,
            "accepted": accepted,
            "rejected": rejected,
        }
        final_output.append(article_entry)

    grand_total = total_accepted + total_rejected

    # Aggregates: asset-class breakdown + mapper-signal acceptance rate.
    ac_accepted: dict[str, int] = {}
    ac_rejected: dict[str, int] = {}
    sig_accepted: dict[str, int] = {}
    sig_rejected: dict[str, int] = {}

    for v in validation_output:
        sym = v["asset"]
        art_idx = v["article_idx"]
        ac = _ASSET_UNIVERSE.get(sym, {}).get("asset_class", "unknown")
        am_sar = article_results[art_idx].get(sym)
        am_sig = am_sar.signal if am_sar else None
        bucket = sig_accepted if v["valid"] else sig_rejected
        if v["valid"]:
            ac_accepted[ac] = ac_accepted.get(ac, 0) + 1
        else:
            ac_rejected[ac] = ac_rejected.get(ac, 0) + 1
        if am_sig:
            key = f"am_{am_sig}"
            bucket[key] = bucket.get(key, 0) + 1

    all_classes = sorted(set(ac_accepted) | set(ac_rejected))
    by_asset_class = {}
    for ac in all_classes:
        a = ac_accepted.get(ac, 0)
        r = ac_rejected.get(ac, 0)
        t = a + r
        by_asset_class[ac] = {
            "accepted": a, "rejected": r, "total": t,
            "false_positive_rate": round(r / t, 3) if t else 0.0,
        }

    # Count filtered articles per-reason. An article matching multiple reasons
    # contributes to each reason's count (so reason counts sum to >= filtered_articles).
    reason_counts: dict[str, int] = {}
    n_filtered = 0
    for a in articles:
        reasons = a.get("filtered_reasons") or []
        if reasons:
            n_filtered += 1
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1

    aggregate = {
        "total_articles": len(articles),
        "filtered_articles": n_filtered,
        "filtered_by_reason": reason_counts,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "total_pairs": grand_total,
        "false_positive_rate": round(total_rejected / grand_total, 3) if grand_total else 0.0,
        "by_asset_class": by_asset_class,
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "articles": final_output}, f, indent=2, ensure_ascii=False)
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
    start_date: str | None = None,
    end_date: str | None = None,
    random_seed: int | None = None,
    use_validator: bool = True,
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
    )

    mapper = LLMMapper(
        model_path=model_path,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )

    article_results, validation_output = run_pipeline(mapper, articles, use_validator=use_validator)
    if output_json:
        save_final_results_json(articles, article_results, validation_output, output_json)


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
        choices=["gold", "sports", "wikigaming", "djnw"],
        help="Dataset to run: 'gold', 'sports', 'wikigaming', or 'djnw' (default: gold)",
    )
    parser.add_argument(
        "--max-articles", type=int, default=None,
        help="Limit number of articles to process (useful for large datasets)",
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Earliest monthly file to include (YYYY-MM, djnw only)",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="Latest monthly file to include, inclusive (YYYY-MM, djnw only)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=None,
        help="Seed for random sampling of djnw articles within the date range",
    )
    parser.add_argument(
        "--no-validator", action="store_true",
        help="Skip the validator LLM pass; accept every relevant mapper mapping as-is",
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
        elif args.dataset == "djnw":
            args.sample_dir = "/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles"
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
        start_date=args.start_date,
        end_date=args.end_date,
        random_seed=args.random_seed,
        use_validator=not args.no_validator,
    )



if __name__ == "__main__":
    main()
