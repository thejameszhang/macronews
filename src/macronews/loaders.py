"""
Article loaders for the ArticleMapper pipeline.

Each loader returns a list[dict] with the standard schema:
    {"id": str, "headline": str, "paragraphs": list[str], ...}

Extra keys (url, date, source, etc.) are preserved but not required
by the pipeline.
"""

import json
import logging
import re
from pathlib import Path

from macronews.config.paths import DJNW_ARTICLES_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------

# Sentence boundary: period/question/exclamation followed by whitespace + uppercase letter.
# Avoids splitting on abbreviations like "U.S." or "Dr." or decimal numbers.
_SENT_SPLIT = re.compile(
    r'(?<=[.!?])'        # lookbehind: sentence-ending punctuation
    r'(?:\s+)'           # whitespace between sentences
    r'(?=[A-Z"\u201c])'  # lookahead: next sentence starts with uppercase or opening quote
)

# Minimum characters for a paragraph to be meaningful
_MIN_PARA_LEN = 30
# Target sentences per paragraph when splitting a single text block
_SENTS_PER_PARA = 3
# A block whose mean line length is below this is treated as hard-wrapped at
# ~80 cols (single \n is line wrapping, not a paragraph break). DJ wire format
# wraps at column 80; modern paragraph-formatted text averages well above this.
_HARDWRAP_MEAN_THRESHOLD = 80


def clean_web_text(text: str) -> str:
    """Remove web navigation cruft from scraped article text.

    Strips lines that are likely menus, table markup, or boilerplate
    rather than article prose.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip pipe-heavy lines (navigation tables, scoreboards)
        if stripped.count("|") > 2:
            continue
        # Skip very short fragments that are likely UI elements
        if len(stripped) < 15 and not stripped[-1:] in ".!?\"'":
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


def _normalize_whitespace(s: str) -> str:
    """Collapse any run of whitespace (spaces, tabs, CR, unicode separators,
    etc.) to a single space and strip ends. Defends against residual whitespace
    inside a paragraph string being read by the LLM as a paragraph break."""
    return re.sub(r"\s+", " ", s).strip()


def _group_sentences(text: str, sents_per_para: int = _SENTS_PER_PARA) -> list[str]:
    """Bin sentences in groups of ``sents_per_para`` to form paragraphs.
    Used when a block has no usable paragraph structure of its own."""
    if not text:
        return []
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sentences) <= sents_per_para:
        return [text]
    paragraphs = []
    for i in range(0, len(sentences), sents_per_para):
        chunk = " ".join(sentences[i : i + sents_per_para])
        if chunk:
            paragraphs.append(chunk)
    return paragraphs


def split_into_paragraphs(text: str, sents_per_para: int = _SENTS_PER_PARA) -> list[str]:
    """Split a text block into paragraphs robust to DJNW formatting variants.

    Three text shapes appear in the cleaned-v2 corpus:
      A. Modern paragraph format: each paragraph is its own line, separated
         by single ``\\n`` (some articles also have a stray ``\\n\\n`` between
         byline and body).
      B. Hard-wrapped wire format: ``\\n`` every ~80 chars within a paragraph,
         ``\\n\\n`` between paragraphs.
      C. Run-on block with no newlines: split by sentences.

    Algorithm:
      1. ``\\n\\n`` is *always* a paragraph break (explicit editorial signal).
         Split text on ``\\n\\n`` first to get blocks.
      2. For each block, decide whether internal ``\\n``s are paragraph breaks
         or hard-wrap based on the mean line length within that block:
           - mean < 80 chars → hard-wrap; unwrap to a flat string and
             sentence-group (handles shape B and shape C with no ``\\n\\n``).
           - mean >= 80 chars → each ``\\n`` is a paragraph break (handles
             shape A); keep the lines as separate paragraphs.
      3. Drop paragraphs below ``_MIN_PARA_LEN`` chars.
      4. Normalize whitespace within each remaining paragraph so the LLM
         doesn't see residual ``\\t``/``\\r``/Unicode separators that would
         confuse its mental paragraph numbering.
      5. Reproducibility guarantee: any non-whitespace input text yields at
         least one paragraph. If the MIN_PARA_LEN filter would empty the
         result, fall back to the whole-text normalization. This keeps the
         loader's seeded sampling stable across splitter changes — the
         downstream ``unembeddable`` filter can still flag short bodies.

    Per-block routing (rather than a single global decision) is needed
    because some articles mix shapes — e.g. a hard-wrapped table block
    inside an otherwise paragraph-formatted article.
    """
    if not text or not text.strip():
        return []

    # Step 1: \n\n is always a paragraph break.
    blocks = text.split("\n\n") if "\n\n" in text else [text]

    result: list[str] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Step 2: split the block on single \n; route by mean line length.
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue
        if len(lines) == 1:
            result.append(lines[0])
            continue

        mean_line_len = sum(len(l) for l in lines) / len(lines)
        if mean_line_len < _HARDWRAP_MEAN_THRESHOLD:
            # Hard-wrap within this block: unwrap, then sentence-group.
            unwrapped = " ".join(lines)
            result.extend(_group_sentences(unwrapped, sents_per_para))
        else:
            # Each line is a paragraph in its own right.
            result.extend(lines)

    # Step 3+4: filter tiny fragments and normalize internal whitespace.
    paragraphs = [
        _normalize_whitespace(p)
        for p in result
        if len(p) >= _MIN_PARA_LEN
    ]

    # Step 5: never drop a non-empty article. If everything was below
    # MIN_PARA_LEN, return the whole text normalized as one paragraph.
    if not paragraphs:
        fallback = _normalize_whitespace(text)
        return [fallback] if fallback else []
    return paragraphs


# ---------------------------------------------------------------------------
# Gold articles (data/articles_sample/gold_*.json)
# ---------------------------------------------------------------------------

def load_gold_articles(sample_dir: Path) -> list[dict]:
    """Load gold_*.json files sorted by id."""
    files = sorted(sample_dir.glob("gold_*.json"))
    if not files:
        raise FileNotFoundError(f"No gold_*.json files in {sample_dir}")
    articles = []
    for f in files:
        with open(f) as fh:
            articles.append(json.load(fh))
    logger.info("Loaded %d gold articles from %s", len(articles), sample_dir)
    return articles


# ---------------------------------------------------------------------------
# Sports news (data/sports_news_1994_2000/YYYY/*.json)
# ---------------------------------------------------------------------------

def load_sports_articles(
    data_dir: Path,
    max_articles: int | None = None,
    year: int | None = None,
    max_tokens: int | None = None,
    tokenizer_path: str | None = None,
    chars_per_token: float = 2.0,
) -> list[dict]:
    """Load sports news articles, converting to the standard schema.

    Parameters
    ----------
    data_dir : Path
        Root directory (e.g., data/sports_news_1994_2000).
    max_articles : int, optional
        Limit number of articles loaded.
    year : int, optional
        Load only articles from a specific year subdirectory.
    max_tokens : int, optional
        Skip articles whose text exceeds this token count (same filter as the
        djnw loader, so an over-long article can't blow the model context).
    tokenizer_path : str, optional
        Model path for the ``max_tokens`` filter's tokenizer.
    chars_per_token : float, optional
        Conservative lower bound on chars-per-token for the fast-path filter.
    """
    if year is not None:
        year_dir = data_dir / str(year)
        if not year_dir.exists():
            raise FileNotFoundError(f"Year directory not found: {year_dir}")
        files = sorted(year_dir.glob("*.json"))
    else:
        files = sorted(data_dir.rglob("*.json"))

    # Exclude summary.json
    files = [f for f in files if f.name != "summary.json"]

    if not files:
        raise FileNotFoundError(f"No article JSON files in {data_dir}")

    if max_articles is not None:
        files = files[:max_articles]

    # Lazy tokenizer for the optional length filter (mirrors load_djnw_articles).
    tokenizer = None
    fast_path_limit_chars = int(max_tokens * chars_per_token) if max_tokens else None
    skipped_long = 0

    articles = []
    for f in files:
        with open(f) as fh:
            raw = json.load(fh)

        text = clean_web_text(raw["text"])

        # Token-length filter. Fast path: text under chars_per_token*max_tokens
        # chars cannot exceed max_tokens. Slow path: tokenize and check exactly.
        if max_tokens is not None and len(text) > fast_path_limit_chars:
            if tokenizer is None:
                from transformers import AutoTokenizer
                if tokenizer_path is None:
                    raise ValueError("tokenizer_path required when max_tokens is set")
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if len(tokenizer.encode(text, add_special_tokens=False)) > max_tokens:
                skipped_long += 1
                continue

        paragraphs = split_into_paragraphs(text)
        if not paragraphs:
            logger.warning("Skipping %s — empty text", f.name)
            continue

        articles.append({
            "id": f.stem,
            "headline": raw.get("title", ""),
            "paragraphs": paragraphs,
            # Preserve extra metadata
            "date": raw.get("date", ""),
            "source": raw.get("source", ""),
            "sport": raw.get("sport", ""),
            "url": raw.get("url", ""),
        })

    logger.info(
        "Loaded %d sports articles from %s%s",
        len(articles), data_dir,
        f" (year={year})" if year else "",
    )
    if skipped_long:
        logger.info("Skipped %d sports articles exceeding %d tokens", skipped_long, max_tokens)
    return articles


# ---------------------------------------------------------------------------
# Wiki Gaming (data/WikiGaming.jsonl)
# ---------------------------------------------------------------------------

def load_wikigaming_articles(
    data_path: Path,
    max_articles: int | None = None,
) -> list[dict]:
    """Load WikiGaming JSONL articles, converting to the standard schema.

    Parameters
    ----------
    data_path : Path
        Path to the .jsonl file (e.g., data/WikiGaming.jsonl).
    max_articles : int, optional
        Limit number of articles loaded.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"WikiGaming file not found: {data_path}")

    articles = []
    with open(data_path) as fh:
        for i, line in enumerate(fh):
            if max_articles is not None and len(articles) >= max_articles:
                break
            raw = json.loads(line)

            text = clean_web_text(raw["text"])
            paragraphs = split_into_paragraphs(text)
            if not paragraphs:
                logger.warning("Skipping line %d — empty text", i)
                continue

            articles.append({
                "id": f"wiki_{raw.get('page_id', i)}",
                "headline": raw.get("title", ""),
                "paragraphs": paragraphs,
                "url": raw.get("url", ""),
                "source": "wikipedia",
                "domain": raw.get("domain", ""),
            })

    logger.info("Loaded %d WikiGaming articles from %s", len(articles), data_path)
    return articles


# ---------------------------------------------------------------------------
# Dow Jones Newswires (rc2573/output/cleaned/v2/articles/YYYY-MM_clean.jsonl)
# ---------------------------------------------------------------------------

from djnw.read import load_djnw_articles


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


def load_as_production(shard: Path) -> list[dict]:
    """The exact article set `macronews mapper run` loads for this shard.

    Anything that measures the pipeline must measure the articles the pipeline actually
    sees. That means applying the LENGTH FILTER, which is not a detail: it follows from
    max_model_len, so a serving knob decides which articles exist. Loading without it
    (max_tokens=None) picks up 13 articles across 2014 that the real run never saw --
    enough to put a "corpus-wide" count 326 calls away from a scorecard of the same gate.

    Defined here, once, because both the corpus call-count and the recall scorecard need
    it and each got it wrong in its own way first.
    """
    from macronews.config.runconfig import MapperConfig   # local: runconfig imports paths

    # output_dir is unused -- MapperConfig requires exactly one output destination, and
    # this caller writes nothing. The alternative is a second config class for readers.
    cfg = MapperConfig(input_file=shard, output_dir=Path("/dev/null"))
    return load_djnw_articles(
        DJNW_ARTICLES_DIR,
        input_file=cfg.input_file,
        max_tokens=cfg.max_article_tokens,
        tokenizer_path=str(cfg.model),
        chars_per_token=2.0,
    )
