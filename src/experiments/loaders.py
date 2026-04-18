"""
Article loaders for the three-stage experiment.

Each loader returns a list[dict] with the standard schema:
    {"id": str, "headline": str, "paragraphs": list[str], ...}

Extra keys (url, date, source, etc.) are preserved but not required
by the pipeline.
"""

import json
import logging
import re
from pathlib import Path

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


def split_into_paragraphs(text: str, sents_per_para: int = _SENTS_PER_PARA) -> list[str]:
    """Split a text block into paragraphs.

    Handles three text shapes seen in the wild:
      A. Proper paragraph breaks (``\\n\\n`` between blocks, e.g. long DJNW filings
         or web-scraped articles).
      B. Hard-wrapped single-``\\n`` line breaks with no paragraph structure
         (many DJNW stories are pre-wrapped at ~80 cols — splitting on ``\\n``
         would give one "paragraph" per line).
      C. A single run-on block with no newlines at all (short DJNW headlines/
         correction notices) — split by sentences.

    Strategy:
      1. Prefer ``\\n\\n`` paragraph breaks when they exist. Within each block
         unwrap single-``\\n`` line wrapping.
      2. Otherwise, unwrap the whole text into one block.
      3. If that one block has enough sentences, group them into paragraphs;
         otherwise return it as a single paragraph.
    """
    def _unwrap(block: str) -> str:
        # Collapse single-newline line wrapping into spaces, preserving nothing.
        return re.sub(r"\s*\n\s*", " ", block).strip()

    # Step 1: prefer double-newline paragraph breaks
    if "\n\n" in text:
        blocks = [_unwrap(b) for b in text.split("\n\n")]
        meaningful = [b for b in blocks if len(b) >= _MIN_PARA_LEN]
        if len(meaningful) > 1:
            return meaningful

    # Step 2: unwrap the entire text into one block
    full_text = _unwrap(text)
    if not full_text:
        return []

    sentences = _SENT_SPLIT.split(full_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= sents_per_para:
        return [full_text]

    # Step 3: group sentences into paragraphs of size ``sents_per_para``
    paragraphs = []
    for i in range(0, len(sentences), sents_per_para):
        chunk = " ".join(sentences[i : i + sents_per_para])
        if chunk:
            paragraphs.append(chunk)
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
) -> list[dict]:
    """Load sports news articles, converting to the standard schema.

    Parameters
    ----------
    data_dir : Path
        Root directory (e.g., data/sports_news_1994_2000).
    max_articles : int, optional
        Limit number of articles loaded (for quick experiments).
    year : int, optional
        Load only articles from a specific year subdirectory.
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

    articles = []
    for f in files:
        with open(f) as fh:
            raw = json.load(fh)

        text = clean_web_text(raw["text"])
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

def load_djnw_articles(
    data_dir: Path,
    max_articles: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    random_seed: int | None = None,
    max_tokens: int | None = None,
    tokenizer_path: str | None = None,
    chars_per_token: float = 2.0,
) -> list[dict]:
    """Load DJNW articles from monthly JSONL files, converting to standard schema.

    Parameters
    ----------
    data_dir : Path
        Directory containing YYYY-MM_clean.jsonl files (e.g., .../cleaned/v2/articles).
    max_articles : int, optional
        Limit total articles loaded.
    start_date : str, optional
        Only load files from this YYYY-MM onward (e.g., "1990-01").
    end_date : str, optional
        Only load files up to this YYYY-MM inclusive (e.g., "1995-12").
    random_seed : int, optional
        If set, load ALL matching articles and randomly sample ``max_articles`` from
        them with this seed. Default is to stream files in order and stop at the
        first ``max_articles`` (which biases toward the earliest chunk).
    max_tokens : int, optional
        Skip articles whose text exceeds this many tokens (measured with the
        model tokenizer). Prevents vLLM from crashing on prompts that exceed
        the model's context window.
    tokenizer_path : str, optional
        HuggingFace model path (dir or repo id) used to load the tokenizer for
        ``max_tokens``. Required if ``max_tokens`` is set.
    """
    files = sorted(data_dir.glob("*_clean.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *_clean.jsonl files in {data_dir}")

    if start_date:
        files = [f for f in files if f.name[:7] >= start_date]
    if end_date:
        files = [f for f in files if f.name[:7] <= end_date]

    # When random sampling, disable the streaming early-break so we see every article.
    streaming_cap = max_articles if random_seed is None else None

    # Lazy tokenizer for length filtering. ``chars_per_token`` is a lower bound on
    # the tokenizer's char-to-token ratio --- a text shorter than
    # ``chars_per_token * max_tokens`` chars cannot possibly exceed ``max_tokens``
    # tokens, and skips tokenization entirely. Gemma tokenizes at ~3-4 chars/token
    # for English (default 2.0 is a safe conservative bound).
    tokenizer = None
    fast_path_limit_chars = int(max_tokens * chars_per_token) if max_tokens else None
    skipped_long = 0

    articles = []
    for f in files:
        if streaming_cap is not None and len(articles) >= streaming_cap:
            break
        with open(f) as fh:
            for line in fh:
                if streaming_cap is not None and len(articles) >= streaming_cap:
                    break
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)

                text = raw.get("text", "")
                if not text:
                    continue

                # Token-length filter. Fast path: char count ≥ 2× max_tokens
                # cannot possibly be under the limit. Slow path: tokenize and
                # check exactly.
                if max_tokens is not None and len(text) > fast_path_limit_chars:
                    if tokenizer is None:
                        from transformers import AutoTokenizer
                        if tokenizer_path is None:
                            raise ValueError("tokenizer_path required when max_tokens is set")
                        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                    n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                    if n_tokens > max_tokens:
                        skipped_long += 1
                        continue

                paragraphs = split_into_paragraphs(text)
                if not paragraphs:
                    continue

                # docdate is YYYYMMDD string
                docdate = raw.get("docdate", "")
                date_str = f"{docdate[:4]}-{docdate[4:6]}-{docdate[6:8]}" if len(docdate) == 8 else docdate

                articles.append({
                    "id": raw["accession_number"],
                    "headline": raw.get("headline", ""),
                    "paragraphs": paragraphs,
                    "date": date_str,
                    "source": "djnw",
                    # Preserve taxonomy codes — useful for asset pre-filtering
                    "codes": raw.get("codes", {}),
                })

    if random_seed is not None:
        import random
        rng = random.Random(random_seed)
        rng.shuffle(articles)
        if max_articles is not None and len(articles) > max_articles:
            articles = articles[:max_articles]

    if skipped_long:
        logger.info("Skipped %d DJNW articles exceeding %d tokens", skipped_long, max_tokens)
    logger.info("Loaded %d DJNW articles from %s", len(articles), data_dir)
    return articles
