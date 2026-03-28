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

    Strategy:
      1. If the text has newline-separated paragraphs (>1 meaningful block),
         use those directly.
      2. Otherwise, split by sentences and group into chunks of `sents_per_para`.
    """
    # Try newline splitting first
    blocks = [b.strip() for b in text.split("\n") if b.strip()]
    meaningful = [b for b in blocks if len(b) >= _MIN_PARA_LEN]

    if len(meaningful) > 1:
        return meaningful

    # Single block — split by sentences
    full_text = " ".join(blocks)  # rejoin in case there were short fragments
    sentences = _SENT_SPLIT.split(full_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= sents_per_para:
        return [full_text] if full_text else []

    # Group sentences into paragraphs
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
