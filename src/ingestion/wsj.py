"""
WSJIngestor — ingests article metadata from the WSJ public archive.

Phase 1 (headline-only): fetches https://www.wsj.com/news/archive/YYYY/MM/DD,
extracts the __NEXT_DATA__ JSON blob, and produces ArticleRecord Parquet files
with body_excerpt = headline (no authentication required).

Phase 2 (with body): pass cookie_file= pointing to a Netscape-format cookies.txt
exported from a logged-in WSJ session. Article pages are fetched in parallel
(max_workers threads) to extract standFirst + body paragraphs (≤ 1 200 chars).

See wsj.md for the full exploration log and design rationale.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .base import (
    AbstractIngestor,
    assign_trading_day,
    is_buffer_zone,
    load_trading_days,
)
from .schemas import MIN_WORD_COUNT_HEADLINE_ONLY

ET = ZoneInfo("America/New_York")


_EXCLUDED_SECTIONS: frozenset[str] = frozenset({
    "opinion",
    "arts-culture",
    "sports",
    "lifestyle",
    "health",
    "real-estate",
    "video",
})


_EXCLUDE_HEADLINE_RE = re.compile(
    r"(crossword"
    r"|tv\s+(sport|listing|schedule)"
    r"|comings\s+and\s+goings"
    r"|(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+puzzle"
    r"|pepper\b.*\bsalt\b"
    r"|notable\s*&\s*quotable"
    r"|\bwork\s+week\b"
    r"|travel\s+index"
    r"|(theater|theatre|concert|movie|film|book)\s+review"
    r"|\bobituari"
    r"|sports\s+(report|score|result)"
    r"|\bNFL\b|\bNBA\b|\bNHL\b|\bMLB\b"
    r")",
    re.IGNORECASE,
)

_ARTICLE_SLUG_RE = re.compile(r'-[0-9a-f]{8}$')
_UUID_RE = re.compile(r'^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$')

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_NEXT_DATA_RE = re.compile(
    r'"__NEXT_DATA__"[^>]*>(.*?)</script>',
    re.DOTALL,
)

_BODY_EXCERPT_MAX = 1200



def _load_cookie_dict(cookie_file: Path) -> dict[str, str]:
    """Parse a Netscape-format cookies.txt into a name→value dict."""
    cookies: dict[str, str] = {}
    with open(cookie_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                cookies[parts[5]] = parts[6]
    return cookies



def _fetch_archive_html(archive_date: date, timeout: int = 20) -> str:
    url = (
        f"https://www.wsj.com/news/archive/"
        f"{archive_date.year}/{archive_date.month:02d}/{archive_date.day:02d}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _parse_next_data_archive(html: str) -> list[dict]:
    match = _NEXT_DATA_RE.search(html)
    if not match:
        return []
    data = json.loads(match.group(1))
    return data["props"]["pageProps"].get("newsArchiveArticles", [])



def _extract_body_from_article_data(ad: dict) -> str:
    """
    Pull standFirst + body paragraphs out of articleData (__NEXT_DATA__).

    Returns combined plain text trimmed to _BODY_EXCERPT_MAX characters.
    """
    parts: list[str] = []

    # 1. standFirst — editor-written lede / subheadline
    sf_text = (
        ((ad.get("standFirst") or {}).get("content") or {}).get("text", "")
    ).strip()
    if sf_text:
        parts.append(sf_text)

    # 2. Body paragraphs (text tokens concatenated)
    for item in (ad.get("flattenedBody") or []):
        if item.get("type") != "paragraph":
            continue
        tokens = [
            node.get("text", "")
            for node in (item.get("content") or [])
            if node.get("text")
        ]
        text = "".join(tokens).strip()
        if text:
            parts.append(text)

    return " ".join(parts)


def _fetch_article_body(article_url: str, session, article_delay: float) -> str:
    """
    Fetch one article page and return its body excerpt.

    Uses the caller-supplied requests.Session so DataDome cookie updates
    (Set-Cookie response headers) persist within each thread's session.
    Returns empty string on any error.
    """
    try:
        resp = session.get(
            article_url,
            timeout=15,
            headers={
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.wsj.com/",
            },
        )
        if resp.status_code != 200:
            return ""
        html = resp.text
    except Exception:
        return ""
    finally:
        # Polite delay even on error; each thread sleeps independently
        time.sleep(article_delay + random.uniform(0, article_delay))

    m = _NEXT_DATA_RE.search(html)
    if not m:
        return ""
    try:
        data = json.loads(m.group(1))
        ad = data["props"]["pageProps"].get("articleData") or {}
    except Exception:
        return ""

    return _extract_body_from_article_data(ad)



def _parse_timestamp(ts_str: str) -> datetime | None:
    """Parse WSJ UTC ISO 8601 string (with or without Z) to ET-aware datetime."""
    if not ts_str:
        return None
    ts_str = ts_str.rstrip("Z") + "+00:00"
    try:
        return datetime.fromisoformat(ts_str).astimezone(ET)
    except ValueError:
        return None


def _get_section_subsection(article_url: str) -> tuple[str | None, str | None]:
    """
    Extract WSJ section and optional subsection from URL path.

    Post-2021 URLs:
      /{section}/{slug-{8hex}}              → (section, None)
      /{section}/{subsection}/{slug-{8hex}} → (section, subsection)

    Legacy /articles/ and /news/ paths → (None, None).
    """
    if not article_url or "wsj.com/" not in article_url:
        return None, None

    path = article_url.split("wsj.com", 1)[1].split("?")[0].split("#")[0]
    parts = [p for p in path.split("/") if p]

    if not parts or parts[0] in ("articles", "news", "video", ""):
        return None, None

    section = parts[0]

    if len(parts) == 2:
        return section, None

    if len(parts) >= 3:
        candidate_slug = parts[-1]
        candidate_sub  = parts[1]
        if _ARTICLE_SLUG_RE.search(candidate_slug) or _UUID_RE.match(candidate_slug.upper()):
            subsection = candidate_sub if candidate_sub != section else None
            return section, subsection

    return section, None


def _make_article_id(seo_id: str) -> str:
    return hashlib.sha256(f"wsj:{seo_id}".encode()).hexdigest()[:20]


def _should_exclude(headline: str, section: str | None) -> bool:
    if section in _EXCLUDED_SECTIONS:
        return True
    if _EXCLUDE_HEADLINE_RE.search(headline):
        return True
    return False



class WSJIngestor(AbstractIngestor):
    """
    Ingests WSJ article metadata from the public archive pages.

    Parameters
    ----------
    output_root:
        Root directory for Parquet output (output_root/YYYY/MM/DD.parquet).
    trading_days:
        Set of valid US trading days (from sync_daily.csv).
    archive_delay:
        Seconds to sleep after each archive page request (one per calendar day).
    article_delay:
        Base seconds to sleep after each article page request (Phase 2 only).
        Actual sleep is jittered to [article_delay, article_delay * 2].
    cookie_file:
        Path to a Netscape-format cookies.txt from a logged-in WSJ session.
        When provided, article bodies are fetched in parallel (Phase 2).
        Without it, body_excerpt = headline (Phase 1).
    max_workers:
        Thread-pool size for parallel article body fetches (Phase 2 only).
        Each worker maintains its own requests.Session so DataDome cookie
        updates are handled independently per thread.
    """

    source_name = "wsj"

    def __init__(
        self,
        output_root: Path,
        trading_days: frozenset[date],
        archive_delay: float = 2.0,
        article_delay: float = 0.3,
        cookie_file: Path | None = None,
        max_workers: int = 16,
    ) -> None:
        super().__init__(output_root, trading_days)
        self.archive_delay = archive_delay
        self.article_delay = article_delay
        self.max_workers   = max_workers
        self._cookie_file  = cookie_file
        self._cookie_dict: dict[str, str] = {}
        self._thread_local = threading.local()

        if cookie_file is not None:
            self._cookie_dict = _load_cookie_dict(cookie_file)

    # ------------------------------------------------------------------
    # Thread-local session management
    # ------------------------------------------------------------------

    def _get_session(self):
        """Return (creating if needed) a requests.Session for this thread."""
        if not hasattr(self._thread_local, "session"):
            import requests as _requests
            s = _requests.Session()
            s.cookies.update(self._cookie_dict)
            s.headers.update({"User-Agent": _USER_AGENT})
            self._thread_local.session = s
        return self._thread_local.session

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ingest_date(self, target_date: date) -> list[dict]:
        """
        Fetch and process all articles from the WSJ archive page for target_date.

        Phase 1 (no cookie_file): metadata only, body_excerpt = headline.
        Phase 2 (cookie_file provided): article bodies fetched in parallel
        via a ThreadPoolExecutor with max_workers threads.

        Retries up to 3 times with exponential backoff on transient HTTP errors.
        """
        html = None
        for attempt in range(3):
            try:
                html = _fetch_archive_html(target_date)
                break
            except Exception as exc:
                wait = self.archive_delay * (2 ** attempt) + random.uniform(0, 2)
                import sys
                print(
                    f"    [retry {attempt+1}/3] {target_date} fetch failed: {exc!r} — "
                    f"waiting {wait:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)

        time.sleep(self.archive_delay + random.uniform(0, self.archive_delay * 0.5))

        if html is None:
            return []

        raw_articles = _parse_next_data_archive(html)

        # Phase 1: no auth — process sequentially, no body fetch
        if not self._cookie_dict:
            return [r for raw in raw_articles if (r := self._process_raw(raw)) is not None]

        # Phase 2: parallel body fetches --------------------------------
        # First pass: filter out articles we won't keep (section / headline / trading day)
        # so we don't waste fetches on excluded articles.
        candidates: list[tuple[dict, dict | None]] = []
        for raw in raw_articles:
            skeleton = self._build_skeleton(raw)
            if skeleton is not None:
                candidates.append((raw, skeleton))

        if not candidates:
            return []

        # Fetch bodies in parallel
        def fetch_one(item: tuple[dict, dict]) -> dict:
            raw, skeleton = item
            body = _fetch_article_body(
                skeleton["_url"], self._get_session(), self.article_delay
            )
            if body:
                skeleton["body_excerpt"] = body
                skeleton["word_count"]   = len(body.split())
            del skeleton["_url"]   # internal key not part of schema
            return skeleton

        records: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(fetch_one, item): item for item in candidates}
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except Exception:
                    pass

        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_skeleton(self, raw: dict) -> dict | None:
        """
        Validate and build an ArticleRecord dict with body_excerpt = headline.
        Stores the article URL in the private '_url' key for the body fetch.
        Returns None if the article should be excluded.
        """
        headline = (raw.get("headline") or "").replace("\n", " ").strip()
        if len(headline.split()) < MIN_WORD_COUNT_HEADLINE_ONLY:
            return None

        eff_time = _parse_timestamp(raw.get("timestamp", ""))
        if eff_time is None:
            return None

        article_url = raw.get("articleUrl") or ""
        seo_id      = raw.get("seoId") or ""
        if not seo_id and article_url:
            seo_id = article_url.rstrip("/").split("/")[-1]
        section, subsection = _get_section_subsection(article_url)

        if _should_exclude(headline, section):
            return None

        effective_date = assign_trading_day(eff_time, self.trading_days)
        if effective_date is None:
            return None

        return {
            "_url":           article_url,          # removed after body fetch
            "article_id":     _make_article_id(seo_id),
            "effective_date": effective_date,
            "effective_time": eff_time,
            "buffer_zone":    is_buffer_zone(eff_time),
            "source":         "wsj",
            "headline":       headline,
            "body_excerpt":   headline,              # overwritten if body fetch succeeds
            "subject_codes":  [c for c in [section, subsection] if c],
            "rics":           [],
            "word_count":     len(headline.split()),
        }

    def _process_raw(self, raw: dict) -> dict | None:
        """Phase 1 path: build skeleton and strip the internal _url key."""
        skeleton = self._build_skeleton(raw)
        if skeleton is None:
            return None
        del skeleton["_url"]
        return skeleton
