"""Streaming line-buffered NML parser for the DJN raw archive.

Yields one NMLArticle per <doc>...</doc> block. We don't use xml.etree because the
raw archive is 171 GB across 1,260 monthly files and full-tree parsing per file is
wasteful — we only need:

  - the ``accession-number`` attribute on <djn-mdata>
  - the verbatim text inside each <pre>...</pre> block (whitespace preserved)
  - the text inside each non-trailer <p>...</p> body block

Trailer recognition mirrors `_WIRE_MARKER_ONLY_RE` in loaders.py, plus
ISO-timestamp-only <p>s.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Trailer detection — match the rules used in loaders.py.
# ---------------------------------------------------------------------------

# Wire markers: matches loaders.py's _WIRE_MARKER_ONLY_RE.
_WIRE_MARKER_RE = re.compile(r"^(?:-0-?|\(END\)|(?:\(MORE\)\s*)+)\s*$")

# ISO timestamp <p> trailer, e.g.:
#   "June 13, 1979 08:08 ET (12:08 GMT)"
#   "September 01, 2002 00:06 ET (04:06 GMT)"
#   "Corrected July 19, 2004 16:49 ET (20:49 GMT)"   ← DJN editorial prefix
#   "Updated October 1, 2010 12:00 ET"               ← short-form, no GMT
# Allow an optional single capitalized-word prefix (Corrected/Updated/Refiled/etc.)
_TIMESTAMP_RE = re.compile(
    r"^(?:[A-Z][a-z]+\s+)?"  # optional leading editorial verb
    r"[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}\s+\d{1,2}:\d{2}\s+[A-Z]{2,4}"
    r"(?:\s+\([^)]+\))?\s*$"
)

# "(END) Dow Jones Newswires" — wire-source trailer line variant.
_END_DJ_RE = re.compile(r"^\(END\)\s+Dow\s+Jones\s+Newswires?\s*$", re.IGNORECASE)

# "(MORE TO FOLLOW) Dow Jones Newswires" — continuation-marker trailer used
# by DJ on multi-part wires (e.g. Fed open-market operations, earnings
# coverage). Distinct from the bare "(MORE)" wire marker handled above.
_MORE_TO_FOLLOW_DJ_RE = re.compile(
    r"^\(MORE TO FOLLOW\)\s+Dow\s+Jones\s+Newswires?\s*$", re.IGNORECASE
)


def _is_trailer_text(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    return bool(
        _WIRE_MARKER_RE.match(s)
        or _TIMESTAMP_RE.match(s)
        or _END_DJ_RE.match(s)
        or _MORE_TO_FOLLOW_DJ_RE.match(s)
    )


# ---------------------------------------------------------------------------
# Block extraction within a single <doc>.
# ---------------------------------------------------------------------------

_DOC_OPEN_RE = re.compile(r"<doc\b")
_DOC_CLOSE_RE = re.compile(r"</doc>")
_ACCESSION_RE = re.compile(r'accession-number="([^"]+)"')
_PRE_RE = re.compile(r"<pre\b[^>]*>(.*?)</pre>", re.DOTALL)
_P_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.DOTALL)


@dataclass(frozen=True)
class NMLArticle:
    """One <doc> from the raw NML file.

    p_blocks_text and pre_blocks_text preserve document order; the order
    is informational (the detector aggregates regardless of order).
    """

    accession_number: str
    p_blocks_text: list[str] = field(default_factory=list)
    pre_blocks_text: list[str] = field(default_factory=list)


def _parse_doc(doc_xml: str) -> NMLArticle | None:
    """Parse one <doc>...</doc> XML chunk into an NMLArticle.

    Returns None if no accession_number can be extracted (corrupt doc).
    """
    m = _ACCESSION_RE.search(doc_xml)
    if not m:
        return None
    accession = m.group(1)

    pre_blocks = list(_PRE_RE.findall(doc_xml))

    p_blocks = []
    for raw in _P_RE.findall(doc_xml):
        decoded = html.unescape(raw).strip()
        if not decoded:
            continue
        if _is_trailer_text(decoded):
            continue
        p_blocks.append(decoded)

    return NMLArticle(
        accession_number=accession,
        p_blocks_text=p_blocks,
        pre_blocks_text=pre_blocks,
    )


def _iter_per_doc(nml_path: Path) -> Iterator[NMLArticle]:
    """Yield one NMLArticle per <doc> block in the file (no aggregation).

    Streams the file line-by-line, accumulating buffer between <doc> and
    </doc> markers, then parses one doc at a time. Used internally by
    iter_nml_articles, which aggregates by accession_number.
    """
    buf: list[str] = []
    in_doc = False
    with open(nml_path, "r", encoding="ISO-8859-1") as f:
        for line in f:
            if not in_doc:
                if _DOC_OPEN_RE.search(line):
                    in_doc = True
                    buf = [line]
                continue
            buf.append(line)
            if _DOC_CLOSE_RE.search(line):
                doc_xml = "".join(buf)
                article = _parse_doc(doc_xml)
                if article is not None:
                    yield article
                buf = []
                in_doc = False


def iter_nml_articles(nml_path: Path) -> Iterator[NMLArticle]:
    """Yield one NMLArticle per unique accession_number across the file.

    Two distinct DJN multi-doc patterns share an accession across <doc>s:

      1. Multi-part articles (continuations): headlines like "PR -1-",
         "PR -2-", "PR -3-" appear consecutively with one accession.
      2. Live-update articles (headline "(+N updates)"): docs scatter
         throughout the file as the day progresses, all sharing one
         accession.

    Rob's cleaner concatenates BOTH into one ``text`` field per accession.
    We mirror that by accumulating <p> and <pre> blocks per accession in
    a dict, then yielding once at end-of-file in first-seen order.

    Memory: ~20k articles per monthly file × ~50 KB body each ≈ ~1 GB
    worst case per file, well under the SLURM mem allocation.
    """
    by_accession: dict[str, dict[str, list[str]]] = {}
    order: list[str] = []

    for doc in _iter_per_doc(nml_path):
        aid = doc.accession_number
        if aid not in by_accession:
            by_accession[aid] = {"p": [], "pre": []}
            order.append(aid)
        by_accession[aid]["p"].extend(doc.p_blocks_text)
        by_accession[aid]["pre"].extend(doc.pre_blocks_text)

    for aid in order:
        bucket = by_accession[aid]
        yield NMLArticle(
            accession_number=aid,
            p_blocks_text=bucket["p"],
            pre_blocks_text=bucket["pre"],
        )


def count_p_body_tokens(p_blocks_text: list[str]) -> int:
    """Sum whitespace-split token counts across all <p> body blocks.

    Caller passes the list from NMLArticle.p_blocks_text. Returns 0 for
    empty lists or empty strings.
    """
    return sum(len(b.split()) for b in p_blocks_text)
