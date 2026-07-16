"""Column-alignment classifier for raw NML <pre> blocks.

Per-line classification rule (combined):

  A line is TABULAR iff
      (a) it is a pure separator line (only =, -, :, _, |, +, *, ~, . and
          whitespace — no word characters), OR
      (b) it has >= MIN_MID_LINE_GAPS multi-space gaps, OR
      (c) it has >= 1 multi-space gap AND >= NUMERIC_TOKEN_FRAC of its tokens
          look numeric

Otherwise the line is NARRATIVE. Aggregation across all <pre> blocks in an article
drives the tabular_body filter in loaders.py.

Pure-stdlib, no dependencies.
"""

import re

# Module constants — locked after benchmark validation; describe what
# "column-aligned" structurally IS, not statistical thresholds.
MIN_GAP_WIDTH = 2          # ≥2 consecutive spaces = a column gap
MIN_MID_LINE_GAPS = 2      # primary rule: line tabular if it has this many gaps
NUMERIC_TOKEN_FRAC = 0.50  # secondary rule: with ≥1 gap, line tabular if this
                           # fraction of tokens is numeric
MIN_PRE_LINES_FOR_TABLE = 3  # blocks with <3 non-blank lines are forced narrative

_GAP_RE = re.compile(r" {%d,}" % MIN_GAP_WIDTH)
# Numeric-looking token: starts with digit (optionally signed/$-prefixed),
# composed of digits + . , / - : with optional trailing %.
# Matches: 70.25, $8.302, 12.9%, 1,634, +0.5, -0.13, 11/04, 100,000
# Doesn't match: Sept., Dec, up, dn, Q3 (any non-trailing alpha)
_NUMERIC_TOKEN_RE = re.compile(r"^[+\-]?\$?\d[\d.,/\-:]*%?$")
# Separator line: at least one non-whitespace char, and every non-whitespace
# char is one of =, -, :, _, |, +, *, ~, .  (rule structure dividers like
# `--------------- --------` or `==========================:================:`).
_SEPARATOR_LINE_RE = re.compile(r"^[\s=\-:_|+*~.]*[=\-:_|+*~.][\s=\-:_|+*~.]*$")


def _count_mid_line_gaps(line: str) -> int:
    """Count gaps of >=MIN_GAP_WIDTH spaces inside the line, excluding the
    leading and trailing whitespace runs."""
    stripped = line.strip()
    if not stripped:
        return 0
    return len(_GAP_RE.findall(stripped))


def _numeric_token_fraction(line: str) -> float:
    tokens = line.split()
    if not tokens:
        return 0.0
    n_numeric = sum(1 for t in tokens if _NUMERIC_TOKEN_RE.match(t))
    return n_numeric / len(tokens)


def _is_separator_line(line: str) -> bool:
    """Pure-separator line: only =, -, :, _, |, +, *, ~, . and whitespace.

    Empty/whitespace-only lines do NOT match (regex requires ≥1 separator char).
    """
    return bool(_SEPARATOR_LINE_RE.match(line))


def _line_is_tabular(line: str) -> bool:
    if _is_separator_line(line):
        return True
    n_gaps = _count_mid_line_gaps(line)
    if n_gaps >= MIN_MID_LINE_GAPS:
        return True
    if n_gaps >= 1 and _numeric_token_fraction(line) >= NUMERIC_TOKEN_FRAC:
        return True
    return False


def classify_pre_block(pre_content: str) -> tuple[list[bool], list[int]]:
    """Classify each non-blank line in a <pre> block as tabular or narrative.

    Parameters
    ----------
    pre_content
        Raw text content of one <pre>...</pre> block, whitespace preserved.

    Returns
    -------
    (is_tabular, token_counts)
        ``is_tabular[i]`` is True iff line i (0-indexed over non-blank
        lines) is classified as tabular. ``token_counts[i]`` is the
        whitespace-split token count of that line.

    Behavior:
      - Blank lines (whitespace-only) are dropped before classification.
      - If fewer than MIN_PRE_LINES_FOR_TABLE non-blank lines remain, the
        entire block is forced to narrative.
      - Otherwise each line is classified per the combined rule (see module
        docstring).
    """
    lines = [ln for ln in pre_content.split("\n") if ln.strip()]
    token_counts = [len(ln.split()) for ln in lines]

    if len(lines) < MIN_PRE_LINES_FOR_TABLE:
        return [False] * len(lines), token_counts

    is_tabular = [_line_is_tabular(ln) for ln in lines]
    return is_tabular, token_counts
