"""Tests for src/tabular/detector.py — column-alignment classifier."""

from macronews.tabular.detector import classify_pre_block


# ---------------------------------------------------------------------------
# Empty / too-short blocks: cannot confirm tabular structure.
# ---------------------------------------------------------------------------

def test_empty_pre_returns_empty_lists():
    is_tab, toks = classify_pre_block("")
    assert is_tab == []
    assert toks == []


def test_whitespace_only_pre_returns_empty_lists():
    is_tab, toks = classify_pre_block("   \n  \n\n   \n")
    assert is_tab == []
    assert toks == []


def test_single_line_forced_narrative():
    # One line of clearly tabular content, but only one row exists.
    # Without 3+ rows we can't confirm column structure → narrative.
    block = "  Issue       Amt       Price"
    is_tab, toks = classify_pre_block(block)
    assert is_tab == [False]
    assert toks == [3]


def test_two_lines_forced_narrative():
    block = (
        "Symbol  Price  Change\n"
        "ABC     100.50  +5.50"
    )
    is_tab, toks = classify_pre_block(block)
    assert is_tab == [False, False]


# ---------------------------------------------------------------------------
# Pure-table content: column-aligned across many rows.
# ---------------------------------------------------------------------------

def test_aligned_table_all_lines_tabular():
    block = (
        "Bank Credit                 5,695.3      24.9\n"
        "Securities                  1,630.8      10.2\n"
        "U.S. Gov't Securities         958.1      19.2\n"
        "Other Securities              672.7      -9.0\n"
        "Real Estate                 1,907.2      10.1\n"
    )
    is_tab, toks = classify_pre_block(block)
    assert is_tab == [True, True, True, True, True]
    assert all(t > 0 for t in toks)


# ---------------------------------------------------------------------------
# Pure narrative: wrapped prose, single-space-separated tokens.
# ---------------------------------------------------------------------------

def test_wrapped_narrative_all_lines_narrative():
    block = (
        "ATLANTA (AP)--Four transplant patients may have been infected with the West\n"
        "Nile virus from the donated organs of a Georgia woman who died in August, the\n"
        "Centers for Disease and Control Prevention said Saturday.\n"
        "The CDC said that three of the four patients have developed symptoms of\n"
        "encephalitis, the inflammation of the brain that is the most serious illness.\n"
    )
    is_tab, toks = classify_pre_block(block)
    assert is_tab == [False, False, False, False, False]


# ---------------------------------------------------------------------------
# Mixed content: caption interleaved with table rows.
# ---------------------------------------------------------------------------

def test_caption_between_table_rows_caption_is_narrative():
    block = (
        "Bank Credit                 5,695.3      24.9\n"
        "Securities                  1,630.8      10.2\n"
        "U.S. Gov't Securities         958.1      19.2\n"
        "(In billions of dollars)\n"
        "Real Estate                 1,907.2      10.1\n"
        "Consumer                      582.2       3.5\n"
    )
    is_tab, toks = classify_pre_block(block)
    # Caption row should be narrative; the rest tabular.
    assert is_tab == [True, True, True, False, True, True]


def test_short_caption_label_alone_is_narrative():
    block = (
        "                                ASSETS\n"
        "Bank Credit                 5,695.3      24.9\n"
        "Securities                  1,630.8      10.2\n"
        "Real Estate                 1,907.2      10.1\n"
    )
    is_tab, toks = classify_pre_block(block)
    # First line is just a centered single-word header → no mid-line gaps → narrative.
    assert is_tab == [False, True, True, True]


# ---------------------------------------------------------------------------
# Token counts always match the non-blank line count.
# ---------------------------------------------------------------------------

def test_token_counts_match_lines():
    block = "Hello world\nFoo bar baz\nA B C D E"
    is_tab, toks = classify_pre_block(block)
    assert toks == [2, 3, 5]


def test_tight_data_rows_with_single_gap_classified_tabular():
    # CME Livestock-style: short rows with ONE multi-space gap (between
    # "up"/"dn" and the change value) AND mostly-numeric tokens. Each row
    # alone has 1 mid-line gap (under our >=2 threshold), but the high
    # numeric density should tip it to tabular under the secondary rule.
    block = (
        "Chgo Live Cattle\n"
        "       Oct 70.25 unch\n"
        "       Dec 72.45 up  0.03\n"
        "       Feb 73.10 dn  0.13\n"
        "Live Hogs\n"
        "       Oct 35.90 up  0.38\n"
        "       Dec 38.25 up  0.23\n"
        "       Feb 45.60 up  0.63\n"
    )
    is_tab, toks = classify_pre_block(block)
    # Expect at least 5 of the 8 lines flagged tabular: the data rows with
    # `up`/`dn` change indicators (1 gap + 50% numeric tokens).
    n_tabular = sum(1 for flag in is_tab if flag)
    assert n_tabular >= 5, f"expected >=5 tabular rows, got {n_tabular}: {list(zip(is_tab, toks))}"


def test_narrative_bullet_with_single_gap_stays_narrative():
    # Honda PR-style bullets: "*  ..." has 1 mid-line gap (the 2 spaces
    # after the asterisk) but ~0-15% numeric tokens. Must stay narrative
    # under the combined rule (≥1 gap requires ≥50% numeric).
    block = (
        "*  Hovnanian achieved all-time record highs for any third quarter in the\n"
        "Company's history in home deliveries, net contracts, sales backlog,\n"
        "*  Total American Honda August sales of 137,273 were up 12.9 percent\n"
        "from last year and broke the previous monthly record of 117,768 set\n"
        "*  Revenues increased 38% to $704.6 million for the third quarter.\n"
    )
    is_tab, toks = classify_pre_block(block)
    assert all(not flag for flag in is_tab), f"got is_tab={is_tab}"


def test_blank_lines_in_middle_are_skipped():
    block = (
        "Bank Credit                 5,695.3      24.9\n"
        "\n"
        "Securities                  1,630.8      10.2\n"
        "\n"
        "Real Estate                 1,907.2      10.1\n"
    )
    is_tab, toks = classify_pre_block(block)
    # 3 non-blank lines should be returned.
    assert len(is_tab) == 3
    assert len(toks) == 3
    assert is_tab == [True, True, True]


def test_separator_lines_classified_tabular():
    # Separator-only lines (`---...`, `===:===:===`) sit between data rows in
    # real tables and must count as tabular structure, not narrative.
    block = (
        "                       Current Bid-Ask             Previous\n"
        "                       ---------------             --------\n"
        "  Gold                 1659.50-1660.30      1676.00-1676.80\n"
        "  Silver                   30.32-30.37          30.73-30.78\n"
        "==========================:================:=================:================\n"
    )
    is_tab, toks = classify_pre_block(block)
    assert len(is_tab) == 5
    # The 2nd line (dashes) and 5th line (=:= mix) are pure separators.
    assert is_tab[1] is True, "dash separator must be tabular"
    assert is_tab[4] is True, "=/: separator must be tabular"


def test_separator_with_word_stays_narrative():
    # A line with separator chars PLUS a real word (table-wrap from wide
    # header) is not a pure separator — must follow the gap/numeric rules.
    block = (
        "  A                      B                      C\n"
        "  1                      2                      3\n"
        "=================:  NONREPORTABLE\n"
    )
    is_tab, toks = classify_pre_block(block)
    # Line 3 has the word NONREPORTABLE, only 1 gap, no numerics → narrative.
    assert is_tab[2] is False
