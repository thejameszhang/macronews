"""Tests for src/tabular/nml_reader.py — streaming NML parser."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tabular.nml_reader import iter_nml_articles, NMLArticle, count_p_body_tokens  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture builder — synthesizes minimal NML files.
# ---------------------------------------------------------------------------

NML_HEADER = '<?xml version="1.0" encoding="ISO-8859-1" ?>\n<!DOCTYPE doc SYSTEM "djnml-1.0b.dtd">\n'


def make_doc(accession: str, body: str) -> str:
    """Wrap a <body>...</body> snippet in the minimal <doc>/<head>/<docdata>
    boilerplate so it round-trips through the NML reader."""
    return (
        f'<doc msize="0" md5="x" sysId="x">\n'
        f'<djnml publisher="DJN" docdate="20020101" product="DN" seq="1" xml:lang="en-us" >\n'
        f'<head>\n'
        f'<docdata>\n'
        f'<djn>\n'
        f'<djn-newswires news-source="DJDN" origin="DJ" service-id="CO" >\n'
        f'<djn-mdata accession-number="{accession}" />\n'
        f'</djn-newswires>\n'
        f'</djn>\n'
        f'</docdata>\n'
        f'</head>\n'
        f'{body}\n'
        f'</djnml>\n'
        f'</doc>\n'
    )


# ---------------------------------------------------------------------------
# Test cases.
# ---------------------------------------------------------------------------

def test_iter_yields_zero_articles_for_empty_file(tmp_path):
    p = tmp_path / "empty.nml"
    p.write_text(NML_HEADER)
    assert list(iter_nml_articles(p)) == []


def test_iter_yields_accession_numbers(tmp_path):
    nml = NML_HEADER
    for aid in ("20020101000001", "20020101000002", "20020101000003"):
        nml += make_doc(aid, "<body><headline>x</headline><text></text></body>")
    p = tmp_path / "three.nml"
    p.write_text(nml)
    arts = list(iter_nml_articles(p))
    assert [a.accession_number for a in arts] == [
        "20020101000001",
        "20020101000002",
        "20020101000003",
    ]


def test_pre_block_content_preserved_with_whitespace(tmp_path):
    body = (
        "<body>\n"
        "<headline>Treasury</headline>\n"
        "<text>\n"
        "<pre>\n"
        "         Issue         Amt     Net Amt\n"
        "  10-3/4S 8/05      $9.270       3,302\n"
        " 11-5/8S 11/04      $8.302       3,607\n"
        "</pre>\n"
        "</text>\n"
        "</body>"
    )
    p = tmp_path / "pre.nml"
    p.write_text(NML_HEADER + make_doc("20020101000001", body))
    arts = list(iter_nml_articles(p))
    assert len(arts) == 1
    assert len(arts[0].pre_blocks_text) == 1
    block = arts[0].pre_blocks_text[0]
    # Whitespace alignment must survive.
    assert "         Issue         Amt     Net Amt" in block
    assert "  10-3/4S 8/05      $9.270       3,302" in block


def test_p_body_blocks_extracted(tmp_path):
    body = (
        "<body>\n"
        "<headline>News</headline>\n"
        "<text>\n"
        "<p>\n"
        "  NEW YORK--Markets traded higher today on strong earnings.\n"
        "</p>\n"
        "<p>\n"
        "  The S&amp;P 500 closed up 1.2 percent.\n"
        "</p>\n"
        "</text>\n"
        "</body>"
    )
    p = tmp_path / "p.nml"
    p.write_text(NML_HEADER + make_doc("20020101000001", body))
    arts = list(iter_nml_articles(p))
    assert len(arts[0].p_blocks_text) == 2
    assert "Markets traded higher" in arts[0].p_blocks_text[0]
    assert "S&P 500" in arts[0].p_blocks_text[1] or "S&amp;P 500" in arts[0].p_blocks_text[1]


def test_trailer_p_blocks_excluded(tmp_path):
    body = (
        "<body>\n"
        "<text>\n"
        "<p>\n"
        "  NEW YORK--Real narrative content.\n"
        "</p>\n"
        "<p>\n"
        "  (END) Dow Jones Newswires</p>\n"
        "<p>\n"
        "  June 13, 1979 08:08 ET (12:08 GMT)</p>\n"
        "</text>\n"
        "</body>"
    )
    p = tmp_path / "trailer.nml"
    p.write_text(NML_HEADER + make_doc("20020101000001", body))
    arts = list(iter_nml_articles(p))
    # Only the real narrative <p> should be retained.
    assert len(arts[0].p_blocks_text) == 1
    assert "Real narrative content" in arts[0].p_blocks_text[0]


def test_corrected_timestamp_p_excluded(tmp_path):
    # DJN sometimes prepends an editorial verb ("Corrected", "Updated",
    # "Refiled") before the standard timestamp trailer. These should still
    # be detected as trailers and not contribute to p_tokens.
    body = (
        "<body>\n"
        "<text>\n"
        "<pre>\n"
        " row1 1 2 3\n"
        " row2 4 5 6\n"
        " row3 7 8 9\n"
        "</pre>\n"
        "<p>\n"
        "  Corrected July 19, 2004 16:49 ET (20:49 GMT)</p>\n"
        "<p>\n"
        "  (END)</p>\n"
        "</text>\n"
        "</body>"
    )
    p = tmp_path / "corrected.nml"
    p.write_text(NML_HEADER + make_doc("20020101000001", body))
    arts = list(iter_nml_articles(p))
    assert arts[0].p_blocks_text == [], (
        f"expected all <p> trailers stripped; got {arts[0].p_blocks_text}"
    )


def test_more_marker_p_excluded(tmp_path):
    body = (
        "<body>\n"
        "<text>\n"
        "<p>\n"
        "  Real narrative here.\n"
        "</p>\n"
        "<p>\n"
        "(MORE)</p>\n"
        "<p>\n"
        "-0-</p>\n"
        "</text>\n"
        "</body>"
    )
    p = tmp_path / "more.nml"
    p.write_text(NML_HEADER + make_doc("20020101000001", body))
    arts = list(iter_nml_articles(p))
    assert len(arts[0].p_blocks_text) == 1


def test_more_to_follow_dj_p_excluded(tmp_path):
    # "(MORE TO FOLLOW) Dow Jones Newswires" — multi-part wire continuation
    # marker, distinct from "(MORE)". Appears on Fed operations, earnings
    # coverage, etc. Must be treated as a trailer and stripped.
    body = (
        "<body>\n"
        "<text>\n"
        "<p>\n"
        "  Real narrative here.\n"
        "</p>\n"
        "<p>(MORE TO FOLLOW) Dow Jones Newswires</p>\n"
        "<p>(MORE TO FOLLOW) Dow Jones Newswires</p>\n"
        "</text>\n"
        "</body>"
    )
    p = tmp_path / "more_to_follow.nml"
    p.write_text(NML_HEADER + make_doc("20020101000002", body))
    arts = list(iter_nml_articles(p))
    assert len(arts[0].p_blocks_text) == 1, \
        f"expected only the real-narrative <p> to remain; got {arts[0].p_blocks_text}"


def test_multiple_pre_blocks_in_order(tmp_path):
    body = (
        "<body>\n"
        "<text>\n"
        "<pre>\n"
        "first block\n"
        "</pre>\n"
        "<p>narrative paragraph between</p>\n"
        "<pre>\n"
        "second block\n"
        "</pre>\n"
        "<pre>\n"
        "third block\n"
        "</pre>\n"
        "</text>\n"
        "</body>"
    )
    p = tmp_path / "multi.nml"
    p.write_text(NML_HEADER + make_doc("20020101000001", body))
    arts = list(iter_nml_articles(p))
    assert len(arts[0].pre_blocks_text) == 3
    assert "first block" in arts[0].pre_blocks_text[0]
    assert "second block" in arts[0].pre_blocks_text[1]
    assert "third block" in arts[0].pre_blocks_text[2]


def test_empty_pre_block_yields_empty_string(tmp_path):
    body = (
        "<body>\n"
        "<text>\n"
        "<pre>\n"
        " \n"
        "</pre>\n"
        "<p>narrative</p>\n"
        "</text>\n"
        "</body>"
    )
    p = tmp_path / "empty_pre.nml"
    p.write_text(NML_HEADER + make_doc("20020101000001", body))
    arts = list(iter_nml_articles(p))
    assert len(arts[0].pre_blocks_text) == 1
    # Empty/whitespace-only pre body is preserved as-is (caller decides via detector).
    assert arts[0].pre_blocks_text[0].strip() == ""


def test_zero_pre_blocks(tmp_path):
    body = (
        "<body>\n"
        "<text>\n"
        "<p>only narrative here</p>\n"
        "</text>\n"
        "</body>"
    )
    p = tmp_path / "no_pre.nml"
    p.write_text(NML_HEADER + make_doc("20020101000001", body))
    arts = list(iter_nml_articles(p))
    assert arts[0].pre_blocks_text == []
    assert len(arts[0].p_blocks_text) == 1


def test_html_entities_decoded_in_p_text(tmp_path):
    body = (
        "<body>\n"
        "<text>\n"
        "<p>\n"
        "  Don&apos;t fight the Fed &amp; markets.\n"
        "</p>\n"
        "</text>\n"
        "</body>"
    )
    p = tmp_path / "entities.nml"
    p.write_text(NML_HEADER + make_doc("20020101000001", body))
    arts = list(iter_nml_articles(p))
    text = arts[0].p_blocks_text[0]
    assert "Don't" in text
    assert "Fed & markets" in text


def test_count_p_body_tokens_helper():
    # The helper that callers use to derive p_tokens for the sidecar.
    assert count_p_body_tokens(["Hello world", "Foo bar baz"]) == 5
    assert count_p_body_tokens([]) == 0
    assert count_p_body_tokens([""]) == 0


def test_multi_doc_file_yields_each(tmp_path):
    body1 = "<body><text><pre>aaaa\nbbbb\ncccc\n</pre></text></body>"
    body2 = "<body><text><p>narrative</p></text></body>"
    nml = NML_HEADER + make_doc("20020101000001", body1) + make_doc("20020101000002", body2)
    p = tmp_path / "multi.nml"
    p.write_text(nml)
    arts = list(iter_nml_articles(p))
    assert len(arts) == 2
    assert arts[0].accession_number == "20020101000001"
    assert arts[1].accession_number == "20020101000002"
    assert len(arts[0].pre_blocks_text) == 1
    assert len(arts[1].pre_blocks_text) == 0
    assert len(arts[0].p_blocks_text) == 0
    assert len(arts[1].p_blocks_text) == 1


def test_continuation_docs_with_same_accession_concatenated(tmp_path):
    # A multi-part article in the DJN archive uses the same accession_number
    # across continuation <doc> blocks (e.g. "PR -1-", "PR -2-", "PR -3-").
    # Our reader must concatenate them under one NMLArticle, matching Rob's
    # cleaner.
    body1 = "<body><text><p>part one narrative</p></text></body>"
    body2 = "<body><text><pre>part two table\nrow A\nrow B\n</pre></text></body>"
    body3 = "<body><text><p>part three more narrative</p></text></body>"
    nml = (
        NML_HEADER
        + make_doc("20020101000001", body1)
        + make_doc("20020101000001", body2)
        + make_doc("20020101000001", body3)
    )
    p = tmp_path / "continuations.nml"
    p.write_text(nml)
    arts = list(iter_nml_articles(p))
    assert len(arts) == 1
    art = arts[0]
    assert art.accession_number == "20020101000001"
    # Both <p> blocks from parts 1 and 3 should be present
    assert len(art.p_blocks_text) == 2
    # The <pre> block from part 2 should be present
    assert len(art.pre_blocks_text) == 1
    assert "part one narrative" in art.p_blocks_text[0]
    assert "part three more narrative" in art.p_blocks_text[1]
    assert "part two table" in art.pre_blocks_text[0]


def test_non_consecutive_same_accession_merged(tmp_path):
    # DJN live-update articles (headline "(+N updates)") scatter <doc> blocks
    # across the file with the same accession_number. Rob's cleaner merges
    # them all under one accession; our reader must do the same regardless
    # of doc position.
    body_a1 = "<body><text><p>A first chunk</p></text></body>"
    body_b = "<body><text><p>B unrelated</p></text></body>"
    body_a2 = "<body><text><p>A update one</p></text></body>"
    body_c = "<body><text><p>C unrelated</p></text></body>"
    body_a3 = "<body><text><pre>A table\nrow1\nrow2\nrow3\n</pre></text></body>"
    nml = (
        NML_HEADER
        + make_doc("20020101000001", body_a1)
        + make_doc("20020101000002", body_b)
        + make_doc("20020101000001", body_a2)
        + make_doc("20020101000003", body_c)
        + make_doc("20020101000001", body_a3)
    )
    p = tmp_path / "scattered.nml"
    p.write_text(nml)
    arts = list(iter_nml_articles(p))
    by_acc = {a.accession_number: a for a in arts}
    assert len(arts) == 3, f"expected 3 unique articles, got {len(arts)}"
    assert "20020101000001" in by_acc
    assert "20020101000002" in by_acc
    assert "20020101000003" in by_acc
    a = by_acc["20020101000001"]
    # All 3 fragments of A merged: 2 <p> blocks + 1 <pre> block
    assert len(a.p_blocks_text) == 2
    assert len(a.pre_blocks_text) == 1
    assert "A first chunk" in a.p_blocks_text[0]
    assert "A update one" in a.p_blocks_text[1]
    assert "A table" in a.pre_blocks_text[0]


def test_unrelated_accessions_remain_separate(tmp_path):
    # Continuations are detected by accession_number, NOT by headline.
    # Distinct accessions yield distinct articles even if interleaved.
    body_a = "<body><text><p>A first</p></text></body>"
    body_a2 = "<body><text><p>A continuation</p></text></body>"
    body_b = "<body><text><p>B unrelated</p></text></body>"
    nml = (
        NML_HEADER
        + make_doc("20020101000001", body_a)
        + make_doc("20020101000001", body_a2)
        + make_doc("20020101000002", body_b)
    )
    p = tmp_path / "mixed.nml"
    p.write_text(nml)
    arts = list(iter_nml_articles(p))
    assert len(arts) == 2
    assert arts[0].accession_number == "20020101000001"
    assert len(arts[0].p_blocks_text) == 2
    assert arts[1].accession_number == "20020101000002"
    assert len(arts[1].p_blocks_text) == 1
