"""Tests for src/tabular/runner.py — NML → sidecar JSONL CLI."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tabular.runner import build_result, write_sidecar  # noqa: E402
from tabular.schemas import TabularResult  # noqa: E402
from tabular.nml_reader import NMLArticle  # noqa: E402


def test_build_result_pure_narrative_p_only():
    art = NMLArticle(
        accession_number="x",
        p_blocks_text=["Narrative paragraph one.", "Narrative paragraph two."],
        pre_blocks_text=[],
    )
    r = build_result(art)
    assert r.accession_number == "x"
    assert r.p_tokens == 6
    assert r.pre_tabular_tokens == 0
    assert r.pre_narrative_tokens == 0


def test_build_result_pure_tabular_pre():
    pre = (
        "Bank Credit                 5,695.3      24.9\n"
        "Securities                  1,630.8      10.2\n"
        "U.S. Gov't Securities         958.1      19.2\n"
        "Real Estate                 1,907.2      10.1\n"
    )
    art = NMLArticle(
        accession_number="x",
        p_blocks_text=[],
        pre_blocks_text=[pre],
    )
    r = build_result(art)
    assert r.p_tokens == 0
    assert r.pre_narrative_tokens == 0
    assert r.pre_tabular_tokens > 0
    assert r.pre_aligned_pct == 1.0


def test_build_result_mixed():
    # Narrative <p> body + a table inside <pre>.
    art = NMLArticle(
        accession_number="x",
        p_blocks_text=["Long narrative paragraph with stopwords."],
        pre_blocks_text=[
            "Header                Col1     Col2\n"
            "Row1                   1.5      2.5\n"
            "Row2                   3.5      4.5\n"
            "Row3                   5.5      6.5\n"
        ],
    )
    r = build_result(art)
    assert r.p_tokens == 5
    assert r.pre_tabular_tokens > 0


def test_build_result_aggregates_multiple_pre_blocks():
    pre1 = "Header A\nRow1   1\nRow2   2\nRow3   3\n"  # only 1 mid-line gap each → narrative-classified
    pre2 = (
        "Item        Col1     Col2\n"
        "ItemA        1.5      2.5\n"
        "ItemB        3.5      4.5\n"
        "ItemC        5.5      6.5\n"
    )
    art = NMLArticle(
        accession_number="x",
        p_blocks_text=[],
        pre_blocks_text=[pre1, pre2],
    )
    r = build_result(art)
    # pre2 is aligned, pre1 is not — token totals reflect both blocks.
    assert r.pre_tabular_tokens > 0
    assert r.pre_narrative_tokens > 0


def test_build_result_empty_pre_block_contributes_zero():
    art = NMLArticle(
        accession_number="x",
        p_blocks_text=["narrative"],
        pre_blocks_text=["   \n  \n"],
    )
    r = build_result(art)
    assert r.pre_tabular_tokens == 0
    assert r.pre_narrative_tokens == 0


def test_write_sidecar_one_line_per_article(tmp_path):
    arts = [
        NMLArticle(accession_number="a1", p_blocks_text=["one two three"], pre_blocks_text=[]),
        NMLArticle(accession_number="a2", p_blocks_text=[], pre_blocks_text=[]),
    ]
    out = tmp_path / "sidecar.jsonl"
    n = write_sidecar(arts, out)
    assert n == 2
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    r0 = json.loads(lines[0])
    assert r0["accession_number"] == "a1"
    assert r0["p_tokens"] == 3
    r1 = json.loads(lines[1])
    assert r1["accession_number"] == "a2"
    assert r1["p_tokens"] == 0


def test_write_sidecar_roundtrips_via_TabularResult(tmp_path):
    arts = [
        NMLArticle(
            accession_number="x",
            p_blocks_text=["one two"],
            pre_blocks_text=[
                "Header                Col1     Col2\n"
                "Row1                   1.5      2.5\n"
                "Row2                   3.5      4.5\n"
                "Row3                   5.5      6.5\n"
            ],
        ),
    ]
    out = tmp_path / "sc.jsonl"
    write_sidecar(arts, out)
    line = out.read_text().strip()
    r = TabularResult.model_validate_json(line)
    assert r.accession_number == "x"
    assert r.p_tokens == 2
    assert r.pre_tabular_tokens > 0


def test_write_sidecar_creates_parent_dir(tmp_path):
    arts = [NMLArticle(accession_number="x", p_blocks_text=[], pre_blocks_text=[])]
    out = tmp_path / "nested" / "subdir" / "sc.jsonl"
    write_sidecar(arts, out)
    assert out.exists()
