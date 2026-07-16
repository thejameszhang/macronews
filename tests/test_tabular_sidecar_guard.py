"""A wrong sidecar path must not read as missing data.

Individual months may legitimately have no tabular sidecar -- 200 of the corpus's
555 months do, and the loader correctly continues past those. A missing DIRECTORY
used to be treated the same way: it logged one INFO line, continued, and the
tabular-body filter silently evaporated -- exit status 0, a different corpus. It
now raises instead (test_a_missing_sidecar_DIRECTORY_raises below); this module
guards that regression.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

import pytest

import macronews.loaders as loaders

SHARD = Path(
    "/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles/2014-05c_clean.jsonl"
)
MODEL = Path("/nfs/roberts/scratch/pi_btk22/jyz32/gemma-4-26b-a4b-it")

pytestmark = pytest.mark.skipif(
    not SHARD.is_file() or not (MODEL / "config.json").is_file(),
    reason="djnw shard or tokenizer not on disk (scratch auto-purges by atime)",
)


def _load(n=200, **kw):
    # tokenizer_path is REQUIRED whenever max_tokens is set (loaders.py raises
    # otherwise), and at n=2000 the shard contains an article long enough to reach
    # the slow path -- so it cannot be omitted.
    return loaders.load_articles(
        "djnw", SHARD.parent, max_articles=n, input_file=SHARD,
        max_tokens=63_536, chars_per_token=2.0, tokenizer_path=str(MODEL), **kw
    )


def test_a_missing_sidecar_DIRECTORY_raises(monkeypatch):
    """The package move turns the sidecar path into src/results/tabular. That must
    crash, not silently drop the filter."""
    monkeypatch.setattr(loaders, "_TABULAR_SIDECAR_DIR", Path("src/results/tabular"))
    with pytest.raises(FileNotFoundError, match="WRONG PATH"):
        _load()


def test_a_missing_MONTH_still_works(tmp_path, monkeypatch):
    """Partial coverage is legitimate: 200 of 555 corpus months have no sidecar."""
    monkeypatch.setattr(loaders, "_TABULAR_SIDECAR_DIR", tmp_path)   # exists, but empty
    articles = _load()
    assert articles, "an empty-but-present sidecar dir must still load articles"
    assert not any("tabular_body" in (a.get("filtered_reasons") or []) for a in articles), \
        "no sidecar for this month -> nothing should be flagged tabular_body"


def test_the_real_path_still_filters():
    """The shipped path must keep firing -- this is the regression the guard protects.

    Same load path as test_corpus_fingerprint (tokenizer_path included), so the 127 is
    pinned under one set of conditions, not two.
    """
    if not (REPO / "results" / "tabular").is_dir():   # canonical, NOT the module constant
        pytest.skip("no tabular sidecars on disk")
    articles = _load(n=2000)
    flagged = sum(1 for a in articles
                  if "tabular_body" in (a.get("filtered_reasons") or []))
    assert flagged == 127
