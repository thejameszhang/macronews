"""The corpus is what reaches the model. Pin it.

loaders.py derives the tabular sidecar dir from __file__. A missing MONTH within
that directory `continue`s (logger.info, no raise) -- legitimate, 200 of 555 months
have none. A missing DIRECTORY raises instead (see test_tabular_sidecar_guard.py).
This test pins what reaches the model so a path change that resolves to some other
real-but-wrong directory still gets caught, not just an outright-missing one.

CPU only: a stub mapper, no GPU, no model, seconds.
"""
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

import pytest

import macronews.loaders as loaders
import macronews.pipeline as pipeline
from macronews.config.runconfig import gate_default
from macronews.loaders import load_articles
from macronews.mapping.schemas import SingleAssetResult

SHARD = Path(
    "/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles/2014-05c_clean.jsonl"
)
MODEL = Path("/nfs/roberts/scratch/pi_btk22/jyz32/gemma-4-26b-a4b-it")
# The CANONICAL sidecar location. Deliberately NOT loaders._TABULAR_SIDECAR_DIR: that
# constant is what this test checks, so skipping on it would disarm the gate.
TABULAR_DIR = REPO / "results" / "tabular"
N_ARTICLES = 2000

# (surviving_articles, llm_calls, sha256-16 over every prompt) for the first 2000
# articles of 2014-05c with the gate on. MEASURED on sector-exclusion.
# The mapper universe is 39 groups (no US equity sectors), so llm_calls and the
# digest moved from (1295, 8183, "834ceae48524ccf6") on main @ 7f57b9e. `surviving`
# is unchanged BY CONSTRUCTION -- the article filter runs upstream of the group
# fan-out. test_the_mapper_universe_removes_exactly_the_sector_prompts proves the
# delta is exactly the sector pairs and nothing else.
# Of those 2000 the tabular-body filter drops 127 -- the number that goes to 0 if the
# sidecar path breaks. (Over the FULL shard it drops 1,386; this test reads the first
# 2000 articles so it stays a CPU-seconds guard.)
_PINNED = (1295, 4367, "d31b180db75a0a54")


class StubMapper:
    def __init__(self):
        self.seen: list[str] = []
        self.system_prompt = ""

    def asset_class_rules(self, asset_class):
        return ("", "")

    def map_single_asset(self, texts, max_tokens=512):
        self.seen.extend(texts)
        return [SingleAssetResult(relevance_score=0.0, relevant=False) for _ in texts]


@pytest.fixture(scope="module")
def articles():
    if not SHARD.is_file():
        pytest.skip(f"djnw shard not readable: {SHARD}")
    if not (MODEL / "config.json").is_file():
        # scratch auto-purges weights by atime; skip rather than error, or the plan's
        # one hard gate becomes an unrelated crash.
        pytest.skip(f"tokenizer not on disk: {MODEL} -- re-run slurm/download_llm.sh")
    if not TABULAR_DIR.is_dir():
        # Skip on the CANONICAL location, never on loaders._TABULAR_SIDECAR_DIR --
        # that constant is the very thing this test exists to check. Guarding on it
        # would make the fixture SKIP exactly when the path is wrong, i.e. disarm the
        # gate on the one bug it is here to catch.
        pytest.skip("no results/tabular/ -- run the tabular stage first")
    # Mirrors pipeline.run_experiment exactly: max(1024, max_model_len - 2000).
    return load_articles(
        "djnw", SHARD.parent, max_articles=N_ARTICLES, input_file=SHARD,
        max_tokens=max(1024, 65536 - 2000),
        tokenizer_path=str(MODEL), chars_per_token=2.0,
    )


def test_the_tabular_filter_actually_fires(articles):
    """If the sidecar path breaks, this drops to 0 and the corpus silently grows."""
    tabular = sum(1 for a in articles
                  if "tabular_body" in (a.get("filtered_reasons") or []))
    assert tabular == 127, (
        f"tabular_body filtered {tabular} articles, expected 127. If 0, the sidecar "
        f"is missing and load_djnw_articles's per-month `continue` swallowed it "
        f"(not the directory-missing raise). The corpus has silently changed."
    )


def test_corpus_fingerprint_is_stable(articles):
    """Which articles survive, and which (article, group) pairs reach the model."""
    m = StubMapper()
    pipeline.run_pipeline(m, articles, keyword_gate=gate_default("djnw"))

    digest = hashlib.sha256()
    for text in m.seen:
        digest.update(text.encode())

    surviving = sum(1 for a in articles if not a.get("filtered_reasons"))
    actual = (surviving, len(m.seen), digest.hexdigest()[:16])
    assert actual == _PINNED, f"corpus changed: {actual} != {_PINNED}"


def test_the_mapper_universe_removes_exactly_the_sector_prompts(articles, monkeypatch):
    """The 39-group batch is the 50-group batch minus the sector pairs, and nothing else.

    This is what makes re-pinning _PINNED honest. The pin proves a number moved;
    this proves WHICH prompts left -- same articles, same order, byte-identical
    prompts for every surviving pair.
    """
    from macronews.utils.groups import (
        group_keys, load_group_universe, load_mapper_group_universe,
    )

    def batch_for(universe):
        monkeypatch.setattr(pipeline, "_GROUP_UNIVERSE", universe)
        monkeypatch.setattr(pipeline, "ALL_GROUPS", group_keys(universe))
        m = StubMapper()
        pipeline.run_pipeline(m, articles, keyword_gate=gate_default("djnw"))
        return list(m.seen)

    seen_50 = batch_for(load_group_universe())
    seen_39 = batch_for(load_mapper_group_universe())

    # pipeline._group_label renders "<name> | <asset_class> — constituents: <members>"
    # into the group-last [ASSET_GROUP] block, so the class is present verbatim.
    mark = " | US equity sector — constituents: "
    assert any(mark in t for t in seen_50), (
        "no sector prompts in the 50-group batch -- this test would prove nothing"
    )
    assert not any(mark in t for t in seen_39), "a sector prompt reached the model"
    assert seen_39 == [t for t in seen_50 if mark not in t], (
        "the filter changed more than the sector pairs: an article, an order, or a "
        "prompt body moved"
    )
