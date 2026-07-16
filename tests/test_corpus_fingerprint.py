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
# articles of 2014-05c with the gate on. MEASURED on main @ 7f57b9e.
# Of those 2000 the tabular-body filter drops 127 -- the number that goes to 0 if the
# sidecar path breaks. (Over the FULL shard it drops 1,386; this test reads the first
# 2000 articles so it stays a CPU-seconds guard.)
_PINNED = (1295, 8183, "834ceae48524ccf6")


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
