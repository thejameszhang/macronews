"""The gate is wired into run_pipeline's fan-out, not just importable.

test_gate.py pins the keyword predicate. These tests pin the thing that
actually saves the money: that run_pipeline asks the model about FEWER
(article, group) pairs when the gate is on, and about all 50 when it is off.
Without them, deleting the `continue` in the fan-out turns the gate into a
no-op and the whole suite still passes.
"""
import pytest

import macronews.pipeline as pipeline
from macronews.config.runconfig import gate_default
from macronews.mapping.schemas import SingleAssetResult


class StubMapper:
    """Records every text it is asked about; never loads a model."""

    def __init__(self):
        self.seen: list[str] = []
        self.system_prompt = ""

    def asset_class_rules(self, asset_class: str) -> tuple[str, str]:
        return ("", "")

    def map_single_asset(self, texts, max_tokens=512):
        self.seen.extend(texts)
        return [SingleAssetResult(relevance_score=0.0, relevant=False) for _ in texts]


# An oil article: it mentions no cocoa, no sugar, no yen.
ARTICLES = [
    {
        "id": "a1",
        "headline": "Oil Prices Climb on Supply Concerns",
        "paragraphs": ["Crude futures rose after the cartel signalled output cuts."],
    }
]


def _groups_asked(mapper: StubMapper) -> set[str]:
    """Recover which group each call was about from the [ASSET_GROUP] suffix."""
    return {t.rsplit("[ASSET_GROUP]", 1)[1].split("|")[0].strip() for t in mapper.seen}


def test_gate_on_skips_calls():
    m = StubMapper()
    _, stats = pipeline.run_pipeline(m, ARTICLES, keyword_gate=True)

    assert stats["keyword_gate"] is True
    assert len(m.seen) == stats["calls_made"]
    assert stats["calls_skipped"] > 0, "gate fired on nothing — it is a no-op"
    assert stats["calls_made"] + stats["calls_skipped"] == len(pipeline.ALL_GROUPS)
    assert len(m.seen) < len(pipeline.ALL_GROUPS)

    asked = _groups_asked(m)
    assert "Crude Oil" in asked
    assert "Cocoa" not in asked, "gate let through a group with no keyword hit"


def test_gate_off_calls_every_group():
    m = StubMapper()
    _, stats = pipeline.run_pipeline(m, ARTICLES, keyword_gate=False)

    assert stats["keyword_gate"] is False
    assert stats["calls_skipped"] == 0
    assert len(m.seen) == len(pipeline.ALL_GROUPS)
    assert "Cocoa" in _groups_asked(m)


def test_gate_is_the_only_difference_between_the_two_runs():
    """Every call the gated run makes, the ungated run also makes."""
    gated, ungated = StubMapper(), StubMapper()
    pipeline.run_pipeline(gated, ARTICLES, keyword_gate=True)
    pipeline.run_pipeline(ungated, ARTICLES, keyword_gate=False)
    assert _groups_asked(gated) < _groups_asked(ungated)


def test_gate_stats_reach_the_summary(tmp_path):
    """A gated shard and an ungated one must be distinguishable from artifacts."""
    import json

    m = StubMapper()
    results, stats = pipeline.run_pipeline(m, ARTICLES, keyword_gate=True)
    out = tmp_path / "shard.jsonl"
    pipeline.save_results_jsonl(ARTICLES, results, out, stats, run_record={})

    summary = json.loads((tmp_path / "shard.summary.json").read_text())
    assert summary["gate"]["keyword_gate"] is True
    assert summary["gate"]["calls_skipped"] > 0
    assert summary["gate"]["calls_saved_pct"] > 0


# ---------------------------------------------------------------------------
# The gate is a PRODUCTION cost optimization; the other datasets are instruments
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset,gated", [
    ("djnw", True),        # production: skip calls whose answer was going to be "no"
    ("gold", False),       # the labelled benchmark
    ("sports", False),     # negative control
    ("wikigaming", False), # negative control
])
def test_gate_defaults_on_only_for_production_data(dataset, gated):
    assert gate_default(dataset) is gated


def test_negative_control_still_asks_the_model():
    """The controls exist to show the MODEL rejects non-financial text.

    A sports article trips no keywords, so a gated run would skip all 50 calls
    and emit an empty result — the control would 'pass' without the model ever
    running, and a model that tagged everything would pass it too. So the
    control must reach the model on every group.
    """
    m = StubMapper()
    sports = [{
        "id": "s1",
        "headline": "Rangers Win in Overtime as Goaltender Stops 40 Shots",
        "paragraphs": ["The winger scored on a power play midway through the third period."],
    }]
    pipeline.run_pipeline(m, sports, keyword_gate=gate_default("sports"))
    assert len(m.seen) == len(pipeline.ALL_GROUPS), (
        "the gate silently removed the model from its own negative control"
    )


def test_a_directory_cannot_end_up_half_gated(tmp_path):
    """The array launcher skips shards that already exist. If the gate is flipped
    between runs, the old shards stay ungated and the new ones are gated — a
    corpus that is half one thing and half the other. Fail loud instead."""
    m = StubMapper()
    ungated_results, ungated_stats = pipeline.run_pipeline(m, ARTICLES, keyword_gate=False)
    pipeline.save_results_jsonl(ARTICLES, ungated_results, tmp_path / "shard_a.jsonl",
                                ungated_stats, run_record={})

    gated_results, gated_stats = pipeline.run_pipeline(StubMapper(), ARTICLES, keyword_gate=True)
    with pytest.raises(ValueError, match="half-gated"):
        pipeline.save_results_jsonl(ARTICLES, gated_results, tmp_path / "shard_b.jsonl",
                                    gated_stats, run_record={})


def test_a_pre_gate_summary_counts_as_ungated(tmp_path):
    """The real prod/2014-ungated shards predate the gate: their summaries have no
    `gate` block at all. Resuming that run with gated code is the likeliest way to
    corrupt a corpus, so a missing block must read as ungated, not unknown."""
    import json

    (tmp_path / "old.summary.json").write_text(json.dumps({"total_articles": 10}))
    results, stats = pipeline.run_pipeline(StubMapper(), ARTICLES, keyword_gate=True)
    with pytest.raises(ValueError, match="keyword_gate=False"):
        pipeline.save_results_jsonl(ARTICLES, results, tmp_path / "new.jsonl", stats,
                                    run_record={})


def test_matching_gate_state_is_fine(tmp_path):
    """Two shards agreeing on the gate is the normal case and must not raise."""
    for shard in ("shard_a", "shard_b"):
        results, stats = pipeline.run_pipeline(StubMapper(), ARTICLES, keyword_gate=True)
        pipeline.save_results_jsonl(ARTICLES, results, tmp_path / f"{shard}.jsonl", stats,
                                    run_record={})
    assert len(list(tmp_path.glob("*.summary.json"))) == 2


def test_rewriting_the_same_shard_is_fine(tmp_path):
    """Re-running one shard in place must not trip the guard on its own summary."""
    for _ in range(2):
        results, stats = pipeline.run_pipeline(StubMapper(), ARTICLES, keyword_gate=True)
        pipeline.save_results_jsonl(ARTICLES, results, tmp_path / "shard_a.jsonl", stats,
                                    run_record={})


def test_run_experiment_takes_a_config(monkeypatch, tmp_path):
    """One frozen object in, no loose positional args to get out of order."""
    from djnw import runtime as invariants
    from macronews.config.runconfig import MapperConfig
    from macronews import pipeline as pl

    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    invariants.apply()                       # the entrypoint does this

    seen = {}
    monkeypatch.setattr(pl, "load_articles", lambda *a, **k: (seen.update(k) or []))
    monkeypatch.setattr(pl, "LLMMapper", lambda **k: StubMapper())

    cfg = MapperConfig(dataset="gold", output_file=tmp_path / "g.jsonl")
    pl.run_experiment(cfg)

    # the corpus filter must come from the config, not be recomputed
    assert seen["max_tokens"] == cfg.max_article_tokens == 63_536
    assert seen["tokenizer_path"] == str(cfg.model)


def test_gate_fires_makes_the_same_decisions_as_the_pipeline():
    """gate_fires() and run_pipeline() are the two implementations of one rule.

    They cannot share the loop: run_pipeline decides inline while it assembles the batch,
    and gate_fires must hand back the per-article breakdown that assembling discards. So
    the rule is written twice, and this is what keeps it one rule.

    Everything that MEASURES the gate -- the corpus call-count, the recall scorecard --
    reads gate_fires. If it drifts from the pipeline, every published cost and recall
    figure drifts with it, silently, and the mapper keeps working the whole time.
    """
    from macronews.mapping.gate import gate_fires

    _, stats = pipeline.run_pipeline(StubMapper(), ARTICLES, keyword_gate=True)
    fires = gate_fires(ARTICLES)

    assert sum(len(groups) for groups in fires.values()) == stats["calls_made"]
    assert stats["calls_made"] > 0, "a gate that fires on nothing proves nothing here"
