"""The benchmark harness needs GPU-seconds and vLLM stats. Both are off by
default so production behaviour is unchanged; the compute-cost benchmark
turns them on."""
import json

import pytest

import macronews.pipeline as pipeline
from macronews.config.runconfig import MapperConfig
from macronews.mapping.schemas import SingleAssetResult


class StubMapper:
    def __init__(self):
        self.seen: list[str] = []
        self.system_prompt = ""

    def asset_class_rules(self, asset_class: str) -> tuple[str, str]:
        return ("", "")

    def map_single_asset(self, texts, max_tokens=512):
        self.seen.extend(texts)
        return [SingleAssetResult(relevance_score=0.0, relevant=False) for _ in texts]

    def _init_llm(self):
        """No-op: the stub never loads a model. Present so run_experiment's
        explicit weight-load call works without a production-side fallback."""


ARTICLES = [{
    "id": "a1",
    "headline": "Oil Prices Climb on Supply Concerns",
    "paragraphs": ["Crude futures rose after the cartel signalled output cuts."],
}]


def test_report_stats_defaults_off(tmp_path):
    """Production must not change. The flag exists only for the benchmark."""
    cfg = MapperConfig(dataset="gold", output_file=tmp_path / "g.jsonl")
    assert cfg.report_stats is False


def test_report_stats_reaches_the_mapper(monkeypatch, tmp_path):
    """A config field nobody reads is the bug this repo keeps shipping."""
    from djnw import runtime as invariants
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    invariants.apply()

    seen = {}
    monkeypatch.setattr(pipeline, "load_articles", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "LLMMapper", lambda **k: seen.update(k) or StubMapper())

    cfg = MapperConfig(dataset="gold", output_file=tmp_path / "g.jsonl", report_stats=True)
    pipeline.run_experiment(cfg)

    assert seen["report_stats"] is True

    # estimate.py hard-requires all four. Missing corpus_s would otherwise surface
    # as a KeyError AFTER the 10-shard GPU spend.
    summary = json.loads((tmp_path / "g.summary.json").read_text())
    assert set(summary["run"]["timing"]) == {"corpus_s", "load_s", "map_s", "n_articles"}


def test_timing_lands_in_the_summary(tmp_path):
    """Per-shard GPU-seconds are the benchmark's whole output."""
    results, gate_stats = pipeline.run_pipeline(StubMapper(), ARTICLES, keyword_gate=True)
    pipeline.save_results_jsonl(
        ARTICLES, results, tmp_path / "shard.jsonl", gate_stats,
        run_record={"timing": {"corpus_s": 0.3, "load_s": 1.5, "map_s": 42.0,
                               "n_articles": 1}},
    )

    summary = json.loads((tmp_path / "shard.summary.json").read_text())
    assert summary["run"]["timing"]["map_s"] == 42.0
    assert summary["run"]["timing"]["load_s"] == 1.5
