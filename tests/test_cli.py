"""One entrypoint, 3 stages, and the invariants applied before vLLM loads."""
import importlib
import os
import sys

import pytest

from macronews import cli


def test_every_stage_is_reachable():
    assert sorted(cli._MAPPER_STAGES) == ["grade", "run"]


@pytest.mark.parametrize("mod", ["macronews.tabular.runner"])
def test_every_stage_module_resolves(mod):
    """The stage dict's keys are checked above; this checks the VALUES resolve to
    real modules. A typo'd module path ships clean and dies at runpy after the GPU
    is allocated. All target modules import cleanly in milliseconds."""
    importlib.import_module(mod)


def test_every_stage_name_appears_in_the_help_text():
    """cli.__doc__ IS `macronews --help`'s output -- a stage added to
    _MAPPER_STAGES but forgotten here is silently missing from --help forever."""
    for name in {*cli._MAPPER_STAGES, "tabular"}:
        assert name in cli.__doc__, f"{name!r} missing from cli.__doc__ (= --help)"


def test_the_invariants_are_applied_before_a_stage_runs(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    seen = {}
    monkeypatch.setattr(cli, "_mapper_run",
                        lambda argv: seen.update(env=os.environ.get("VLLM_BATCH_INVARIANT")))
    monkeypatch.setattr(sys, "argv", ["macronews", "mapper", "run"])
    cli.main()
    assert seen["env"] == "1", "a stage ran before the invariants were applied"


def test_mapper_run_builds_a_validated_config(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(cli, "_run_experiment", lambda cfg: captured.update(cfg=cfg))
    monkeypatch.setattr(sys, "argv", [
        "macronews", "mapper", "run", "--dataset", "gold",
        "--output-file", str(tmp_path / "g.jsonl"),
    ])
    cli.main()
    cfg = captured["cfg"]
    assert cfg.dataset == "gold"
    assert cfg.keyword_gate is False        # gold is an instrument, not production
    assert cfg.max_model_len == 65_536
    assert cfg.model.name == "gemma-4-26b-a4b-it"


@pytest.mark.parametrize("argv,gated", [
    (["mapper", "run", "--dataset", "djnw", "--input-file", "X", "--output-dir", "d"], True),
    (["mapper", "run", "--dataset", "gold", "--output-file", "o.jsonl"], False),
    (["mapper", "run", "--dataset", "gold", "--output-file", "o.jsonl", "--keyword-gate"], True),
])
def test_the_gate_default_follows_the_dataset(argv, gated, monkeypatch):
    shard = ("/nfs/roberts/project/pi_btk22/rc2573/output/cleaned/v2/articles/"
             "2014-05c_clean.jsonl")
    argv = [a if a != "X" else shard for a in argv]
    captured = {}
    monkeypatch.setattr(cli, "_run_experiment", lambda cfg: captured.update(cfg=cfg))
    monkeypatch.setattr(sys, "argv", ["macronews", *argv])
    cli.main()
    assert captured["cfg"].keyword_gate is gated


def test_a_bad_value_is_rejected_by_the_config(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "macronews", "mapper", "run", "--dataset", "gold",
        "--output-file", str(tmp_path / "g.jsonl"), "--max-model-len", "999999",
    ])
    with pytest.raises(SystemExit):
        cli.main()
