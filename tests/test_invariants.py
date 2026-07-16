"""The invariants are pinned in Python, not in bash.

VLLM_BATCH_INVARIANT was exported by eight SLURM scripts and set nowhere in the
source, so any stage run by hand silently lost batch invariance -- ~29% tag drift,
no error, no warning.
"""
import ast
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

import pytest
from macronews import invariants


def test_apply_sets_batch_invariance(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    invariants.apply()
    assert os.environ["VLLM_BATCH_INVARIANT"] == "1"


def test_apply_refuses_to_run_with_it_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    with pytest.raises(RuntimeError, match="invariant, not a setting"):
        invariants.apply()


def test_attention_backend_is_triton_not_flash():
    """FA2 is sm_90-only and crashes on B200/sm_100. Never FLASH_ATTN."""
    assert invariants.ATTENTION_BACKEND == "TRITON_ATTN"


def test_greedy_decode_and_single_gpu():
    assert invariants.TEMPERATURE == 0.0
    assert invariants.TENSOR_PARALLEL_SIZE == 1


def test_an_artifact_cannot_claim_invariants_it_did_not_have(monkeypatch):
    """record() must refuse rather than write vllm_batch_invariant: null. An
    artifact misdescribing its own provenance is what invalidated an A/B."""
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    with pytest.raises(RuntimeError, match="has not run"):
        invariants.record()


def test_record_is_artifact_ready(monkeypatch):
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    invariants.apply()
    assert invariants.record() == {
        "vllm_batch_invariant": "1",
        "temperature": 0.0,
        "tensor_parallel_size": 1,
        "attention_backend": "TRITON_ATTN",
    }


@pytest.mark.parametrize("rel", (
    "mapping/llm.py", "mapping/grading/llm.py", "kg/temporal_extractor.py",
    "kg/grading/llm.py", "kg/invalidate_llm.py",
))
def test_every_llm_call_site_uses_greedy_decode(rel):
    """A source-text grep for the substring 'temperature=0.0' passes even when the
    LIVE call reads temperature=0.7 and '# temperature=0.0' only survives in a
    trailing comment. Parse the AST instead: every SamplingParams(...) call's
    temperature kwarg must be the literal 0.0."""
    tree = ast.parse((REPO / "src" / "macronews" / rel).read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and (getattr(n.func, "id", None) == "SamplingParams"
                  or getattr(n.func, "attr", None) == "SamplingParams")]
    assert calls, f"{rel} does not call SamplingParams(...)"
    for call in calls:
        temp = next((kw.value for kw in call.keywords if kw.arg == "temperature"), None)
        assert temp is not None, f"{rel}: SamplingParams(...) has no temperature kwarg"
        assert isinstance(temp, ast.Constant) and temp.value == 0.0, (
            f"{rel}: SamplingParams(...) temperature is not the literal 0.0"
        )


def test_no_entrypoint_exposes_tensor_parallel_size():
    """TP=1 is an invariant. No CLI may offer to change it."""
    for rel in ("pipeline.py", "mapping/grading/runner.py",
                "kg/runner.py", "kg/grading/runner.py"):
        src = (REPO / "src" / "macronews" / rel).read_text()
        assert "--tensor-parallel-size" not in src, f"{rel} still exposes TP"


def test_no_entrypoint_hardcodes_a_model_path():
    """kg/link_groups.py had an absolute /nfs/... path inside add_argument. Model paths
    live in config.paths so they move when the model moves."""
    for rel in ("kg/link_groups.py", "kg/runner.py", "kg/grading/runner.py",
                "kg/invalidate_llm.py"):
        src = (REPO / "src" / "macronews" / rel).read_text()
        assert "/nfs/roberts/scratch" not in src, f"{rel} hardcodes a model path"


def test_the_kg_extractor_does_not_advertise_the_superseded_model():
    """kg/runner.py's --model help said 'Path to Gemma 4 31B'; its docstring said
    ~/models/gemma-4-31b-it. Production moved to 26B-A4B: same quality, +45% coverage,
    2.84x faster extraction."""
    src = (REPO / "src" / "macronews" / "kg" / "runner.py").read_text()
    assert "31B" not in src and "31b" not in src


def test_the_embedder_is_named_for_what_it_is():
    """kg/disambiguate.py called its sentence-transformers embedder DEFAULT_MODEL, which
    reads exactly like the LLM constants and is not one."""
    from macronews.config.paths import EMBED_MODEL
    assert EMBED_MODEL == "BAAI/bge-large-en-v1.5"
    src = (REPO / "src" / "macronews" / "kg" / "disambiguate.py").read_text()
    assert "DEFAULT_MODEL" not in src


import importlib


@pytest.mark.parametrize("mod", [
    "macronews.kg.runner", "macronews.kg.grading.runner",
    "macronews.kg.invalidate_llm", "macronews.kg.link_groups",
])
def test_every_llm_entrypoint_applies_the_invariants_first(mod):
    """cli.main() applies them -- but `python -m macronews.kg.runner` bypasses the CLI,
    and that is what anyone debugging a stage actually runs -- so the entrypoint
    itself must apply the invariants, and FIRST: invariants.py states that apply()
    must run before vLLM builds an LLM. A source-text assert for the substring
    'invariants.apply()' passes even if the call is moved to the end of main() (after
    the LLM is already built) -- so parse the AST and check main()'s FIRST statement,
    not just that the call is present somewhere in the file.

    IMPORT the module first, do not just grep it. A source-text assert passes even
    when the module cannot import -- e.g. `invariants.apply()` present but
    `invariants` never imported, which is a NameError on the first line of main().
    """
    m = importlib.import_module(mod)                     # fails loudly on a missing import
    tree = ast.parse(Path(m.__file__).read_text())        # Path, not pathlib.Path
    imports_invariants = any(
        isinstance(n, ast.ImportFrom) and n.module == "macronews"
        and any(alias.name == "invariants" for alias in n.names)
        for n in ast.walk(tree)
    )
    assert imports_invariants, f"{mod} calls apply() without importing it"
    main_fn = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "main")
    first = main_fn.body[0]
    assert isinstance(first, ast.Expr) and ast.unparse(first.value) == "invariants.apply()", (
        f"{mod}.main()'s first statement must be invariants.apply(), got: "
        f"{ast.unparse(first)}"
    )


def test_the_facts_that_cost_money_are_still_written_down():
    """Shortening the comments must not delete the ones that cost real money or real
    correctness when forgotten."""
    src = REPO / "src" / "macronews"
    must_survive = {
        "mapping/llm.py":          ["prefix-cache", "evidence_paragraphs"],
        "mapping/grading/llm.py":  ["TRITON_ATTN", "sm_90"],
        "kg/grading/llm.py":       ["TRITON_ATTN"],
        "invariants.py":           ["TP=2", "FA2", "29%"],
        "config/runconfig.py":     ["2000", "3000"],
        "mapping/gate.py":         ["ASSET_GROUP"],
    }
    for rel, needles in must_survive.items():
        text = (src / rel).read_text()
        for needle in needles:
            assert needle in text, f"{rel} no longer mentions {needle!r}"
