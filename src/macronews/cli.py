"""`macronews` -- one entrypoint for every stage.

    macronews mapper run        articles -> asset groups
    macronews mapper grade      QwQ verifies the mapper's tags
    macronews tabular           precompute the tabular-body sidecar

The mapper lane resolves its parameters through config.runconfig, so a run cannot
start with a value nobody meant.

Dispatch is by hand, not argparse subparsers: a subparser per lane consumes --help
before the stage sees it, and since the mapper's own main() is gone, this is the only
help surface there is.

invariants.apply() runs first: vLLM reads VLLM_BATCH_INVARIANT when it builds an LLM,
and without it the same (article, group) pair can flip verdict between runs. It used
to live only in the SLURM scripts, so a hand-run silently lost it.
"""

import argparse
import logging
import runpy
import sys
from pathlib import Path

from pydantic import ValidationError

from djnw import runtime as invariants


def _passthrough(module: str, argv: list[str]) -> None:
    """Hand straight to a stage that owns its own argparse."""
    sys.argv = [module, *argv]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def _run_experiment(cfg) -> None:            # seam for tests
    from macronews.pipeline import run_experiment
    run_experiment(cfg)


def _run_grade(cfg) -> None:                 # seam for tests
    from macronews.mapping.grading.runner import grade
    grade(cfg)


def _echo(cfg) -> None:
    """Print what the run resolved to, before it costs a GPU-hour."""
    logging.info("=== %s ===", type(cfg).__name__)
    for key, value in cfg.record().items():
        logging.info("  %-20s %s", key, value)


def _build(cls, parser, argv):
    """Parse, then validate. A bad value exits with the parser's usage, not a traceback.

    Catch ValidationError, not ValueError: the latter is broad enough to swallow bugs
    from anywhere in the call stack. (Note pydantic wraps a validator's ValueError --
    including the json.JSONDecodeError from a corrupt config.json -- INTO a
    ValidationError, so a corrupt weights file still reports as a usage error. Fixing
    that means validating the model file outside the validator; out of scope here.)
    """
    args = parser.parse_args(argv)
    try:
        return cls(**{k: v for k, v in vars(args).items() if v is not None})
    except ValidationError as e:
        parser.error(str(e))


def _mapper_run(argv: list[str]) -> None:
    from macronews.config.runconfig import MapperConfig

    p = argparse.ArgumentParser(prog="macronews mapper run")
    p.add_argument("--dataset", choices=["gold", "sports", "wikigaming", "djnw"],
                   help="Inferred as djnw when --input-file is given")
    p.add_argument("--input-file", type=Path, help="A single djnw *_clean.jsonl shard")
    p.add_argument("--sample-dir", type=Path, help="Override the dataset's default dir")
    p.add_argument("--output-file", type=Path, help="Write one JSONL here")
    p.add_argument("--output-dir", type=Path, help="Derive the name from --input-file")
    p.add_argument("--model", type=Path)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--max-articles", type=int)
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--random-seed", type=int)
    gate = p.add_mutually_exclusive_group()
    gate.add_argument("--keyword-gate", dest="keyword_gate", action="store_true",
                      default=None, help="Force the gate on (default for djnw)")
    gate.add_argument("--no-keyword-gate", dest="keyword_gate", action="store_false",
                      default=None,
                      help="Call the model on every pair. Default for gold/sports/"
                           "wikigaming -- they are instruments, not production.")
    p.add_argument("--report-stats", dest="report_stats", action="store_true",
                   default=None,
                   help="Have vLLM report prefix-cache hit rate and prefill/decode "
                        "throughput. Off in production; used by the compute-cost "
                        "benchmark. Changes reporting only, never computation.")

    cfg = _build(MapperConfig, p, argv)
    _echo(cfg)
    _run_experiment(cfg)


def _mapper_grade(argv: list[str]) -> None:
    from macronews.config.runconfig import GraderConfig

    p = argparse.ArgumentParser(prog="macronews mapper grade")
    p.add_argument("--mapper-output", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", required=True, choices=["gold", "djnw"])
    p.add_argument("--input-file", type=Path, help="[djnw] the source *_clean.jsonl")
    p.add_argument("--sample-dir", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--max-model-len", type=int)
    p.add_argument("--max-tokens", type=int)

    cfg = _build(GraderConfig, p, argv)
    _echo(cfg)
    _run_grade(cfg)


# Stage NAMES, not function objects. A dict of objects captures them at import, so
# monkeypatch.setattr(cli, "_mapper_run", ...) would rebind the module attribute and
# the dict would still call the original.
_MAPPER_STAGES = {"run": "_mapper_run", "grade": "_mapper_grade"}


def main() -> None:
    invariants.apply()  # before any stage imports vLLM
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return

    lane, rest = argv[0], argv[1:]

    if lane == "tabular":
        return _passthrough("macronews.tabular.runner", rest)

    if lane != "mapper":
        sys.exit(f"unknown lane {lane!r}. Try: macronews --help")

    stages = _MAPPER_STAGES
    if not rest:
        sys.exit(f"macronews {lane} <stage>: {', '.join(sorted(stages))}")

    stage, rest = rest[0], rest[1:]
    if stage not in stages:
        sys.exit(f"unknown {lane} stage {stage!r}: {', '.join(sorted(stages))}")

    getattr(sys.modules[__name__], _MAPPER_STAGES[stage])(rest)   # resolved now


if __name__ == "__main__":
    main()
