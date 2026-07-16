"""Settings that are not configuration and must never vary.

Each was measured, and each fails silently rather than loudly:

  VLLM_BATCH_INVARIANT=1    Without it the same (article, group) pair can flip
                            verdict between runs: ~29% tag drift, against ~0.1%
                            with it. It lived only in the SLURM scripts, so any
                            stage run by hand quietly produced different numbers.
  temperature = 0.0         Greedy decode. Anything else is not reproducible.
  tensor_parallel_size = 1  The weights fit one B200. TP=2 was 18% slower (the
                            all-reduce tax) and changes numerics.
  attention backend         Qwen2-arch (both graders) needs TRITON_ATTN to be
                            batch-invariant. FA2 is sm_90-only and crashes on
                            B200/sm_100. No env var works -- it goes in the LLM ctor.

TENSOR_PARALLEL_SIZE is the one value meant to be CONSUMED by the stages: there is no
per-entrypoint --tensor-parallel-size flag, only this constant.
TEMPERATURE and ATTENTION_BACKEND are here so an artifact can RECORD what it ran under; the
five call sites still set them literally, and test_every_llm_call_site_uses_greedy_decode
pins that. Rewiring them through this module would be a behaviour-neutral change with a
behaviour-changing blast radius, so it is deliberately not done.

apply() must run before vLLM builds an LLM. vLLM is imported lazily (all 18
`from vllm import ...` lines are inside functions), so calling this first from an
entrypoint is early enough.
"""

import os

TEMPERATURE = 0.0
TENSOR_PARALLEL_SIZE = 1
ATTENTION_BACKEND = "TRITON_ATTN"

_BATCH_INVARIANT = "VLLM_BATCH_INVARIANT"


def apply() -> None:
    """Pin the environment invariants. Refuses to run with them turned off."""
    current = os.environ.get(_BATCH_INVARIANT)
    if current is not None and current != "1":
        raise RuntimeError(
            f"{_BATCH_INVARIANT}={current!r} is set in the environment. It is an "
            f"invariant, not a setting: without it the same (article, group) pair "
            f"can flip verdict between runs (~29% tag drift vs ~0.1%). Unset it."
        )
    os.environ[_BATCH_INVARIANT] = "1"


def record() -> dict:
    """What an artifact carries so its numbers can be reproduced.

    Refuses to describe a run that is not actually under the invariants. An artifact
    claiming batch-invariance it did not have is worse than one claiming nothing --
    it is why a whole A/B had to be thrown away.
    """
    if os.environ.get(_BATCH_INVARIANT) != "1":
        raise RuntimeError(
            "invariants.apply() has not run, so this artifact cannot record what it "
            "was produced under. Call it at the entrypoint, before vLLM loads."
        )
    return {
        "vllm_batch_invariant": os.environ[_BATCH_INVARIANT],
        "temperature": TEMPERATURE,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "attention_backend": ATTENTION_BACKEND,
    }
