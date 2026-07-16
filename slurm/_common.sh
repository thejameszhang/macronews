#!/bin/bash
# Shared launcher setup. Source this; do not execute it.
#
# The same forty lines were pasted across eleven launchers, and that is how the
# billed-partition default got fixed in the single-shard launchers and missed in the
# array ones -- the launchers a full-corpus run actually uses (424de57). One
# definition, one place to fix, and tests/test_launchers.py stops it regressing.

# Repo root, from THIS file's own location -- BASH_SOURCE[0] is _common.sh, not the
# caller. That is correct in both contexts:
#   * interactive:  slurm/_common.sh -> dirname slurm -> /.. -> repo root.
#   * array worker: the worker already did `cd "$SLURM_SUBMIT_DIR"` and sources
#     `slurm/_common.sh` by repo-relative path, so this resolves to the repo root too.
#
# Do NOT prefer $SLURM_SUBMIT_DIR here. salloc sets it (claude.sh opens one), so a
# launcher run from a git worktree or a feature checkout inside an salloc would cd into
# whichever tree the salloc started from -- silently submitting the WRONG code and
# writing into the WRONG results/.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# `source .venv/bin/activate` alone is NOT enough on Bouchet: bare .venv/bin/python
# dies with a libpython load error. module load puts libpython3.12.so.1.0 on
# LD_LIBRARY_PATH. We still call the venv python by ABSOLUTE path, because module load
# re-prepends its own bin/ and a bare `python` resolves to the system interpreter
# (which has no vllm).
#
# NO `|| true` here. Swallowing a failed module load produces exactly the libpython
# crash it exists to prevent -- on eleven launchers at once.
module load Python/3.12.3-GCCcore-13.3.0
PY=".venv/bin/python"

# FREE tiers are the DEFAULT. priority_gpu/prio_btk22 BILLS the lab's paid allocation
# -- it is for GPU B200 production when the normal queue is too slow, and NEVER for CPU
# work. Separate vars per tier: one shared $PARTITION would mean steering a GPU job
# silently re-points the CPU ones.
GPU_PARTITION="${GPU_PARTITION:-gpu_b200}"
GPU_ACCOUNT="${GPU_ACCOUNT:-pi_btk22}"
GPU_QOS="${GPU_QOS:-normal}"
GPU_GRES="${GPU_GRES:-gpu:b200:1}"
CPU_PARTITION="${CPU_PARTITION:-day}"
CPU_ACCOUNT="${CPU_ACCOUNT:-pi_btk22}"

# a1117u29n01 is ~3.4x slower than its peers (6h35m vs ~2h on the 2014 prod run). Set
# EXCLUDE_NODES= (empty) to allow it back if Yale fixes it.
EXCLUDE_NODES="${EXCLUDE_NODES-a1117u29n01}"
EXCLUDE_FLAG=""
[ -n "$EXCLUDE_NODES" ] && EXCLUDE_FLAG="--exclude=${EXCLUDE_NODES}"
