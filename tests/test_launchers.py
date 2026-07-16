"""No launcher may reach the billed allocation.

priority_gpu / prio_btk22 BILL the lab's paid allocation. The same billed default
shipped twice because the block was copy-pasted; _common.sh is the single definition
and this is the guard.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SLURM = REPO / "slurm"


def _every_launcher():
    """Every script that can submit an sbatch job -- INCLUDING the gitignored ones.

    scripts/misc/ is untracked, so on a fresh clone this finds nothing there and the guard
    falls back to slurm/ alone. That is fine. What is NOT fine is a launcher that bills the
    lab's allocation escaping the check merely because it lives somewhere untracked: money
    is spent from a working tree, not from a git index.
    """
    return sorted(SLURM.glob("*.sh")) + sorted((REPO / "scripts").rglob("*.sh"))


def test_no_launcher_can_reach_the_billed_tier():
    """Match the bare strings, not just the ${VAR:-default} form -- a hardcoded
    `#SBATCH --partition=priority_gpu` would sail past a `:-priority` check.
    `#SBATCH` lines are live directives here (see download_llm.sh), so they must
    NOT be skipped as comments the way plain prose comments are.
    """
    offenders = []
    for sh in _every_launcher():
        for i, line in enumerate(sh.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") and not stripped.startswith("#SBATCH"):
                continue        # prose comment -- but NEVER skip an #SBATCH directive
            if "priority" in stripped or "prio_btk22" in stripped:
                offenders.append(f"{sh.name}:{i}: {stripped}")
    assert not offenders, "these can reach the BILLED allocation:\n" + "\n".join(offenders)


def test_every_launcher_sources_common():
    for sh in sorted(SLURM.glob("*.sh")):
        if sh.name == "_common.sh":
            continue
        assert "_common.sh" in sh.read_text(), f"{sh.name} does not source _common.sh"


def test_no_launcher_still_points_at_the_old_layout():
    for sh in sorted(SLURM.glob("*.sh")):
        text = sh.read_text()
        assert "PYTHONPATH=src" not in text, f"{sh.name} still sets PYTHONPATH=src"
        assert "src/pipeline.py" not in text, f"{sh.name} still calls src/pipeline.py"
        assert "-m src.kg" not in text, f"{sh.name} still uses the src.kg accident"
        # Every launcher moved scripts/*.sh -> slurm/*.sh, and sbatch is handed the array
        # worker BY PATH, so a stale `scripts/_run_pipeline_task.sh` would submit a job
        # that dies on the node. Match any .sh under scripts/ -- note the leading
        # underscore: a check for "scripts/run_" does NOT catch "scripts/_run_".
        #
        # Only the MOVED launchers are stale (scripts/run_*.sh, scripts/_run_*_task.sh).
        # The gate-measurement launcher lives under scripts/misc/keywords/ by design and
        # invokes a .py, not a .sh, so it never matches this.
        stale = re.findall(r"scripts/[\w/-]*\.sh", text)
        assert not stale, f"{sh.name} still invokes a moved launcher: {stale}"


def test_no_launcher_hardcodes_the_repo_root():
    """build_kg_graph.sh had `cd /nfs/roberts/.../macronews`. That is the exact hazard
    _common.sh exists to kill: run from a worktree, it silently executes main's checkout."""
    for sh in sorted(SLURM.glob("*.sh")):
        assert "cd /nfs/roberts" not in sh.read_text(), f"{sh.name} hardcodes the repo root"


def test_every_array_launcher_honors_dry_run():
    """run_tabular.sh used to ignore DRY_RUN while its two siblings honored it, so
    `DRY_RUN=1 bash slurm/run_tabular.sh` submitted a live 356-task array that rewrote
    every month of results/tabular/. One launcher opting out of the convention is worse
    than none having it: the habit transfers, the safety does not."""
    for name in ("run_pipeline_array.sh", "run_grader_array.sh", "run_tabular.sh"):
        text = (SLURM / name).read_text()
        assert 'DRY_RUN:-0' in text, f"{name} ignores DRY_RUN -- it submits for real"
