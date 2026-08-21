#!/bin/bash
#SBATCH --job-name=download_llm
#SBATCH --partition=day
#SBATCH --account=pi_btk22
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/download_llm_%j.out
#SBATCH --error=logs/download_llm_%j.err

set -euo pipefail

module load Python/3.12.3-GCCcore-13.3.0

# SLURM_SUBMIT_DIR is the cwd at sbatch time; expect the user to have
# submitted from their repo root, so this lands in the repo and resolves
# .env / .venv there.
cd "${SLURM_SUBMIT_DIR:?must be invoked via sbatch from the repo root}"
source slurm/_common.sh          # repo-relative: BASH_SOURCE[0] in an sbatch'd script
                                  # points at the SLURM spool copy, not the repo

# Load HF_TOKEN from .env (in the repo root).
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
else
    echo "ERROR: .env not found at $(pwd)/$ENV_FILE — copy .env.example and set HF_TOKEN"
    exit 1
fi

# Override via env: MODEL_REPO=... LOCAL_DIR=... sbatch slurm/download_llm.sh
MODEL_REPO="${MODEL_REPO:-google/gemma-4-26B-A4B-it}"
: "${LOCAL_DIR:?LOCAL_DIR must be set, e.g. LOCAL_DIR=/path/to/scratch/gemma-4-26b-a4b-it (no default; prevents accidental writes to another user scratch dir)}"

# Fetch ONLY what transformers needs to load the model. A bare snapshot_download
# pulls every serialization format the repo happens to ship: bert-large-uncased
# carries .safetensors AND .bin AND .h5 AND .msgpack AND .ot AND a tarball, which
# is 7.7 GB for a 1.25 GB model. roberta-large is 7.1 GB for 1.32 GB. The
# pi_btk22 scratch fileset runs near its 10 TB quota, so this default matters.
#
# Override for a repo that ships no safetensors:
#     ALLOW_PATTERNS='*.bin,*.json,tokenizer*' sbatch slurm/download_llm.sh
ALLOW_PATTERNS="${ALLOW_PATTERNS:-*.safetensors,*.safetensors.index.json,*.json,*.txt,*.model,tokenizer*}"

mkdir -p "$LOCAL_DIR" logs

echo "Downloading $MODEL_REPO → $LOCAL_DIR"
echo "Patterns:  $ALLOW_PATTERNS"
echo "Start: $(date)"

# Use the venv's Python directly — NOT `uv run python` — otherwise uv will
# auto-sync the venv to uv.lock and downgrade vllm/transformers out from under us.
$PY - <<EOF
from huggingface_hub import snapshot_download
pats = [p.strip() for p in "$ALLOW_PATTERNS".split(",") if p.strip()]
snapshot_download(
    repo_id="$MODEL_REPO",
    local_dir="$LOCAL_DIR",
    local_dir_use_symlinks=False,
    allow_patterns=pats,
)
EOF

# Fail loudly rather than leave a directory that looks downloaded but cannot load.
$PY - <<EOF
import sys, pathlib
d = pathlib.Path("$LOCAL_DIR")
weights = list(d.glob("*.safetensors")) + list(d.glob("*.bin"))
if not weights:
    sys.exit(f"ERROR: no weight files in {d} -- ALLOW_PATTERNS excluded everything. "
             "Check what the repo ships and override ALLOW_PATTERNS.")
if not (d / "config.json").exists():
    sys.exit(f"ERROR: no config.json in {d}")
print(f"OK: {len(weights)} weight file(s), config.json present")
EOF

echo "Done: $(date)"
echo "Size: $(du -sh $LOCAL_DIR | cut -f1)"
