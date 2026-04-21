#!/bin/bash
#SBATCH --job-name=build_vllm
#SBATCH --partition=day
#SBATCH --account=pi_btk22
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/build_vllm_%j.out
#SBATCH --error=logs/build_vllm_%j.err

set -euo pipefail

module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.9.1
module load CMake/3.31.8-GCCcore-13.3.0
module load Ninja/1.12.1-GCCcore-13.3.0

export CUDA_HOME=/apps/software/system/software/CUDA/12.9.1
export MAX_JOBS=8

cd /nfs/roberts/project/pi_btk22/jyz32/macronews

PYTHON312=/apps/software/2024a/software/Python/3.12.3-GCCcore-13.3.0/bin/python3.12

echo "=== Recreating venv with stdlib venv module (no uv) ==="
echo "Python: $($PYTHON312 --version)"
rm -rf .venv
$PYTHON312 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "=== Clearing any stale pip build cache for vllm ==="
pip cache remove vllm 2>/dev/null || true

echo "=== Installing vllm 0.19.0 from PyPI sdist ==="
echo "ninja: $(which ninja)"
echo "cmake: $(which cmake)"
echo "nvcc:  $(which nvcc)"
echo "Start: $(date)"

# Let pip handle build isolation — it will resolve its own torch for the build.
# This matches the approach that successfully compiled in job 8012462/8015489.
pip install --verbose vllm==0.19.0

echo "=== Installed package list ==="
pip list | grep -iE "vllm|torch|transformers"

echo "=== Verifying Gemma4 support ==="
python -c "
import vllm
print(f'vllm version: {vllm.__version__}')
from vllm.model_executor.models.registry import _VLLM_MODELS
gemma = {k: v for k, v in _VLLM_MODELS.items() if 'emma' in k}
print(f'Gemma models in registry: {gemma}')
print(f'Total models: {len(_VLLM_MODELS)}')
"

echo "Done: $(date)"
