#!/bin/bash
#SBATCH --job-name=dcgb_env
#SBATCH --partition=htc
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_env_%j.out
# Build the environment for external models (NeuralForecast, Chronos, ...) on scratch.
# Run from the repo root:
#   sbatch -A <account> scripts/slurm/build_env.sbatch
# Optional overrides (export before sbatch):
#   DCGB_ENV    venv path            (default $SCRATCH/dcgridbench_env)
#   DCGB_CACHE  pip / HF / torch cache (default $SCRATCH/dcgridbench_cache)
#   DCGB_PKGS   packages to install  (default: the pins below)
#   DCGB_OPT_PKGS  optional packages; a failure is logged, not fatal
#   DCGB_REBUILD=1  delete and recreate the venv
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV=${DCGB_ENV:-$SCRATCH/dcgridbench_env}
CACHE=${DCGB_CACHE:-$SCRATCH/dcgridbench_cache}
PKGS=${DCGB_PKGS:-"neuralforecast==3.1.9 chronos-forecasting==2.3.2"}
OPT_PKGS=${DCGB_OPT_PKGS:-"timesfm[torch]==3.0.2 gluonts==0.16.2"}
# gluonts 0.16.2 without its [torch] extra: the extra pins lightning<2.5, the module has 2.5.1.
mkdir -p "$CACHE"
export PIP_CACHE_DIR=$CACHE/pip HF_HOME=$CACHE/hf TORCH_HOME=$CACHE/torch XDG_CACHE_HOME=$CACHE

export PATH=$(echo "$PATH" | tr ":" "\n" | grep -v "\.venv" | paste -sd:)
module load python/3.11.11/pytorch/2025.04.01
BASE_PY=$(command -v python)
echo "base python: $BASE_PY"

if [ "${DCGB_REBUILD:-0}" = 1 ]; then rm -rf "$ENV"; fi
if [ ! -x "$ENV/bin/python" ]; then
  "$BASE_PY" -m venv --system-site-packages "$ENV"
fi
PY=$ENV/bin/python
# Keep the module's torch build (2.6): pin it so pip never replaces it.
# neuralforecast>=3.2 needs torch>=2.9.1, hence 3.1.9.
"$PY" -c "import torch; print(f'torch=={torch.__version__.split(\"+\")[0]}')" > "$CACHE/constraints.txt"
"$PY" -c "import numpy; print(f'numpy=={numpy.__version__}')" >> "$CACHE/constraints.txt"
cat "$CACHE/constraints.txt"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -c "$CACHE/constraints.txt" $PKGS
"$PY" -m pip install -c "$CACHE/constraints.txt" --no-deps -e .
"$PY" -m pip install -c "$CACHE/constraints.txt" pytest pyyaml lightgbm
for p in $OPT_PKGS; do
  "$PY" -m pip install -c "$CACHE/constraints.txt" "$p" || echo "OPTIONAL INSTALL FAILED: $p"
done

"$PY" - <<'EOF'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
for m in ("numpy", "pandas", "neuralforecast", "pytorch_lightning", "chronos", "timesfm", "dcgridbench"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "NOT AVAILABLE:", type(e).__name__, e)
EOF
"$PY" -m pip freeze > "$CACHE/pip_freeze_$(date +%Y%m%d).txt"
echo "done: $ENV"
