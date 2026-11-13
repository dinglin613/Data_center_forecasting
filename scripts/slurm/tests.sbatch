#!/bin/bash
#SBATCH --job-name=dcgb_tests
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tests_%j.out
# Unit tests on a compute node:  sbatch -A <account> scripts/slurm/tests.sbatch
# Interactive equivalent: srun -A <account> -p gpu-dev --gres=gpu:1 bash scripts/slurm/tests.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
ENV=${DCGB_ENV:-$SCRATCH/dcgridbench_env}
CACHE=${DCGB_CACHE:-$SCRATCH/dcgridbench_cache}
export HF_HOME=$CACHE/hf TORCH_HOME=$CACHE/torch XDG_CACHE_HOME=$CACHE
export PATH=$(echo "$PATH" | tr ":" "\n" | grep -v "\.venv" | paste -sd:)
module load python/3.11.11/pytorch/2025.04.01
export PYTHONPATH=$PWD
"$ENV/bin/python" -m pytest -q -p no:cacheprovider "${@:-tests}"
