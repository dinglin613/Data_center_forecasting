#!/bin/bash
#SBATCH --job-name=dcgb_bench
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/bench_%x_%j.out
# One (model, task, seed) run with the scratch environment from build_env.sbatch.
#   sbatch -A <account> --export=ALL,MODEL=nhits,TASK=mit-1h-24h,SEED=0 scripts/slurm/run_benchmark.sbatch
# Seeds as an array: add --array=0-2 and leave SEED unset.
#   SERIES  processed series (default data/processed/mit_cluster_1min.parquet)
#   OUT     result directory (default results/benchmark)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV=${DCGB_ENV:-$SCRATCH/dcgridbench_env}
CACHE=${DCGB_CACHE:-$SCRATCH/dcgridbench_cache}
export HF_HOME=$CACHE/hf TORCH_HOME=$CACHE/torch XDG_CACHE_HOME=$CACHE PIP_CACHE_DIR=$CACHE/pip
export PATH=$(echo "$PATH" | tr ":" "\n" | grep -v "\.venv" | paste -sd:)
module load python/3.11.11/pytorch/2025.04.01
export PYTHONPATH=$PWD
SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
"$ENV/bin/python" scripts/run_benchmark.py --model "${MODEL:?set MODEL}" --task "${TASK:-mit-1h-24h}" \
  --seed "$SEED" --series "${SERIES:-data/processed/mit_cluster_1min.parquet}" --out "${OUT:-results/benchmark}"
