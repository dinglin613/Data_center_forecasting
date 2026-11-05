#!/bin/bash
#SBATCH --job-name=dcgb_jobs
#SBATCH --partition=fp-gpgpu-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/joblist_%A_%a.out
# Run a job list in chunks: array task i runs lines [i*PER_TASK, (i+1)*PER_TASK), PAR at a time.
#   sbatch -A <account> --array=0-9 --export=ALL,JOBS=jobs/tune_1h.jsonl,PER_TASK=16,PAR=4 scripts/slurm/joblist.sbatch
# CPU-only lists: add -p htc --gres=none (or override on the command line).
#   OUT     result directory (default $SCRATCH/dcgridbench_results/benchmark)
#   SERIES  processed series (default data/processed/mit_cluster_1min.parquet)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV=${DCGB_ENV:-$SCRATCH/dcgridbench_env}
CACHE=${DCGB_CACHE:-$SCRATCH/dcgridbench_cache}
export HF_HOME=$CACHE/hf TORCH_HOME=$CACHE/torch XDG_CACHE_HOME=$CACHE PIP_CACHE_DIR=$CACHE/pip
export PATH=$(echo "$PATH" | tr ":" "\n" | grep -v "\.venv" | paste -sd:)
module load python/3.11.11/pytorch/2025.04.01
export PYTHONPATH=$PWD OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PER=${PER_TASK:?set PER_TASK}
I=${SLURM_ARRAY_TASK_ID:-0}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$ENV/bin/python" scripts/run_joblist.py --jobs "${JOBS:?set JOBS}" --start $((I * PER)) --count "$PER" \
  --par "${PAR:-4}" --out "${OUT:-$SCRATCH/dcgridbench_results/benchmark}" \
  --series "${SERIES:-data/processed/mit_cluster_1min.parquet}"
