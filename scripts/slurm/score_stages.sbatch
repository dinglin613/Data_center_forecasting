#!/bin/bash
#SBATCH --job-name=dcgb_stages
#SBATCH --partition=htc
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/stages_%A_%a.out
# Stage 2-4 scoring of final runs, one task per array element (CPU).
#   sbatch -A <account> scripts/slurm/score_stages.sbatch            # all models
#   sbatch -A <account> --export=ALL,MODELS="nhits tft" scripts/slurm/score_stages.sbatch
#   RESULTS  result directory (default $SCRATCH/dcgridbench_results/benchmark)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV=${DCGB_ENV:-$SCRATCH/dcgridbench_env}
export PATH=$(echo "$PATH" | tr ":" "\n" | grep -v "\.venv" | paste -sd:)
module load python/3.11.11/pytorch/2025.04.01
export PYTHONPATH=$PWD
TASKS=(mit-5min-1h mit-15min-4h mit-1h-24h)
T=${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}
ARGS=(--results "${RESULTS:-$SCRATCH/dcgridbench_results/benchmark}" --task "$T")
if [ -n "${MODELS:-}" ]; then ARGS+=(--model $MODELS); fi
if [ -n "${FORCE:-}" ]; then ARGS+=(--force); fi
"$ENV/bin/python" scripts/score_stages.py "${ARGS[@]}"
