#!/bin/bash
#SBATCH --job-name=calib-scen
#SBATCH --partition=htc
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cpu_pipeline_%j.out
cd "$SLURM_SUBMIT_DIR"
set -e
.venv/bin/python scripts/run_calibration.py
for t in mit-5min-1h mit-15min-4h mit-1h-24h; do
  for m in persistence qlgbm_rel; do
    .venv/bin/python scripts/run_scenarios.py --task $t --model $m
  done
done
