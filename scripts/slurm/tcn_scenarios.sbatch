#!/bin/bash
#SBATCH --job-name=qtcn-scen
#SBATCH --partition=htc
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/qtcn_scen_%j.out
cd "$SLURM_SUBMIT_DIR"
for t in mit-5min-1h mit-15min-4h mit-1h-24h; do
  .venv/bin/python scripts/run_scenarios.py --task $t --model qtcn --calib-dir results/calibration_tcn/seed0
done
