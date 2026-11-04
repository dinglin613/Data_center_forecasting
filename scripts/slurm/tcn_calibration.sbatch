#!/bin/bash
#SBATCH --job-name=qtcn
#SBATCH --partition=gpu-dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/qtcn_%A_%a.out
# One seed per array task; seed 0 also runs the scenario experiments.
cd "$SLURM_SUBMIT_DIR"
export PATH=$(echo "$PATH" | tr ":" "\n" | grep -v "\.venv" | paste -sd:)
module load python/3.11.11/pytorch/2025.04.01
export PYTHONPATH=$PWD
S=$SLURM_ARRAY_TASK_ID
python scripts/run_calibration.py --models qtcn --seed $S --out results/calibration_tcn/seed$S
if [ "$S" = 0 ]; then
  for t in mit-5min-1h mit-15min-4h mit-1h-24h; do
    python scripts/run_scenarios.py --task $t --model qtcn --calib-dir results/calibration_tcn/seed0
  done
fi
