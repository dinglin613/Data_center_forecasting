#!/bin/bash
#SBATCH --job-name=mit-agg
#SBATCH --partition=htc
#SBATCH --array=0-99%40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mit_agg_%a.out
# Intermediate per-minute job power goes to $SCRATCH (~1.2 GB, regenerable).
# Folders that already have a parquet are skipped, so the array can be resubmitted.
cd "$SLURM_SUBMIT_DIR"
OUT=${AIDC_INTERIM:-$SCRATCH/dcgridbench/mit_gpu_1min}
mkdir -p "$OUT"
.venv/bin/python scripts/mit_aggregate.py --array-index "$SLURM_ARRAY_TASK_ID" --workers 12 --out "$OUT"
