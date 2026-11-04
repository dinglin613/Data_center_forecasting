"""Stream-aggregate MIT Supercloud GPU folders. Usage:
    python scripts/mit_aggregate.py --folders 0 1 2 --out data/interim/mit_gpu_1min
    python scripts/mit_aggregate.py --array-index $SLURM_ARRAY_TASK_ID  (one folder per task)
"""
import argparse
from pathlib import Path
from aidc_gridbench.data.mit_supercloud import aggregate_folder

ap = argparse.ArgumentParser()
ap.add_argument("--folders", type=int, nargs="*")
ap.add_argument("--array-index", type=int)
ap.add_argument("--out", default="data/interim/mit_gpu_1min")
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--limit", type=int)
a = ap.parse_args()
folders = a.folders if a.folders else [a.array_index]
for f in folders:
    print(aggregate_folder(f, Path(a.out), limit=a.limit, workers=a.workers), flush=True)
