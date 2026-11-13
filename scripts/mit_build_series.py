"""Build the cluster GPU power series and a data-quality report.

    python scripts/mit_build_series.py --agg $SCRATCH/dcgridbench/mit_gpu_1min
Outputs (small, kept in the repo's gitignored data/ folder):
    data/processed/mit_cluster_1min.parquet
    results/eda/mit_overview.png, results/eda/mit_quality.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dcgridbench.data.mit_supercloud import build_cluster_series

ap = argparse.ArgumentParser()
ap.add_argument("--agg", required=True)
ap.add_argument("--slurm", default="data/raw/mit_supercloud/slurm-log.csv")
ap.add_argument("--out", default="data/processed/mit_cluster_1min.parquet")
a = ap.parse_args()

s = build_cluster_series(Path(a.agg), Path(a.slurm))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
s.to_parquet(a.out)

# how much of the allocated GPU capacity has telemetry
ratio = (s.gpus_metered / s.gpus_alloc.where(s.gpus_alloc > 0)).clip(upper=2)
q = {
    "start": str(s.index.min()), "end": str(s.index.max()), "n_minutes": len(s),
    "power_kw_mean": s.power_w.mean() / 1e3, "power_kw_max": s.power_w.max() / 1e3,
    "jobs_power_kw_mean": s.power_jobs_w.mean() / 1e3,
    "metered_over_alloc_median": float(ratio.median()),
    "metered_over_alloc_p10": float(ratio.quantile(0.1)),
    "minutes_zero_alloc": int((s.gpus_alloc == 0).sum()),
    "max_5min_ramp_kw": float(s.power_w.resample("5min").mean().diff().abs().max() / 1e3),
}
# share of 5-min power-change variance explained by changes in allocated GPUs
d5 = s[["power_w", "gpus_alloc"]].resample("5min").mean().diff().dropna()
q["r2_dpower_on_dalloc_5min"] = float(d5.power_w.corr(d5.gpus_alloc) ** 2)
Path("results/eda").mkdir(parents=True, exist_ok=True)
json.dump(q, open("results/eda/mit_quality.json", "w"), indent=2)
print(json.dumps(q, indent=2))

h = s.resample("1h").mean()
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
ax[0].plot(h.index, h.power_w / 1e3, lw=0.6); ax[0].set_ylabel("Cluster GPU power (kW)")
ax[1].plot(h.index, h.gpus_alloc, lw=0.6, label="allocated (Slurm)")
ax[1].plot(h.index, h.gpus_metered, lw=0.6, label="with telemetry"); ax[1].set_ylabel("GPUs"); ax[1].legend()
ax[2].plot(h.index, h.gpus_queued, lw=0.6); ax[2].set_ylabel("GPUs queued")
fig.tight_layout(); fig.savefig("results/eda/mit_overview.png", dpi=130)
