"""Fit baselines on each task and score them on the test split.

    python scripts/run_baselines.py --tasks mit-5min-1h mit-1h-24h
Writes results/baselines/<task>.json and the test quantiles as .npz.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aidc_gridbench.data.windows import TASKS, origins, resample, target_windows
from aidc_gridbench.metrics import scores as S
from aidc_gridbench.models.baselines import MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--series", default="data/processed/mit_cluster_1min.parquet")
ap.add_argument("--tasks", nargs="*", default=list(TASKS))
ap.add_argument("--models", nargs="*", default=list(MODELS))
ap.add_argument("--out", default="results/baselines")
a = ap.parse_args()

raw = pd.read_parquet(a.series)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

for tname in a.tasks:
    task = TASKS[tname]
    df = resample(raw, task.resolution)
    df[task.target] = df[task.target] / 1e3  # kW
    tr, te = origins(df, task, "train"), origins(df, task, "test")
    y = target_windows(df[task.target].to_numpy(), task.horizon)[te]
    taus = np.array(task.taus)
    lo, hi = taus.searchsorted(0.05), taus.searchsorted(0.95)
    res = {}
    for mname in a.models:
        q = MODELS[mname]().fit(df, tr, task).predict(df, te, task)
        med = q[..., taus.searchsorted(0.5)]
        res[mname] = {
            "mae_kw": float(np.mean(np.abs(y - med))),
            "rmse_kw": float(np.sqrt(np.mean((y - med) ** 2))),
            "crps_kw": S.crps_from_quantiles(y, q, taus),
            "cov90": S.coverage(y, q[..., lo], q[..., hi]),
            "width90_kw": S.interval_width(q[..., lo], q[..., hi]),
            "winkler90": S.winkler(y, q[..., lo], q[..., hi], 0.1),
            "mae_by_h": np.mean(np.abs(y - med), axis=0).round(3).tolist(),
        }
        np.savez_compressed(out / f"{tname}_{mname}.npz", q=q.astype(np.float32), y=y.astype(np.float32),
                            origins=df.index[te].astype("int64"))
        print(tname, mname, {k: round(v, 3) for k, v in res[mname].items() if k != "mae_by_h"}, flush=True)
    json.dump({"task": tname, "n_test": int(len(te)), "n_train": int(len(tr)), "results": res},
              open(out / f"{tname}.json", "w"), indent=2)
