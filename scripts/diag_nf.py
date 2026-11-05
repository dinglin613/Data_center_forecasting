"""Diagnostics for NeuralForecast adapters on one task: config variants scored on val
(and test, for information), plus a check that per-origin prediction matches
NeuralForecast's own predict on a contiguous history.

    python scripts/diag_nf.py --task mit-1h-24h --base nhits --variants '{"exogenous": false}' ...
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from dcgridbench.data.windows import TASKS, origins, resample, target_windows
from dcgridbench.metrics import scores as S
from dcgridbench.models import registry

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="mit-1h-24h")
ap.add_argument("--base", default="nhits")
ap.add_argument("--variants", nargs="*", default=["{}"], help="JSON dicts of parameter overrides")
ap.add_argument("--series", default="data/processed/mit_cluster_1min.parquet")
ap.add_argument("--check-predict", action="store_true")
ap.add_argument("--seeds", type=int, nargs="*", default=[0])
a = ap.parse_args()

task = TASKS[a.task]
df = resample(pd.read_parquet(a.series), task.resolution)
df[task.target] = df[task.target] / 1e3
tr, va, te = (origins(df, task, s) for s in ("train", "val", "test"))
Y = target_windows(df[task.target].to_numpy(), task.horizon)
cfg = registry.load_config(a.base)

for v, seed in [(v, s) for v in a.variants for s in a.seeds]:
    over = json.loads(v)
    c = dict(cfg, params={**registry.resolve_params(cfg, a.task), **over}, task_params={})
    m = registry.build(c, a.task, seed=seed)
    t0 = time.time()
    m.fit(df, tr, task, val_idx=va)
    fit_s = time.time() - t0
    q_va, q_te = m.predict(df, va, task), m.predict(df, te, task)
    lo, hi = np.searchsorted(task.taus, [0.05, 0.95])
    print(json.dumps({"over": over, "seed": seed, "fit_s": round(fit_s, 1), "best_step": m.best_step,
                      "best_val": m.best_val,
                      "val_crps": round(S.crps_from_quantiles(Y[va], q_va, task.taus), 3),
                      "test_crps": round(S.crps_from_quantiles(Y[te], q_te, task.taus), 3),
                      "test_cov90": round(S.coverage(Y[te], q_te[..., lo], q_te[..., hi]), 3)}), flush=True)
    if a.check_predict:
        # NeuralForecast's own predict on the contiguous history up to origin o
        for o in (te[0], te[len(te) // 2]):
            frame = m._frame(df, o + 1 - m.L - 5, o + 1, task)
            fc = m.nf.predict(df=frame)
            from dcgridbench.models.external.neuralforecast_adapter import quantile_columns
            lv, cols = quantile_columns(fc.columns, m.alias)
            ref = np.sort(fc[cols].to_numpy(), axis=-1)  # the adapter removes crossing
            mine = m.predict(df, np.array([o]), task)[0]
            print("predict check origin", o, "max abs diff", float(np.abs(ref - mine).max()), flush=True)
