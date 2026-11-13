"""TFT val->test gap: bug or genuine? (Phase-1 config: scheduler inputs on, robust scaler.)

1. adapter prediction vs NeuralForecast's own predict on a contiguous history (bug check);
2. the same model with scheduler inputs off;
3. the scheduler-on model with test-time scheduler inputs clipped to the train range;
4. how far the scheduler features in val/test leave the train range.

    python scripts/diag_tft.py --series <parquet>
"""
import argparse
import json

import numpy as np
import pandas as pd

from dcgridbench.data.windows import TASKS, origins, resample, target_windows
from dcgridbench.metrics import scores as S
from dcgridbench.models.external.base import SCHED_COLS
from dcgridbench.models.external.neuralforecast_adapter import NeuralForecastModel, quantile_columns

ap = argparse.ArgumentParser()
ap.add_argument("--series", default="data/processed/mit_cluster_1min.parquet")
ap.add_argument("--task", default="mit-1h-24h")
a = ap.parse_args()

task = TASKS[a.task]
df = resample(pd.read_parquet(a.series), task.resolution)
df[task.target] = df[task.target] / 1e3
tr, va, te = (origins(df, task, s) for s in ("train", "val", "test"))
Y = target_windows(df[task.target].to_numpy(), task.horizon)
lo_, hi_ = np.searchsorted(task.taus, [0.05, 0.95])

rng_tr = {c: (float(df[c].iloc[tr].min()), float(df[c].iloc[tr].max())) for c in SCHED_COLS}
for s, idx in (("val", va), ("test", te)):
    out = {c: float(np.mean((df[c].iloc[idx] < rng_tr[c][0]) | (df[c].iloc[idx] > rng_tr[c][1]))) for c in SCHED_COLS}
    mx = {c: float(df[c].iloc[idx].max()) for c in SCHED_COLS}
    print(f"{s}: share of origins outside the train range {out}; max {mx}; train range {rng_tr}", flush=True)

base = dict(model="TFT", loss="mqloss", lookback_hours=168, scaler_type="robust", hidden_size=64, n_head=4,
            dropout=0.1, max_steps=3000, learning_rate=0.001, batch_size=32, windows_batch_size=256,
            val_check_steps=50, early_stop_patience_steps=10, calendar=False, seed=0)


def score(m, frame, name):
    qv, qt = m.predict(frame, va, task), m.predict(frame, te, task)
    c = S.crps_from_quantiles(Y[te], qt, task.taus)
    weekly = pd.Series(2 * np.mean(np.maximum(np.asarray(task.taus) * (Y[te][..., None] - qt),
                                              (np.asarray(task.taus) - 1) * (Y[te][..., None] - qt)), axis=(1, 2)),
                       index=df.index[te]).resample("W").mean().round(2).tolist()
    print(json.dumps({"run": name, "val_crps": round(S.crps_from_quantiles(Y[va], qv, task.taus), 3),
                      "test_crps": round(c, 3), "test_cov90": round(S.coverage(Y[te], qt[..., lo_], qt[..., hi_]), 3),
                      "test_bias": round(float(np.mean(qt[..., 4] - Y[te])), 3), "weekly_test_crps": weekly}), flush=True)
    return qt


for exog in (True, False):
    m = NeuralForecastModel(exogenous=exog, **base).fit(df, tr, task, val_idx=va)
    q = score(m, df, f"exog={exog}")
    o = te[0]
    fc = m.nf.predict(df=m._frame(df, o + 1 - m.L - 5, o + 1, task))
    ref = np.sort(fc[quantile_columns(fc.columns, m.alias)[1]].to_numpy(), axis=-1)
    print("  predict-path check: max |adapter - native| =", float(np.abs(ref - q[0]).max()), flush=True)
    if exog:
        clipped = df.copy()
        for c in SCHED_COLS:
            clipped[c] = clipped[c].clip(*rng_tr[c])
        score(m, clipped, "exog=True, test inputs clipped to train range")
