"""Base models x calibration methods, scored on the test split.

Base models are fit on train. Static calibrators use validation; online ones
start from validation history and keep updating through test with targets
only once they are observed.
    python scripts/run_calibration.py --tasks mit-5min-1h
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aidc_gridbench.data.windows import TASKS, origins, resample, target_windows
from aidc_gridbench.metrics import scores as S
from aidc_gridbench.models.baselines import MODELS
from aidc_gridbench.models.conformal import calibrate_online, calibrate_static
from aidc_gridbench.models.regimes import Regimes

ap = argparse.ArgumentParser()
ap.add_argument("--series", default="data/processed/mit_cluster_1min.parquet")
ap.add_argument("--tasks", nargs="*", default=list(TASKS))
ap.add_argument("--models", nargs="*", default=["persistence", "qlgbm", "qlgbm_rel"])
ap.add_argument("--window-days", type=float, default=14)
ap.add_argument("--out", default="results/calibration")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

raw = pd.read_parquet(a.series)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


def evaluate(y, q, taus, reg):
    taus = np.asarray(taus)
    lo, hi, md = taus.searchsorted(0.05), taus.searchsorted(0.95), taus.searchsorted(0.5)
    r = {"mae_kw": float(np.mean(np.abs(y - q[..., md]))),
         "crps_kw": S.crps_from_quantiles(y, q, taus),
         "cov90": S.coverage(y, q[..., lo], q[..., hi]),
         "width90_kw": S.interval_width(q[..., lo], q[..., hi]),
         "winkler90": S.winkler(y, q[..., lo], q[..., hi], 0.1)}
    r["cov90_by_regime"] = {int(g): S.coverage(y[reg == g], q[reg == g][..., lo], q[reg == g][..., hi])
                            for g in np.unique(reg)}
    return r


for tname in a.tasks:
    task = TASKS[tname]
    df = resample(raw, task.resolution)
    df[task.target] = df[task.target] / 1e3
    tr, va, te = (origins(df, task, s) for s in ("train", "val", "test"))
    Y = target_windows(df[task.target].to_numpy(), task.horizon)
    reg_model = Regimes().fit(df, tr, task)
    reg_va, reg_te = reg_model.label(df, va, task), reg_model.label(df, te, task)
    stream = np.concatenate([va, te])
    reg_stream = np.concatenate([reg_va, reg_te])
    window = int(a.window_days * 24 * 60 / task.dt_min)
    step = max(1, int(60 / task.dt_min))  # refresh corrections hourly
    res = {"regime_share_test": {int(g): float(np.mean(reg_te == g)) for g in np.unique(reg_te)}}
    for mname in a.models:
        if mname == "qtcn":  # torch is optional; import only when asked
            from aidc_gridbench.models.tcn import QuantileTCN
            m = QuantileTCN(seed=a.seed).fit(df, tr, task, val_idx=va)
            print(f"{tname} qtcn best val pinball (scaled) {m.best_val:.4f}", flush=True)
        else:
            m = MODELS[mname]().fit(df, tr, task)
        q_va, q_te = m.predict(df, va, task), m.predict(df, te, task)
        q_stream = np.concatenate([q_va, q_te])
        np.savez_compressed(out / f"{tname}_{mname}_stream.npz", q=q_stream.astype(np.float32),
                            y=Y[stream].astype(np.float32), t=stream, n_val=len(va), reg=reg_stream)
        variants = {
            "raw": q_te,
            "static": calibrate_static(q_va, Y[va], q_te, task.taus),
            "regime": calibrate_static(q_va, Y[va], q_te, task.taus, reg_va, reg_te),
            "online": calibrate_online(stream, Y[stream], q_stream, task.taus, len(va), window, step=step),
            "online_regime": calibrate_online(stream, Y[stream], q_stream, task.taus, len(va), window,
                                              reg=reg_stream, step=step),
        }
        res[mname] = {}
        for vname, q in variants.items():
            res[mname][vname] = evaluate(Y[te], q, task.taus, reg_te)
            np.savez_compressed(out / f"{tname}_{mname}_{vname}.npz", q=q.astype(np.float32))
            r = res[mname][vname]
            print(f"{tname:13s} {mname:12s} {vname:14s} mae={r['mae_kw']:.3f} crps={r['crps_kw']:.3f} "
                  f"cov90={r['cov90']:.3f} width={r['width90_kw']:.2f} wink={r['winkler90']:.2f} "
                  f"byreg={ {k: round(v, 2) for k, v in r['cov90_by_regime'].items()} }", flush=True)
    np.savez_compressed(out / f"{tname}_truth.npz", y=Y[te].astype(np.float32), reg=reg_te,
                        origins=df.index[te].astype("int64"))
    json.dump(res, open(out / f"{tname}.json", "w"), indent=2)
