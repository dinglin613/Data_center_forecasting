"""Fit one configured model on one task and seed; write the standard result JSON.

    python scripts/run_benchmark.py --model nhits --task mit-1h-24h --seed 0

Writes  <out>/<task>/<model>/seed<k>.json        (result record, docs/result_schema.md)
        <out>/<task>/<model>/seed<k>_stream.npz  (val+test quantiles, same layout as
                                                  run_calibration.py, so run_scenarios.py can read it)
        <out>/<task>/<model>/seed<k>_samples.npz (test sample paths, native-sample models only)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dcgridbench.data.windows import TASKS, origins, resample, target_windows
from dcgridbench.leaderboard import results as R
from dcgridbench.models import registry
from dcgridbench.models.external.base import library_version
from dcgridbench.models.regimes import Regimes

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="config name in configs/models or a YAML path")
ap.add_argument("--task", default="mit-1h-24h", choices=list(TASKS))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--series", default="data/processed/mit_cluster_1min.parquet")
ap.add_argument("--out", default="results/benchmark")
ap.add_argument("--n-samples", type=int, default=100, help="sample paths for native-sample models")
a = ap.parse_args()

cfg = registry.load_config(a.model)
task = TASKS[a.task]
df = resample(pd.read_parquet(a.series), task.resolution)
df[task.target] = df[task.target] / 1e3  # kW
tr, va, te = (origins(df, task, s) for s in ("train", "val", "test"))
Y = target_windows(df[task.target].to_numpy(), task.horizon)
reg_model = Regimes().fit(df, tr, task)
reg_va, reg_te = reg_model.label(df, va, task), reg_model.label(df, te, task)

np.random.seed(a.seed)
model = registry.build(cfg, a.task, seed=a.seed)
t0 = time.time()
registry.fit_model(model, df, tr, task, va)
fit_s = time.time() - t0
t0 = time.time()
q = model.predict(df, np.concatenate([va, te]), task)
pred_s = time.time() - t0
q_va, q_te = q[:len(va)], q[len(va):]
assert q.shape == (len(va) + len(te), task.horizon, len(task.taus)), q.shape
assert np.isfinite(q).all(), "non-finite quantiles"
print(f"{a.task} {cfg['name']} seed{a.seed}: fit {fit_s:.1f}s predict {pred_s:.1f}s", flush=True)

out = Path(a.out) / a.task / cfg["name"]
out.mkdir(parents=True, exist_ok=True)
stream = np.concatenate([va, te])
np.savez_compressed(out / f"seed{a.seed}_stream.npz", q=q.astype(np.float32), y=Y[stream].astype(np.float32),
                    t=stream, n_val=len(va), reg=np.concatenate([reg_va, reg_te]),
                    origins=df.index[stream].astype("int64"), taus=np.asarray(task.taus))
extra = {}
if getattr(model, "has_samples", False):
    t0 = time.time()
    scen = model.sample(df, te, task, a.n_samples, seed=a.seed)
    np.savez_compressed(out / f"seed{a.seed}_samples.npz", scen=scen.astype(np.float32), t=te)
    from dcgridbench.metrics import scores as S
    extra["native_samples"] = {"n_samples": int(scen.shape[1]), "sample_seconds": time.time() - t0,
                               "energy_score_kw": S.energy_score(Y[te], scen)}

if hasattr(model, "info"):
    info = model.info()
else:  # built-in baselines
    info = {"library": "dcgridbench", "library_version": library_version("dcgridbench"),
            "hyperparameters": registry.resolve_params(cfg, a.task), "zero_shot": False,
            "native_samples": False}
info["hyperparameters"] = {**registry.resolve_params(cfg, a.task), **info.get("hyperparameters", {})}

splits = {s: {"period": list(getattr(task, s)), "n_origins": int(len(i))}
          for s, i in (("train", tr), ("val", va), ("test", te))}
metrics = {"val": R.stage12_metrics(Y[va], q_va, task.taus, reg_va),
           "test": R.stage12_metrics(Y[te], q_te, task.taus, reg_te)}
runtime = {"fit_seconds": fit_s, "predict_seconds": pred_s,
           "best_val_loss": getattr(model, "best_val", None)}
rec = R.make_record(task=task, cfg=cfg, model_info=info, seed=a.seed, splits=splits, metrics=metrics,
                    runtime=runtime, predictions_file=str(out / f"seed{a.seed}_stream.npz"), extra=extra)
rec["regime_share_test"] = {R.REGIME_NAMES[int(g)]: float(np.mean(reg_te == g)) for g in np.unique(reg_te)}
R.write(rec, out / f"seed{a.seed}.json")
m = metrics["test"]
print(f"test crps={m['stage1']['crps_kw']:.3f} mae={m['stage1']['mae_kw']:.3f} "
      f"cov90={m['stage2']['cov90']:.3f} width90={m['stage2']['width90_kw']:.2f} "
      f"winkler90={m['stage2']['winkler90']:.2f} byreg={ {k: round(v, 2) for k, v in m['stage2']['cov90_by_regime'].items()} }",
      flush=True)
