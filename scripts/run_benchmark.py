"""Fit one configured model on one task and seed; write the standard result JSON.

    python scripts/run_benchmark.py --model nhits --task mit-1h-24h --seed 0
    python scripts/run_benchmark.py --model nhits --task mit-1h-24h --seed 0 --tune --trial 3 \
        --override '{"learning_rate": 0.0005}'          # tuning trial: val split only
    python scripts/run_benchmark.py --model nhits --task mit-1h-24h --seed 10 \
        --override-file <selected>.json                 # final run with the selected config

Writes  <out>/<task>/<model>/seed<k>.json          result record (docs/result_schema.md)
        <out>/<task>/<model>/seed<k>_stream.npz    val+test quantiles (layout of run_calibration.py)
        <out>/<task>/<model>/seed<k>_samples.npz   native test paths on the scenario subset
Tuning: <out>/<task>/<model>/trial<j>_seed<k>.json (val metrics only, no test prediction).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from dcgridbench.data.windows import TASKS, origins, resample, target_windows
from dcgridbench.leaderboard import results as R
from dcgridbench.leaderboard.stages import scenario_subset
from dcgridbench.models import registry
from dcgridbench.models.external.base import library_version
from dcgridbench.models.regimes import Regimes

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="config name in configs/models or a YAML path")
ap.add_argument("--task", default="mit-1h-24h", choices=list(TASKS))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--series", default="data/processed/mit_cluster_1min.parquet")
ap.add_argument("--out", default="results/benchmark")
ap.add_argument("--override", default=None, help="JSON dict of parameter overrides")
ap.add_argument("--override-file", default=None, help="JSON file with an 'override' dict (selected config)")
ap.add_argument("--tune", action="store_true", help="tuning trial: score the val split only")
ap.add_argument("--trial", type=int, default=None)
ap.add_argument("--n-samples", type=int, default=200, help="native paths per test origin")
a = ap.parse_args()

over, selection = {}, None
if a.override_file:
    selection = json.loads(Path(a.override_file).read_text())
    over.update(selection["override"])
if a.override:
    over.update(json.loads(a.override))

cfg = registry.load_config(a.model)
task = TASKS[a.task]
df = resample(pd.read_parquet(a.series), task.resolution)
df[task.target] = df[task.target] / 1e3  # kW
tr, va, te = (origins(df, task, s) for s in ("train", "val", "test"))
Y = target_windows(df[task.target].to_numpy(), task.horizon)
reg_model = Regimes().fit(df, tr, task)
reg_va, reg_te = reg_model.label(df, va, task), reg_model.label(df, te, task)

np.random.seed(a.seed)
model = registry.build(cfg, a.task, seed=a.seed, overrides=over)
t0 = time.time()
registry.fit_model(model, df, tr, task, va)
fit_s = time.time() - t0
pred_idx = va if a.tune else np.concatenate([va, te])
t0 = time.time()
q = model.predict(df, pred_idx, task)
pred_s = time.time() - t0
assert q.shape == (len(pred_idx), task.horizon, len(task.taus)), q.shape
assert np.isfinite(q).all(), "non-finite quantiles"
q_va, q_te = q[:len(va)], q[len(va):]
print(f"{a.task} {cfg['name']} seed{a.seed} trial{a.trial}: fit {fit_s:.1f}s predict {pred_s:.1f}s", flush=True)

out = Path(a.out) / a.task / cfg["name"]
out.mkdir(parents=True, exist_ok=True)
stem = f"trial{a.trial}_seed{a.seed}" if a.tune else f"seed{a.seed}"
extra = {"overrides": over}
if a.trial is not None:
    extra["trial"] = a.trial
if selection is not None:
    extra["selection"] = {k: v for k, v in selection.items() if k != "override"}
pred_file = None
if not a.tune:
    pred_file = out / f"{stem}_stream.npz"
    np.savez_compressed(pred_file, q=q.astype(np.float32), y=Y[pred_idx].astype(np.float32), t=pred_idx,
                        n_val=len(va), reg=np.concatenate([reg_va, reg_te]),
                        origins=df.index[pred_idx].astype("int64"), taus=np.asarray(task.taus))
    if getattr(model, "has_samples", False):
        sel = scenario_subset(len(te))
        t0 = time.time()
        scen = model.sample(df, te[sel], task, a.n_samples, seed=a.seed)
        np.savez_compressed(out / f"{stem}_samples.npz", scen=scen.astype(np.float32), t=te[sel], sel=sel)
        extra["native_samples"] = {"n_samples": int(scen.shape[1]), "n_origins": int(len(sel)),
                                   "sample_seconds": time.time() - t0}

if hasattr(model, "info"):
    info = model.info()
else:  # built-in baselines
    info = {"library": "dcgridbench", "library_version": library_version("dcgridbench"),
            "hyperparameters": {}, "zero_shot": False, "native_samples": False}
info["hyperparameters"] = {**registry.resolve_params(cfg, a.task, over), **info.get("hyperparameters", {})}

splits = {s: {"period": list(getattr(task, s)), "n_origins": int(len(i))}
          for s, i in (("train", tr), ("val", va), ("test", te))}
metrics = {"val": R.stage12_metrics(Y[va], q_va, task.taus, reg_va)}
if not a.tune:
    metrics["test"] = R.stage12_metrics(Y[te], q_te, task.taus, reg_te)
runtime = {"fit_seconds": fit_s, "predict_seconds": pred_s,
           "best_val_loss": getattr(model, "best_val", None)}
rec = R.make_record(task=task, cfg=cfg, model_info=info, seed=a.seed, splits=splits, metrics=metrics,
                    runtime=runtime, predictions_file=str(pred_file) if pred_file else None, extra=extra,
                    purpose="tuning" if a.tune else "final")
rec["regime_share_test"] = {R.REGIME_NAMES[int(g)]: float(np.mean(reg_te == g)) for g in np.unique(reg_te)}
R.write(rec, out / f"{stem}.json")
for split, m in metrics.items():
    print(f"{split} crps={m['stage1']['crps_kw']:.3f} mae={m['stage1']['mae_kw']:.3f} "
          f"cov90={m['stage2']['cov90']:.3f} width90={m['stage2']['width90_kw']:.2f} "
          f"winkler90={m['stage2']['winkler90']:.2f}", flush=True)
