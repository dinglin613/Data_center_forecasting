"""Scenario quality and reserve-decision value on the test split.

Needs results from run_calibration.py (``*_stream.npz`` and calibrated test quantiles).
    python scripts/run_scenarios.py --task mit-5min-1h --model qlgbm_rel --variant online_regime
"""
import argparse
import json
from pathlib import Path

import numpy as np

from aidc_gridbench.data.windows import TASKS
from aidc_gridbench.decisions.reserve import block_from_scenarios, per_step_from_quantiles
from aidc_gridbench.metrics import scores as S
from aidc_gridbench.scenarios.generate import fit_copula, sample

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="mit-5min-1h")
ap.add_argument("--model", default="qlgbm_rel")
ap.add_argument("--variants", nargs="*", default=["raw", "online_regime"])
ap.add_argument("--calib-dir", default="results/calibration")
ap.add_argument("--n-scen", type=int, default=200)
ap.add_argument("--max-origins", type=int, default=3000, help="subsample test origins for speed")
ap.add_argument("--cost-ratios", nargs="*", type=float, default=[0.05, 0.1, 0.2])
ap.add_argument("--out", default="results/scenarios")
a = ap.parse_args()

task = TASKS[a.task]
taus = np.asarray(task.taus)
d = Path(a.calib_dir)
st = np.load(d / f"{a.task}_{a.model}_stream.npz")
nv = int(st["n_val"])
q_val, y_val = st["q"][:nv].astype(float), st["y"][:nv].astype(float)
y_te = st["y"][nv:].astype(float)
t = st["t"]; H = task.horizon

# causal residual history for bootstrap: base-model median errors
med_stream = st["q"][..., taus.searchsorted(0.5)].astype(float)
resid = st["y"] - med_stream
iqr_stream = (st["q"][..., taus.searchsorted(0.75)] - st["q"][..., taus.searchsorted(0.25)]).astype(float)
resid_scaled = resid / np.maximum(iqr_stream, 1e-3)
avail = np.searchsorted(t + H, t, side="right")[nv:]

rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(y_te), min(a.max_origins, len(y_te)), replace=False))
y = y_te[sel]
thr = float(np.quantile(y_val, 0.9))  # "high load" threshold fixed from validation

res = {"task": a.task, "model": a.model, "threshold_kw": thr, "n_origins": int(len(sel))}
for var in a.variants:
    q = np.load(d / f"{a.task}_{a.model}_{var}.npz")["q"].astype(float)[sel]
    med = q[..., taus.searchsorted(0.5)]
    corr = fit_copula(y_val, q_val, taus)
    scen = {
        "independent": sample(q, taus, a.n_scen, "independent"),
        "copula": sample(q, taus, a.n_scen, "copula", corr=corr),
        "bootstrap": sample(q, taus, a.n_scen, "bootstrap", resid_hist=resid, resid_avail=avail[sel]),
        "scaled_bootstrap": sample(q, taus, a.n_scen, "scaled_bootstrap", resid_hist=resid_scaled,
                                   resid_avail=avail[sel]),
        "schaake": sample(q, taus, a.n_scen, "schaake", resid_hist=resid, resid_avail=avail[sel]),
    }
    res[var] = {"per_step_reserve": {str(r): per_step_from_quantiles(y, q, taus, r, 1.0) for r in a.cost_ratios}}
    for g, sc in scen.items():
        pit = S.scenario_stat_pit(y, sc, thr, task.dt_min)
        r = {"energy_score": S.energy_score(y, sc),
             "variogram_score": S.variogram_score(y, sc),
             "exceed_brier": S.exceedance_brier(y, sc, thr),
             # |mean PIT - 0.5| and KS distance from uniform for each grid statistic
             "pit_ks": {k: float(np.max(np.abs(np.sort(v) - (np.arange(len(v)) + 0.5) / len(v))))
                        for k, v in pit.items()},
             "block_reserve": {str(cr): block_from_scenarios(y, sc, med, cr, 1.0) for cr in a.cost_ratios}}
        res[var][g] = r
        print(f"{var:14s} {g:12s} ES={r['energy_score']:.3f} VS={r['variogram_score']:.3f} "
              f"brier={r['exceed_brier']:.4f} ks={ {k: round(v, 3) for k, v in r['pit_ks'].items()} } "
              f"block@0.1={ {k: round(v, 3) for k, v in r['block_reserve']['0.1'].items()} }", flush=True)
    print(var, "per-step reserve @0.1", res[var]["per_step_reserve"]["0.1"], flush=True)

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / f"{a.task}_{a.model}.json", "w"), indent=2)
