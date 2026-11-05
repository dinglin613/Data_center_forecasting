"""Stages 2-4 for a saved prediction stream, with the same settings as
scripts/run_calibration.py (calibration) and scripts/run_scenarios.py (scenarios, reserve).

  stage 2  conformal variants of the raw quantiles: static, regime, online, online_regime
  stage 3  scenario generators on a fixed subset of test origins: independent, copula,
           bootstrap, scaled_bootstrap, schaake, and the model's own paths ("native")
  stage 4  newsvendor reserve: per step from quantiles, block from scenarios
"""
from __future__ import annotations

import numpy as np

from dcgridbench.decisions.reserve import block_from_scenarios, per_step_from_quantiles
from dcgridbench.leaderboard.results import stage12_metrics
from dcgridbench.metrics import scores as S
from dcgridbench.models.conformal import calibrate_online, calibrate_static
from dcgridbench.scenarios.generate import fit_copula, sample

COST_RATIOS = (0.05, 0.1, 0.2)
GENERATORS = ("independent", "copula", "bootstrap", "scaled_bootstrap", "schaake")
VARIANTS = ("raw", "static", "regime", "online", "online_regime")
SCENARIO_VARIANTS = ("raw", "online_regime")


def scenario_subset(n_test: int, max_origins: int = 3000, seed: int = 0) -> np.ndarray:
    """Test-origin subset used for scenarios (same rule as run_scenarios.py)."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_test, min(max_origins, n_test), replace=False))


def calibration_variants(st, task, window_days: float = 14) -> dict:
    """Test quantiles for every stage-2 variant from a ``_stream.npz`` dict."""
    nv, t = int(st["n_val"]), st["t"]
    q, y, reg = st["q"].astype(float), st["y"].astype(float), st["reg"]
    q_va, q_te, y_va = q[:nv], q[nv:], y[:nv]
    window = int(window_days * 24 * 60 / task.dt_min)
    step = max(1, int(60 / task.dt_min))  # refresh corrections hourly
    return {
        "raw": q_te,
        "static": calibrate_static(q_va, y_va, q_te, task.taus),
        "regime": calibrate_static(q_va, y_va, q_te, task.taus, reg[:nv], reg[nv:]),
        "online": calibrate_online(t, y, q, task.taus, nv, window, step=step),  # returns test part
        "online_regime": calibrate_online(t, y, q, task.taus, nv, window, reg=reg, step=step),
    }


def _pit_ks(pit):
    return {k: float(np.max(np.abs(np.sort(v) - (np.arange(len(v)) + 0.5) / len(v)))) for k, v in pit.items()}


def _scen_scores(y, sc, med, thr, dt_min):
    return ({"energy_score": S.energy_score(y, sc), "variogram_score": S.variogram_score(y, sc),
             "exceed_brier": S.exceedance_brier(y, sc, thr), "pit_ks": _pit_ks(S.scenario_stat_pit(y, sc, thr, dt_min))},
            {str(c): block_from_scenarios(y, sc, med, c, 1.0) for c in COST_RATIOS})


def score_stream(st, task, native=None, n_scen: int = 200, max_origins: int = 3000) -> dict:
    """All stage 2-4 metrics for one run. ``native`` = (n_sel, S, H) paths on the scenario subset."""
    taus = np.asarray(task.taus)
    nv, t, H = int(st["n_val"]), st["t"], task.horizon
    y_all, q_all = st["y"].astype(float), st["q"].astype(float)
    y_te, reg_te = y_all[nv:], st["reg"][nv:]
    variants = calibration_variants(st, task)
    out = {"calibration": {v: stage12_metrics(y_te, q, taus, reg_te) for v, q in variants.items()}}

    # causal residual history for the bootstrap generators (as in run_scenarios.py)
    md = taus.searchsorted(0.5)
    resid = y_all - q_all[..., md]
    iqr = q_all[..., taus.searchsorted(0.75)] - q_all[..., taus.searchsorted(0.25)]
    resid_scaled = resid / np.maximum(iqr, 1e-3)
    avail = np.searchsorted(t + H, t, side="right")[nv:]
    sel = scenario_subset(len(y_te), max_origins)
    y = y_te[sel]
    thr = float(np.quantile(y_all[:nv], 0.9))  # high-load threshold from validation
    corr = fit_copula(y_all[:nv], q_all[:nv], taus)
    stage3, stage4 = {}, {}
    for var in SCENARIO_VARIANTS:
        q = variants[var][sel]
        med = q[..., md]
        kw = {"bootstrap": dict(resid_hist=resid, resid_avail=avail[sel]),
              "scaled_bootstrap": dict(resid_hist=resid_scaled, resid_avail=avail[sel]),
              "schaake": dict(resid_hist=resid, resid_avail=avail[sel]),
              "copula": dict(corr=corr), "independent": {}}
        stage3[var], stage4[var] = {}, {"per_step": {str(c): per_step_from_quantiles(y, q, taus, c, 1.0)
                                                     for c in COST_RATIOS}, "block": {}}
        for g in GENERATORS:
            sc = sample(q, taus, n_scen, g, **kw[g])
            stage3[var][g], stage4[var]["block"][g] = _scen_scores(y, sc, med, thr, task.dt_min)
        if native is not None and var == "raw":  # the model's own paths, uncalibrated
            stage3[var]["native"], stage4[var]["block"]["native"] = _scen_scores(y, native, med, thr, task.dt_min)
    out["stage3"], out["stage4"] = stage3, stage4
    out["stage34_info"] = {"n_origins": int(len(sel)), "n_scenarios": n_scen, "threshold_kw": thr,
                           "cost_ratios": list(COST_RATIOS), "c_s": 1.0}
    return out
