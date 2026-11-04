"""Probabilistic and grid-relevant scores.

Shapes:
  y          (n, H)        realised power over the horizon
  q          (n, H, K)     quantile forecasts at levels ``taus`` (K,)
  scen       (n, S, H)     S sample trajectories per forecast origin
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- marginal

def pinball(y, q, taus) -> float:
    """Mean pinball loss over origins, horizons and quantile levels."""
    taus = np.asarray(taus)
    d = y[..., None] - q
    return float(np.mean(np.maximum(taus * d, (taus - 1) * d)))


def crps_from_quantiles(y, q, taus) -> float:
    """CRPS approximated as 2 x mean pinball over evenly spread quantiles."""
    return 2.0 * pinball(y, q, taus)


def coverage(y, lo, hi) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_width(lo, hi) -> float:
    return float(np.mean(hi - lo))


def winkler(y, lo, hi, alpha) -> float:
    """Interval score for a central (1-alpha) interval; lower is better."""
    w = hi - lo
    pen = (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return float(np.mean(w + pen))


# ---------------------------------------------------------------- joint (scenarios)

def energy_score(y, scen) -> float:
    """Energy score of trajectories; sensitive to temporal dependence."""
    term1 = np.linalg.norm(scen - y[:, None, :], axis=-1).mean(axis=1)
    term2 = np.empty(len(y))
    for s in range(0, len(y), 64):  # chunked: the pairwise tensor is (n, S, S, H)
        c = scen[s:s + 64]
        term2[s:s + 64] = np.linalg.norm(c[:, :, None, :] - c[:, None, :, :], axis=-1).mean(axis=(1, 2))
    return float(np.mean(term1 - 0.5 * term2))


def variogram_score(y, scen, p: float = 0.5, max_lag: int | None = None) -> float:
    """Variogram score of order p (Scheuerer & Hamill, 2015) over horizon pairs."""
    H = y.shape[1]
    i, j = np.triu_indices(H, k=1)
    if max_lag is not None:
        keep = (j - i) <= max_lag
        i, j = i[keep], j[keep]
    obs = np.abs(y[:, i] - y[:, j]) ** p
    ens = (np.abs(scen[:, :, i] - scen[:, :, j]) ** p).mean(axis=1)
    return float(np.mean(np.sum((obs - ens) ** 2, axis=1)))


# ---------------------------------------------------------------- grid statistics

def grid_stats(traj, threshold: float, dt_min: float) -> dict[str, np.ndarray]:
    """Per-trajectory peak, max up/down ramp, longest run above ``threshold``.

    ``traj`` (..., H). Ramps are per step; durations in minutes.
    """
    traj = np.asarray(traj)
    d = np.diff(traj, axis=-1)
    above = traj > threshold
    return {
        "peak": traj.max(axis=-1),
        "ramp_up": d.max(axis=-1),
        "ramp_down": -d.min(axis=-1),
        "energy": traj.sum(axis=-1) * dt_min / 60.0,
        "dur_above": _longest_run(above) * dt_min,
        "exceed": above.any(axis=-1).astype(float),
    }


def _longest_run(mask) -> np.ndarray:
    m = mask.reshape(-1, mask.shape[-1])
    best = np.zeros(len(m), dtype=int)
    cur = np.zeros(len(m), dtype=int)
    for k in range(m.shape[1]):
        cur = np.where(m[:, k], cur + 1, 0)
        best = np.maximum(best, cur)
    return best.reshape(mask.shape[:-1])


def scenario_stat_pit(y, scen, threshold: float, dt_min: float) -> dict[str, np.ndarray]:
    """PIT of each observed grid statistic within its scenario distribution.

    A calibrated scenario generator gives PIT ~ Uniform(0, 1) for every statistic;
    check with a histogram or a KS test.
    """
    obs = grid_stats(y, threshold, dt_min)
    ens = grid_stats(scen, threshold, dt_min)
    rng = np.random.default_rng(0)
    out = {}
    for k in obs:
        below = (ens[k] < obs[k][:, None]).mean(axis=1)
        equal = (ens[k] == obs[k][:, None]).mean(axis=1)
        out[k] = below + rng.uniform(size=len(below)) * equal  # randomised for ties
    return out


def exceedance_brier(y, scen, threshold: float) -> float:
    """Brier score of P(any step over threshold) within the horizon."""
    p = (scen > threshold).any(axis=-1).mean(axis=1)
    o = (y > threshold).any(axis=-1).astype(float)
    return float(np.mean((p - o) ** 2))
