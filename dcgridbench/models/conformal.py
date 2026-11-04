"""Model-agnostic calibration of quantile forecasts.

For every quantile level tau and horizon h we correct the base forecast by the
tau-quantile of past scores s = y - q_tau (one-sided conformal correction):

    q'_tau = q_tau + Quantile_tau({s_j : j in calibration set})

The calibration set differs by method:
  static          all validation origins (classical split conformal)
  regime          validation origins in the same regime (Mondrian conformal)
  online          the last ``window`` origins whose target is already observed
  online_regime   as online, restricted to the same regime

"Already observed" means origin_j + h <= origin_i, so no future information is used.
"""
from __future__ import annotations

import numpy as np


def scores(y: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(n, H, K) residuals y - q for each quantile level."""
    return y[..., None] - q


def _corr(s: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Per-level correction from a score matrix s (m, K)."""
    s = s[np.isfinite(s).all(axis=1)]
    if len(s) == 0:
        return np.zeros(len(taus))
    return np.array([np.quantile(s[:, k], taus[k]) for k in range(len(taus))])


def calibrate_static(q_cal, y_cal, q, taus, reg_cal=None, reg=None, min_n=200):
    """Split conformal (optionally per regime) using a fixed calibration set."""
    taus = np.asarray(taus)
    s = scores(y_cal, q_cal)
    out = q.copy()
    H = q.shape[1]
    for h in range(H):
        glob = _corr(s[:, h], taus)
        if reg is None:
            out[:, h] += glob
            continue
        for r in np.unique(reg):
            m_cal = reg_cal == r
            c = _corr(s[m_cal, h], taus) if m_cal.sum() >= min_n else glob
            out[reg == r, h] += c
    return np.sort(out, axis=-1)


def calibrate_online(t, y, q, taus, start, window=2000, reg=None, min_n=200, step=1):
    """Rolling conformal over a time-ordered stream of origins.

    t      (n,) origin positions on the series grid (increasing)
    y, q   realised windows and base quantiles for every origin
    start  first origin index to calibrate (earlier ones only feed the history)
    step   recompute corrections every ``step`` origins (speed); between
           recomputations the last correction is reused, which is still causal.
    """
    taus = np.asarray(taus)
    s = scores(y, q)
    n, H, K = q.shape
    out = q.copy()
    for h in range(H):
        avail = np.searchsorted(t + h + 1, t, side="right")  # #origins observed at each t_i
        c_glob = np.zeros(K)
        c_reg: dict = {}
        for i in range(start, n):
            if (i - start) % step == 0:
                hi = avail[i]
                lo = max(0, hi - window)
                c_glob = _corr(s[lo:hi, h], taus)
                if reg is not None:
                    c_reg = {}
                    past_r = reg[lo:hi]
                    for r in np.unique(past_r):
                        m = past_r == r
                        if m.sum() >= min_n:
                            c_reg[r] = _corr(s[lo:hi, h][m], taus)
            c = c_reg.get(reg[i], c_glob) if reg is not None else c_glob
            out[i, h] += c
    return np.sort(out[start:], axis=-1)
