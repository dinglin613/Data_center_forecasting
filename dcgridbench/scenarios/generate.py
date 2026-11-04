"""Turn quantile forecasts into joint trajectories over the horizon.

Per-step quantiles say nothing about how errors move together across the
horizon, which drives ramps, peak duration and energy. Three generators:

  independent   each step sampled on its own (shows what goes wrong)
  copula        Gaussian copula with a horizon x horizon correlation estimated
                from the normal scores of past PIT values
  bootstrap     whole past error trajectories (y - median) added to the median,
                drawn from origins whose targets are already observed
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _quantile_fn(q_row, taus, u):
    """Inverse CDF by linear interpolation between quantiles, linear tails."""
    t = np.concatenate([[0.0], taus, [1.0]])
    lo = q_row[0] - (q_row[1] - q_row[0]) / (taus[1] - taus[0]) * taus[0]
    hi = q_row[-1] + (q_row[-1] - q_row[-2]) / (taus[-1] - taus[-2]) * (1 - taus[-1])
    v = np.concatenate([[lo], q_row, [hi]])
    return np.interp(u, t, v)


def pit(y, q, taus):
    """F(y) under the interpolated predictive distribution, shape (n, H)."""
    taus = np.asarray(taus)
    n, H, _ = q.shape
    t = np.concatenate([[0.0], taus, [1.0]])
    out = np.empty((n, H))
    for i in range(n):
        for h in range(H):
            row = q[i, h]
            lo = row[0] - (row[1] - row[0]) / (taus[1] - taus[0]) * taus[0]
            hi = row[-1] + (row[-1] - row[-2]) / (taus[-1] - taus[-2]) * (1 - taus[-1])
            out[i, h] = np.interp(y[i, h], np.concatenate([[lo], row, [hi]]), t)
    return out


def fit_copula(y_cal, q_cal, taus, shrink=0.05):
    """Correlation of normal scores of calibration PITs (lightly shrunk to I)."""
    u = np.clip(pit(y_cal, q_cal, taus), 1e-4, 1 - 1e-4)
    z = norm.ppf(u)
    z = z[np.isfinite(z).all(1)]
    C = np.corrcoef(z, rowvar=False)
    return (1 - shrink) * C + shrink * np.eye(len(C))


def sample(q, taus, n_scen, method="copula", corr=None, seed=0,
           resid_hist=None, resid_avail=None):
    """Return scenarios (n, S, H).

    For ``bootstrap`` pass ``resid_hist`` (m, H) past y - median windows and
    ``resid_avail`` (n,) = how many of them are observed at each origin.
    """
    rng = np.random.default_rng(seed)
    taus = np.asarray(taus)
    n, H, K = q.shape
    out = np.empty((n, n_scen, H))
    med = q[..., np.searchsorted(taus, 0.5)]
    if method == "bootstrap":
        for i in range(n):
            m = int(resid_avail[i])
            pick = rng.integers(max(0, m - 4000), m, n_scen)  # recent past only
            out[i] = med[i][None] + resid_hist[pick]
        return out
    if method == "scaled_bootstrap":
        # resid_hist holds past errors divided by that origin's predicted IQR
        iqr = q[..., np.searchsorted(taus, 0.75)] - q[..., np.searchsorted(taus, 0.25)]
        for i in range(n):
            m = int(resid_avail[i])
            pick = rng.integers(max(0, m - 4000), m, n_scen)
            out[i] = med[i][None] + iqr[i][None] * resid_hist[pick]
        return out
    if method == "schaake":
        # conditional marginals from q, rank structure from past error trajectories
        for i in range(n):
            m = int(resid_avail[i])
            pick = rng.integers(max(0, m - 4000), m, n_scen)
            ranks = np.argsort(np.argsort(resid_hist[pick], axis=0), axis=0)  # (S, H)
            u = (np.arange(n_scen) + 0.5) / n_scen
            for h in range(H):
                out[i, :, h] = _quantile_fn(q[i, h], taus, u)[ranks[:, h]]
        return out
    if method == "independent":
        U = rng.uniform(size=(n, n_scen, H))
    elif method == "copula":
        Lc = np.linalg.cholesky(corr)
        U = norm.cdf(rng.standard_normal((n, n_scen, H)) @ Lc.T)
    else:
        raise ValueError(method)
    for i in range(n):
        for h in range(H):
            out[i, :, h] = _quantile_fn(q[i, h], taus, U[i, :, h])
    return out
