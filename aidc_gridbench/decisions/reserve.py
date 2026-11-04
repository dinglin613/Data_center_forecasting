"""Reserve sizing from probabilistic forecasts (newsvendor).

The scheduled power is the median forecast P_hat. Reserve R covers positive
deviations. Cost per decision:  c_R * R + c_S * max(D - R, 0).
The optimal R is the (1 - c_R / c_S) quantile of the predicted deviation D.

  per_step   one reserve per time step, D = P_{t+h} - P_hat_{t+h}
             (needs only per-step quantiles)
  block      one reserve held for the whole horizon,
             D = max_h (P_{t+h} - P_hat_{t+h})  (needs joint scenarios)
"""
from __future__ import annotations

import numpy as np


def realised_cost(R, D, c_r, c_s):
    short = np.maximum(D - R, 0.0)
    return {"cost": float(np.mean(c_r * R + c_s * short)),
            "reserve_kw": float(np.mean(R)),
            "shortfall_prob": float(np.mean(short > 1e-9)),
            "shortfall_kw": float(np.mean(short))}


def per_step_from_quantiles(y, q, taus, c_r, c_s):
    taus = np.asarray(taus)
    level = 1 - c_r / c_s
    med = q[..., np.searchsorted(taus, 0.5)]
    dev_q = q - med[..., None]
    R = np.maximum(np.array([[np.interp(level, taus, dev_q[i, h]) for h in range(q.shape[1])]
                             for i in range(q.shape[0])]), 0.0)
    return realised_cost(R, y - med, c_r, c_s)


def block_from_scenarios(y, scen, med, c_r, c_s):
    level = 1 - c_r / c_s
    D_scen = (scen - med[:, None, :]).max(axis=-1)          # (n, S)
    R = np.maximum(np.quantile(D_scen, level, axis=1), 0.0)
    return realised_cost(R, (y - med).max(axis=-1), c_r, c_s)
