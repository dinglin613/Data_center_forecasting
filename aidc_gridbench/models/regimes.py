"""Workload regimes known at the forecast origin.

  0 low       few GPUs allocated (maintenance, drain, ramp-up)
  1 calm      normal load, low recent volatility
  2 active    normal load, medium volatility
  3 volatile  normal load, high volatility or heavy queue pressure

Volatility thresholds are fitted on the training split only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOTAL_GPUS = 448


class Regimes:
    def __init__(self, low_frac=0.25, queue_frac=1.0):
        self.low_frac, self.queue_frac = low_frac, queue_frac

    def _vol(self, df, target, dt_min):
        w = max(2, int(round(60 / dt_min)))
        return df[target].diff().abs().rolling(w, min_periods=1).mean().to_numpy()

    def fit(self, df, idx, task):
        v = self._vol(df, task.target, task.dt_min)[idx]
        self.cuts = np.nanquantile(v, [1 / 3, 2 / 3])
        return self

    def label(self, df, idx, task):
        v = self._vol(df, task.target, task.dt_min)[idx]
        alloc = df["gpus_alloc"].to_numpy()[idx]
        queue = df["gpus_queued"].to_numpy()[idx]
        r = 1 + np.searchsorted(self.cuts, v)          # 1, 2, 3
        r = np.where(queue > self.queue_frac * TOTAL_GPUS, 3, r)
        return np.where(alloc < self.low_frac * TOTAL_GPUS, 0, r)
