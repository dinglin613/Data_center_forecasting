"""Common base for wrappers around external forecasting libraries.

A wrapper turns a library model into the DCGridBench interface:
    fit(df, train_idx, task, val_idx=None)
    predict(df, idx, task)          -> (n, H, K) quantiles at task.taus
    sample(df, idx, task, n, seed)  -> (n, S, H) paths, only if has_samples

Leakage rules every wrapper follows:
  * training targets come only from train origins, early stopping only from val;
  * the input for origin t uses rows <= t only (see ``context_windows``).
"""
from __future__ import annotations

import importlib.metadata as md

import numpy as np
import pandas as pd
from scipy.stats import norm

SCHED_COLS = ["gpus_alloc", "jobs_running", "gpus_queued", "jobs_queued"]
CAL_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
LOCAL_TZ = "America/New_York"


def calendar(index) -> np.ndarray:
    """(T, 4) sin/cos of local hour of day and day of week; a function of time stamps only."""
    local = pd.DatetimeIndex(index).tz_convert(LOCAL_TZ)
    hour = (local.hour + local.minute / 60).to_numpy() * (2 * np.pi / 24)
    dow = (local.dayofweek + local.hour / 24).to_numpy() * (2 * np.pi / 7)
    return np.stack([np.sin(hour), np.cos(hour), np.sin(dow), np.cos(dow)], axis=1)


class ExternalModel:
    name = "external"
    library = ""            # pip distribution name, used for the version lookup
    has_samples = False
    zero_shot = False

    def __init__(self, **params):
        self.params = params

    def fit(self, df, idx, task, val_idx=None):
        raise NotImplementedError

    def predict(self, df, idx, task):
        raise NotImplementedError

    def sample(self, df, idx, task, n_samples=100, seed=0):
        raise NotImplementedError(f"{self.name} has no native sample paths")

    def info(self) -> dict:
        return {"library": self.library, "library_version": library_version(self.library),
                "hyperparameters": dict(self.params), "zero_shot": self.zero_shot,
                "native_samples": self.has_samples}


def library_version(dist: str) -> str | None:
    if not dist:
        return None
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def steps(hours: float, task) -> int:
    return max(1, int(round(hours * 60 / task.dt_min)))


def context_windows(x: np.ndarray, idx: np.ndarray, length: int) -> np.ndarray:
    """Rows idx-length+1 .. idx of ``x`` for each origin; NaN before the series start.

    ``x`` is (T,) or (T, C); the result is (n, length) or (n, length, C).
    Nothing after the origin is ever read.
    """
    x = np.asarray(x, dtype=np.float64)
    pos = idx[:, None] + np.arange(-length + 1, 1)[None]
    out = x[np.clip(pos, 0, None)]
    out[pos < 0] = np.nan
    return out


def to_task_quantiles(q: np.ndarray, levels, taus, tail: str = "gaussian") -> np.ndarray:
    """Map quantiles at ``levels`` (last axis) to ``taus``.

    Inside the native range: linear interpolation in tau. Outside it:
    ``gaussian`` scales the outermost half-spread by the normal quantile ratio,
    ``clamp`` repeats the outermost quantile. Output is sorted (no crossing).
    """
    levels, taus = np.asarray(levels, float), np.asarray(taus, float)
    order = np.argsort(levels)
    levels, q = levels[order], np.sort(q[..., order], axis=-1)
    if len(levels) == len(taus) and np.allclose(levels, taus):
        return q
    flat = q.reshape(-1, len(levels))
    out = np.stack([np.interp(taus, levels, row) for row in flat]).reshape(q.shape[:-1] + (len(taus),))
    lo, hi = levels[0], levels[-1]
    if tail == "gaussian" and (taus.min() < lo or taus.max() > hi):
        med = np.stack([np.interp(0.5, levels, row) for row in flat]).reshape(q.shape[:-1])
        for k, t in enumerate(taus):
            if t > hi and hi > 0.5:
                out[..., k] = med + (q[..., -1] - med) * norm.ppf(t) / norm.ppf(hi)
            elif t < lo and lo < 0.5:
                out[..., k] = med + (q[..., 0] - med) * norm.ppf(t) / norm.ppf(lo)
    elif tail != "clamp" and tail != "gaussian":
        raise ValueError(f"unknown tail rule {tail}")
    return np.sort(out, axis=-1)
