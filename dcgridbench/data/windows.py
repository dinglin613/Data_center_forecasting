"""Benchmark tasks: resampling, chronological splits and forecast windows.

A *task* fixes resolution, horizon, forecast-origin stride and split dates so
every method is scored on exactly the same targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LOCAL_TZ = "America/New_York"


@dataclass(frozen=True)
class Task:
    name: str
    resolution: str            # pandas offset, e.g. "5min"
    horizon: int               # steps ahead
    stride: int = 1            # steps between forecast origins
    train: tuple[str, str] = ("2021-03-01", "2021-06-30")
    val: tuple[str, str] = ("2021-07-01", "2021-07-31")
    test: tuple[str, str] = ("2021-08-01", "2021-09-30")
    target: str = "power_w"
    taus: tuple[float, ...] = field(default=(0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975))

    @property
    def dt_min(self) -> float:
        return pd.Timedelta(self.resolution).total_seconds() / 60


TASKS = {
    "mit-5min-1h": Task("mit-5min-1h", "5min", horizon=12, stride=1),
    "mit-15min-4h": Task("mit-15min-4h", "15min", horizon=16, stride=1),
    "mit-1h-24h": Task("mit-1h-24h", "1h", horizon=24, stride=1),
}


def resample(series: pd.DataFrame, resolution: str) -> pd.DataFrame:
    """Mean power over each interval (left-labelled), scheduler features too."""
    return series.resample(resolution, label="left", closed="left").mean()


def split_mask(index: pd.DatetimeIndex, period: tuple[str, str]) -> np.ndarray:
    lo = pd.Timestamp(period[0], tz="UTC")
    hi = pd.Timestamp(period[1], tz="UTC") + pd.Timedelta(days=1)
    return (index >= lo) & (index < hi)


def target_windows(y: np.ndarray, horizon: int) -> np.ndarray:
    """Row t holds y[t+1 : t+1+horizon]; NaN where the window runs off the end."""
    n = len(y)
    out = np.full((n, horizon), np.nan)
    for h in range(1, horizon + 1):
        out[: n - h, h - 1] = y[h:]
    return out


def origins(df: pd.DataFrame, task: Task, split: str) -> np.ndarray:
    """Integer positions of forecast origins whose whole target window lies in ``split``.

    Requiring the window inside the split prevents leakage across boundaries.
    """
    period = getattr(task, split)
    inside = split_mask(df.index, period)
    idx = np.flatnonzero(inside)
    idx = idx[idx + task.horizon < len(df)]
    idx = idx[inside[idx + task.horizon]]
    return idx[:: task.stride]
