"""Quantile-forecast baselines. Every model maps origins -> (n, H, K) quantiles.

Interface:  model.fit(df, train_idx, task) ; model.predict(df, idx, task) -> q
All features use information available at the origin only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dcgridbench.data.windows import LOCAL_TZ, Task, target_windows


class EmpiricalResidualModel:
    """Point forecast + empirical quantiles of its training residuals per horizon."""

    def point(self, y: np.ndarray, idx: np.ndarray, task: Task) -> np.ndarray:
        raise NotImplementedError

    def fit(self, df, idx, task):
        y = df[task.target].to_numpy()
        resid = target_windows(y, task.horizon)[idx] - self.point(y, idx, task)
        self.qres = np.nanquantile(resid, task.taus, axis=0).T  # (H, K)
        return self

    def predict(self, df, idx, task):
        y = df[task.target].to_numpy()
        return self.point(y, idx, task)[..., None] + self.qres[None]


class Persistence(EmpiricalResidualModel):
    name = "persistence"

    def point(self, y, idx, task):
        return np.repeat(y[idx, None], task.horizon, axis=1)


class SeasonalNaive(EmpiricalResidualModel):
    """Value one day earlier at the same time of day."""
    name = "seasonal_naive"

    def point(self, y, idx, task):
        period = int(round(24 * 60 / task.dt_min))
        h = np.arange(1, task.horizon + 1)
        src = idx[:, None] + h[None] - period
        return y[np.clip(src, 0, None)]


# ---------------------------------------------------------------- LightGBM

SCHED_COLS = ["gpus_alloc", "jobs_running", "gpus_queued", "jobs_queued"]


CAPACITY_KW_GUESS = 448 * 0.25  # 448 V100 x 250 W, used only to scale features


def make_features(df: pd.DataFrame, task: Task, relative: bool = False) -> pd.DataFrame:
    """Features at each origin: recent power, daily/weekly lags, scheduler state, calendar.

    ``relative=True`` expresses power features as changes from the current value
    and scheduler counts as fractions of the cluster, so trees do not key on
    absolute load levels that shift between regimes.
    """
    if relative:
        return _relative_features(df, task)
    y = df[task.target]
    steps_h = int(round(60 / task.dt_min)) or 1
    day, week = 24 * steps_h, 7 * 24 * steps_h
    f = pd.DataFrame(index=df.index)
    for k in range(min(12, max(3, steps_h * 2))):
        f[f"lag{k}"] = y.shift(k)
    for w in sorted({max(2, steps_h), 6 * steps_h, day}):
        f[f"mean_{w}"] = y.rolling(w, min_periods=1).mean()
        f[f"std_{w}"] = y.rolling(w, min_periods=2).std()
    f["max_day"] = y.rolling(day, min_periods=1).max()
    f["lag_day"] = y.shift(day)
    f["lag_week"] = y.shift(week)
    for c in SCHED_COLS:
        if c in df:
            f[c] = df[c]
            f[f"d_{c}"] = df[c].diff(steps_h)
    local = df.index.tz_convert(LOCAL_TZ)
    f["hour"] = local.hour + local.minute / 60
    f["dow"] = local.dayofweek
    return f


def _relative_features(df, task):
    y = df[task.target]
    steps_h = int(round(60 / task.dt_min)) or 1
    day = 24 * steps_h
    f = pd.DataFrame(index=df.index)
    f["level"] = y / CAPACITY_KW_GUESS
    for k in range(1, min(12, max(3, steps_h * 2))):
        f[f"d{k}"] = y - y.shift(k)
    for w in sorted({max(2, steps_h), 6 * steps_h, day}):
        f[f"rmean_{w}"] = y - y.rolling(w, min_periods=1).mean()
        f[f"std_{w}"] = y.rolling(w, min_periods=2).std()
        f[f"absdiff_{w}"] = y.diff().abs().rolling(w, min_periods=1).mean()
    f["rmax_day"] = y.rolling(day, min_periods=1).max() - y
    f["rmin_day"] = y - y.rolling(day, min_periods=1).min()
    if "gpus_alloc" in df:
        alloc = df["gpus_alloc"].clip(lower=1)
        f["alloc_frac"] = df["gpus_alloc"] / 448
        f["queue_frac"] = df["gpus_queued"] / 448
        f["queue_per_alloc"] = df["gpus_queued"] / alloc
        f["kw_per_gpu"] = y / alloc
        f["d_alloc"] = df["gpus_alloc"].diff(steps_h) / 448
        f["d_queue"] = df["gpus_queued"].diff(steps_h) / 448
        f["jobs_q_frac"] = df["jobs_queued"] / df["jobs_running"].clip(lower=1)
    local = df.index.tz_convert(LOCAL_TZ)
    f["hour"] = local.hour + local.minute / 60
    f["dow"] = local.dayofweek
    return f


class QuantileLGBM:
    """Pooled direct multi-horizon model: horizon is a feature, one booster per quantile.

    Target is the change from the origin value, which is easier to learn and
    keeps the model sensible across load levels.
    """
    name = "qlgbm"

    def __init__(self, n_estimators=600, learning_rate=0.05, num_leaves=63,
                 min_child_samples=50, max_rows=400_000, seed=0, n_jobs=8, relative=False):
        self.relative = relative
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, min_child_samples=min_child_samples,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=seed, verbose=-1, n_jobs=n_jobs)
        self.max_rows, self.seed = max_rows, seed

    def _stack(self, df, idx, task):
        F = make_features(df, task, self.relative).to_numpy()[idx]
        y = df[task.target].to_numpy()
        H = task.horizon
        X = np.concatenate([np.repeat(F, H, axis=0),
                            np.tile(np.arange(1, H + 1), len(idx))[:, None]], axis=1)
        base = np.repeat(y[idx], H)
        tgt = (target_windows(y, H)[idx]).reshape(-1) - base
        return X, tgt, base

    def fit(self, df, idx, task, val_idx=None):
        import lightgbm as lgb
        X, t, _ = self._stack(df, idx, task)
        ok = np.isfinite(t)
        X, t = X[ok], t[ok]
        if len(t) > self.max_rows:
            pick = np.random.default_rng(self.seed).choice(len(t), self.max_rows, replace=False)
            X, t = X[pick], t[pick]
        self.models = [lgb.LGBMRegressor(objective="quantile", alpha=a, **self.params).fit(X, t)
                       for a in task.taus]
        return self

    def predict(self, df, idx, task):
        X, _, base = self._stack(df, idx, task)
        q = np.stack([m.predict(X) for m in self.models], axis=-1) + base[:, None]
        q = np.sort(q, axis=-1)  # remove quantile crossing
        return q.reshape(len(idx), task.horizon, len(task.taus))


class QuantileLGBMRel(QuantileLGBM):
    name = "qlgbm_rel"

    def __init__(self, **kw):
        super().__init__(relative=True, **kw)


MODELS = {m.name: m for m in (Persistence, SeasonalNaive, QuantileLGBM, QuantileLGBMRel)}
