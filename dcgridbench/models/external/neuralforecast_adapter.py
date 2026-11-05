"""NeuralForecast models (NHITS, DeepAR, TFT, PatchTST, ...) behind the DCGridBench interface.

Training: one contiguous segment holding the train targets followed by the
validation targets. NeuralForecast trains only on windows whose outputs end
before the last ``val_size`` points and validates on windows inside them, so
test data are never seen. The weights with the best validation loss are kept.

Prediction: every origin becomes its own short series with the ``input_size``
rows up to and including the origin, so all origins are forecast in one
batched call and no row after an origin is visible.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from dcgridbench.models.external.base import SCHED_COLS, ExternalModel, context_windows, steps, to_task_quantiles

_LEVEL = re.compile(r"-(lo|hi)-([0-9.]+)$")


def central_levels(taus) -> list[float]:
    """Central interval levels (%) whose bounds are the symmetric ``taus``."""
    taus = np.sort(np.asarray(taus, float))
    lv = sorted({round(100 * (1 - 2 * t), 6) for t in taus if t < 0.5})
    back = sorted([(1 - l / 100) / 2 for l in lv] + [(1 + l / 100) / 2 for l in lv] + [0.5])
    if len(back) != len(taus) or not np.allclose(back, taus):
        raise ValueError("NeuralForecast level losses need taus symmetric around 0.5 incl. 0.5")
    return lv


def make_loss(kind: str, taus, distribution: str | None = None):
    from neuralforecast.losses.pytorch import DistributionLoss, MQLoss
    if kind == "mqloss":
        return MQLoss(level=central_levels(taus))
    if kind == "distribution":
        return DistributionLoss(distribution=distribution or "StudentT", level=central_levels(taus))
    raise ValueError(f"unknown loss {kind}")


def quantile_columns(cols, alias):
    """Output columns of ``alias`` with their quantile levels, sorted by level."""
    out = []
    for c in cols:
        c = str(c)
        if not c.startswith(alias):
            continue
        if c == f"{alias}-median":
            out.append((0.5, c))
        elif (m := _LEVEL.search(c)) and c.startswith(f"{alias}-"):
            lv = float(m.group(2)) / 100
            out.append(((1 - lv) / 2 if m.group(1) == "lo" else (1 + lv) / 2, c))
    out.sort()
    return [t for t, _ in out], [c for _, c in out]


def _best_state_callback():
    import pytorch_lightning as pl

    class BestState(pl.Callback):
        """Keeps a CPU copy of the weights with the lowest validation loss."""
        dcgb_best_state = True

        def __init__(self):
            self.state, self.score, self.step = None, float("inf"), None

        def on_validation_end(self, trainer, module):
            v = trainer.callback_metrics.get("ptl/val_loss")
            if trainer.sanity_checking or v is None or not float(v) < self.score:
                return
            self.score, self.step = float(v), int(trainer.global_step)
            self.state = {k: t.detach().cpu().clone() for k, t in module.state_dict().items()}

    return BestState()


class NeuralForecastModel(ExternalModel):
    library = "neuralforecast"

    def __init__(self, model: str, lookback_hours: float = 168, loss: str = "mqloss",
                 distribution: str | None = None, exogenous: bool = True, seed: int = 0,
                 accelerator: str = "auto", restore_best: bool = True, name: str | None = None,
                 predict_batch: int = 4096, **model_kw):
        super().__init__(model=model, lookback_hours=lookback_hours, loss=loss,
                         distribution=distribution, exogenous=exogenous, restore_best=restore_best,
                         **model_kw)
        self.model_name, self.lookback_hours = model, lookback_hours
        self.loss_kind, self.distribution = loss, distribution
        self.exogenous, self.seed, self.accelerator = exogenous, seed, accelerator
        self.restore_best, self.model_kw, self.predict_batch = restore_best, model_kw, predict_batch
        self.name = name or model.lower()

    def _cls(self):
        import neuralforecast.models as M
        return getattr(M, self.model_name)

    def _exog_cols(self, df):
        if not self.exogenous or not getattr(self._cls(), "EXOGENOUS_HIST", False):
            return []
        return [c for c in SCHED_COLS if c in df]

    @staticmethod
    def segment(df, train_idx, val_idx, task):
        """First train origin, end (exclusive) of the fit segment, validation size.

        The train part ends at the last train target, the val part at the last val target.
        """
        first = int(train_idx.min())
        train_stop = int(train_idx.max()) + task.horizon + 1
        if val_idx is None or len(val_idx) == 0:
            return first, train_stop, 0
        if int(val_idx.min()) < train_stop - 1:
            raise ValueError("validation origins overlap the training targets")
        stop = int(val_idx.max()) + task.horizon + 1
        return first, stop, stop - train_stop

    def _frame(self, df, lo, hi, task):
        part = df.iloc[lo:hi]
        y = part[task.target]
        out = pd.DataFrame({"unique_id": "dc", "ds": part.index.tz_convert(None), "y": y.to_numpy(float)})
        for c in self.exog:
            out[c] = part[c].to_numpy(float)
        out = out.ffill().bfill()
        out["available_mask"] = y.notna().to_numpy(float)
        return out

    def fit(self, df, idx, task, val_idx=None):
        from neuralforecast import NeuralForecast

        self.L = steps(self.lookback_hours, task)
        self.exog = self._exog_cols(df)
        first, stop, val_size = self.segment(df, idx, val_idx, task)
        start = max(0, first + 1 - self.L)  # input window of the first train origin
        self.fit_range_ = (int(start), int(stop), int(val_size))

        kw = dict(self.model_kw)
        kw.setdefault("devices", 1)  # one device; NF would otherwise take all GPUs
        if self.exog:
            kw["hist_exog_list"] = self.exog
        callbacks = [_best_state_callback()] if (val_size and self.restore_best) else []
        model = self._cls()(h=task.horizon, input_size=self.L,
                            loss=make_loss(self.loss_kind, task.taus, self.distribution),
                            random_seed=self.seed, accelerator=self.accelerator,
                            enable_progress_bar=False, logger=False, callbacks=callbacks, **kw)
        self.nf = NeuralForecast(models=[model], freq=task.resolution)
        t0 = time.time()
        self.nf.fit(df=self._frame(df, start, stop, task), val_size=val_size)
        self.fit_seconds = time.time() - t0
        fitted = self.nf.models[0]
        self.best_val = self.best_step = None
        for cb in fitted.trainer_kwargs.get("callbacks") or []:
            if getattr(cb, "dcgb_best_state", False) and cb.state is not None:
                fitted.load_state_dict(cb.state)
                self.best_val, self.best_step, cb.state = cb.score, cb.step, None
        self.alias = str(fitted)
        return self

    def _long_frame(self, df, idx, task):
        L = self.L
        pos = idx[:, None] + np.arange(-L + 1, 1)[None]
        ds = df.index.tz_convert(None).to_numpy()[np.clip(pos, 0, None)]
        y = context_windows(df[task.target].to_numpy(), idx, L)
        out = {"unique_id": np.repeat(np.asarray(idx), L), "ds": ds.reshape(-1), "y": y.reshape(-1)}
        for c in self.exog:
            out[c] = context_windows(df[c].to_numpy(), idx, L).reshape(-1)
        frame = pd.DataFrame(out)
        frame["available_mask"] = np.isfinite(y).reshape(-1).astype(float)
        num = ["y"] + self.exog
        if frame[num].isna().to_numpy().any():  # origins near the series start or gaps
            frame[num] = frame.groupby("unique_id")[num].transform(lambda s: s.ffill().bfill())
        if (pos < 0).any():  # timestamps before the start must still be regular
            step = pd.Timedelta(task.resolution)
            ds0 = df.index.tz_convert(None)[0]
            frame["ds"] = np.where(pos.reshape(-1) < 0, ds0 + pos.reshape(-1) * step, frame["ds"])
        return frame

    def predict(self, df, idx, task):
        idx = np.asarray(idx)
        qs = []
        for s in range(0, len(idx), self.predict_batch):
            part = idx[s:s + self.predict_batch]
            fc = self.nf.predict(df=self._long_frame(df, part, task)).sort_values(["unique_id", "ds"])
            levels, cols = quantile_columns(fc.columns, self.alias)
            arr = fc[cols].to_numpy(float).reshape(len(part), task.horizon, len(cols))
            ids = fc["unique_id"].to_numpy().reshape(len(part), task.horizon)[:, 0]
            if not np.array_equal(ids, np.sort(part)):
                raise RuntimeError("forecast ids do not match the requested origins")
            arr = arr[np.searchsorted(ids, part)]  # back to the order of ``part``
            qs.append(to_task_quantiles(arr, levels, task.taus, tail="clamp"))
        return np.concatenate(qs)

    def info(self):
        d = super().info()
        d["exogenous"] = list(getattr(self, "exog", []))
        d["best_val_loss"] = getattr(self, "best_val", None)
        d["best_step"] = getattr(self, "best_step", None)
        return d
