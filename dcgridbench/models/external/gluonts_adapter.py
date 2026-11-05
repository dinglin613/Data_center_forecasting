"""GluonTS (PyTorch) DeepAR with ancestral sampling, behind the DCGridBench interface.

NeuralForecast's DeepAR feeds the predicted mean back during multi-step
prediction, which makes its intervals far too narrow; GluonTS samples whole
paths, as in the DeepAR paper, so it also provides native (n, S, H) samples.

Leakage control:
  * all series start at the same row ``s0`` (DeepAR's age feature counts from it);
  * training windows need a full past and targets inside the train targets;
  * validation windows have all targets inside the val targets;
  * the input for origin t is the history ``s0..t``.
No exogenous or calendar inputs by default (future covariates are not allowed).
"""
from __future__ import annotations

import tempfile
import time

import numpy as np
import pandas as pd

from dcgridbench.models.external.base import ExternalModel, steps


class GluonTSDeepAR(ExternalModel):
    library = "gluonts"
    has_samples = True

    def __init__(self, context_hours: float = 168, lags_hours: list[float] | None = None,
                 num_layers: int = 2, hidden_size: int = 40, dropout_rate: float = 0.1, lr: float = 1e-3,
                 batch_size: int = 64, num_batches_per_epoch: int = 50, max_epochs: int = 100,
                 early_stop_patience: int = 10, distribution: str = "studentt", num_samples: int = 200,
                 predict_batch: int = 256, accelerator: str = "auto", seed: int = 0, name: str = "deepar"):
        super().__init__(context_hours=context_hours, lags_hours=lags_hours, num_layers=num_layers,
                         hidden_size=hidden_size, dropout_rate=dropout_rate, lr=lr, batch_size=batch_size,
                         num_batches_per_epoch=num_batches_per_epoch, max_epochs=max_epochs,
                         early_stop_patience=early_stop_patience, distribution=distribution,
                         num_samples=num_samples)
        self.p = dict(self.params)
        self.predict_batch, self.accelerator, self.seed, self.name = predict_batch, accelerator, seed, name

    def _lags(self, task):
        if self.p["lags_hours"] is None:
            from gluonts.time_feature import get_lags_for_frequency
            return get_lags_for_frequency(task.resolution)
        return sorted({steps(h, task) for h in self.p["lags_hours"]})

    def _distr(self):
        from gluonts.torch.distributions import NormalOutput, StudentTOutput
        return {"studentt": StudentTOutput, "normal": NormalOutput}[self.p["distribution"]]()

    def fit(self, df, idx, task, val_idx=None):
        import lightning.pytorch as pl
        import torch
        from gluonts.torch.model.deepar import DeepAREstimator
        from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler

        pl.seed_everything(self.seed, workers=True)
        H = task.horizon
        C = steps(self.p["context_hours"], task)
        self.lags = self._lags(task)
        self.P = C + max(self.lags)  # rows of history DeepAR reads
        self.y = df[task.target].to_numpy(np.float32)
        self.s0 = max(0, int(idx.min()) + 1 - self.P)
        self.start = pd.Period(df.index[self.s0].tz_convert(None), freq=task.resolution)
        train_stop = int(idx.max()) + H + 1
        train = [{"start": self.start, "target": self.y[self.s0:train_stop]}]
        val, val_sampler = None, None
        if val_idx is not None and len(val_idx):
            if int(val_idx.min()) < train_stop - 1:
                raise ValueError("validation origins overlap the training targets")
            val_stop = int(val_idx.max()) + H + 1
            val = [{"start": self.start, "target": self.y[self.s0:val_stop]}]
            val_sampler = ValidationSplitSampler(min_past=int(val_idx.min()) + 1 - self.s0, min_future=H)
        self._tmp = tempfile.TemporaryDirectory()
        callbacks = []
        if val is not None and self.p["early_stop_patience"]:
            callbacks.append(pl.callbacks.EarlyStopping(monitor="val_loss", patience=self.p["early_stop_patience"]))
        acc = self.accelerator if self.accelerator != "auto" else ("gpu" if torch.cuda.is_available() else "cpu")
        est = DeepAREstimator(
            freq=task.resolution, prediction_length=H, context_length=C, lags_seq=self.lags, time_features=[],
            num_layers=self.p["num_layers"], hidden_size=self.p["hidden_size"],
            dropout_rate=self.p["dropout_rate"], lr=self.p["lr"], batch_size=self.p["batch_size"],
            num_batches_per_epoch=self.p["num_batches_per_epoch"], distr_output=self._distr(),
            num_parallel_samples=self.p["num_samples"],
            train_sampler=ExpectedNumInstanceSampler(num_instances=1.0, min_past=self.P, min_future=H),
            validation_sampler=val_sampler,
            trainer_kwargs={"max_epochs": self.p["max_epochs"], "accelerator": acc, "devices": 1,
                            "callbacks": callbacks, "enable_progress_bar": False, "logger": False,
                            "default_root_dir": self._tmp.name})
        t0 = time.time()
        self.predictor = est.train(training_data=train, validation_data=val)
        self.fit_seconds = time.time() - t0
        self.train_stop_ = train_stop
        return self

    def sample(self, df, idx, task, n_samples=None, seed=None):
        import torch
        torch.manual_seed(self.seed if seed is None else seed)
        y = df[task.target].to_numpy(np.float32)
        n = n_samples or self.p["num_samples"]
        out = []
        for s in range(0, len(idx), self.predict_batch):
            part = idx[s:s + self.predict_batch]
            data = [{"start": self.start, "target": y[self.s0:o + 1]} for o in part]  # history up to o
            out.extend(f.samples for f in self.predictor.predict(data, num_samples=n))
        return np.stack(out).astype(np.float64)  # (n, S, H)

    def predict(self, df, idx, task):
        s = self.sample(df, np.asarray(idx), task)
        return np.sort(np.quantile(s, task.taus, axis=1).transpose(1, 2, 0), axis=-1)

    def info(self):
        d = super().info()
        d["exogenous"] = []
        d["lags_steps"] = list(map(int, getattr(self, "lags", [])))
        return d
