"""Chronos foundation models (zero-shot) from ``chronos-forecasting``.

Supports Chronos-Bolt and Chronos-2 (native quantile heads) and the original
T5 Chronos (sample paths). The context for origin t is the last
``context_steps`` rows up to and including t, left-padded with NaN (treated
as missing by Chronos) where the series is shorter.

Chronos-Bolt is trained on quantiles 0.1..0.9 only. Levels outside that range
are filled by ``tail``: "gaussian" scales the outer half-spread by the normal
quantile ratio, "clamp" repeats the outer quantile (the library default).
"""
from __future__ import annotations

import time

import numpy as np

from dcgridbench.models.external.base import SCHED_COLS, ExternalModel, context_windows, to_task_quantiles


class ChronosModel(ExternalModel):
    library = "chronos-forecasting"
    zero_shot = True

    def __init__(self, checkpoint: str = "amazon/chronos-bolt-base", context_steps: int = 2048,
                 batch_size: int = 256, tail: str = "gaussian", covariates: bool = False,
                 num_samples: int = 100, device: str | None = None, seed: int = 0, name: str | None = None):
        super().__init__(checkpoint=checkpoint, context_steps=context_steps, tail=tail,
                         covariates=covariates, num_samples=num_samples)
        self.checkpoint, self.context_steps, self.batch_size = checkpoint, context_steps, batch_size
        self.tail, self.covariates, self.num_samples = tail, covariates, num_samples
        self.device, self.seed = device, seed
        self.name = name or checkpoint.split("/")[-1]

    @property
    def has_samples(self):
        return getattr(self, "kind", None) == "t5"

    def fit(self, df, idx, task, val_idx=None):
        """Zero-shot: only loads the pretrained weights."""
        import torch
        from chronos import BaseChronosPipeline
        t0 = time.time()
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = BaseChronosPipeline.from_pretrained(self.checkpoint, device_map=dev, torch_dtype=torch.float32)
        kind = type(self.pipe).__name__
        self.kind = {"ChronosBoltPipeline": "bolt", "Chronos2Pipeline": "chronos2", "ChronosPipeline": "t5"}[kind]
        if self.covariates and self.kind != "chronos2":
            raise ValueError("past covariates are only supported by Chronos-2")
        self.fit_seconds = time.time() - t0
        return self

    def _contexts(self, df, idx, task):
        return context_windows(df[task.target].to_numpy(), idx, self.context_steps)

    def _native_quantiles(self, df, part, task):
        import torch
        ctx = self._contexts(df, part, task)
        if self.kind == "bolt":
            out = self.pipe.predict(torch.tensor(ctx, dtype=torch.float32), prediction_length=task.horizon)
            return out.numpy().transpose(0, 2, 1), list(self.pipe.quantiles)
        # Chronos-2: one dict per origin, optional past-only covariates.
        inputs = []
        for k, o in enumerate(part):
            ok = np.isfinite(ctx[k])
            item = {"target": ctx[k][ok]}
            if self.covariates:
                item["past_covariates"] = {c: context_windows(df[c].to_numpy(), np.array([o]),
                                                              self.context_steps)[0][ok]
                                           for c in SCHED_COLS if c in df}
            inputs.append(item)
        out = self.pipe.predict(inputs, prediction_length=task.horizon)
        q = np.stack([o[0].float().cpu().numpy() for o in out]).transpose(0, 2, 1)  # (n, H, Q)
        return q, list(self.pipe.quantiles)

    def predict(self, df, idx, task):
        idx = np.asarray(idx)
        if self.kind == "t5":
            s = self.sample(df, idx, task, self.num_samples, self.seed)
            return np.sort(np.quantile(s, task.taus, axis=1).transpose(1, 2, 0), axis=-1)
        qs = []
        for s in range(0, len(idx), self.batch_size):
            q, levels = self._native_quantiles(df, idx[s:s + self.batch_size], task)
            qs.append(to_task_quantiles(q, levels, task.taus, tail=self.tail))
        return np.concatenate(qs)

    def sample(self, df, idx, task, n_samples=100, seed=0):
        import torch
        if self.kind != "t5":
            raise NotImplementedError(f"{self.checkpoint} outputs quantiles, not sample paths")
        torch.manual_seed(seed)
        out = []
        for s in range(0, len(idx), self.batch_size):
            ctx = torch.tensor(self._contexts(df, idx[s:s + self.batch_size], task), dtype=torch.float32)
            out.append(self.pipe.predict(ctx, prediction_length=task.horizon, num_samples=n_samples).numpy())
        return np.concatenate(out)  # (n, S, H)

    def info(self):
        d = super().info()
        d["native_samples"] = self.has_samples
        d["exogenous"] = [c for c in SCHED_COLS] if self.covariates else []
        d["pipeline"] = getattr(self, "kind", None)
        return d
