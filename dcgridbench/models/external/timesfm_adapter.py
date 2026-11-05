"""TimesFM 2.5 (200M, PyTorch) zero-shot, from the ``timesfm`` package.

The context for origin t is the last ``context_steps`` rows up to and
including t (leading NaN are stripped by TimesFM). TimesFM returns the mean and
quantiles 0.1..0.9; levels outside that range are filled with the same
documented ``tail`` rule as Chronos-Bolt (see base.to_task_quantiles).
"""
from __future__ import annotations

import time

import numpy as np

from dcgridbench.models.external.base import ExternalModel, context_windows, to_task_quantiles

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class TimesFMModel(ExternalModel):
    library = "timesfm"
    zero_shot = True

    def __init__(self, checkpoint: str = "google/timesfm-2.5-200m-pytorch", context_steps: int = 2048,
                 batch_size: int = 128, tail: str = "gaussian", seed: int = 0, name: str = "timesfm"):
        if context_steps % 32:
            raise ValueError("context_steps must be a multiple of 32 (TimesFM patch length)")
        super().__init__(checkpoint=checkpoint, context_steps=context_steps, tail=tail)
        self.checkpoint, self.context_steps, self.batch_size = checkpoint, context_steps, batch_size
        self.tail, self.seed, self.name = tail, seed, name

    def fit(self, df, idx, task, val_idx=None):
        """Zero-shot: only loads and compiles the pretrained model."""
        import timesfm
        t0 = time.time()
        self.tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint, torch_compile=False)
        self.tfm.compile(timesfm.ForecastConfig(
            max_context=self.context_steps, max_horizon=max(128, task.horizon), normalize_inputs=True,
            per_core_batch_size=self.batch_size, use_continuous_quantile_head=True,
            force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
        self.fit_seconds = time.time() - t0
        return self

    def predict(self, df, idx, task):
        ctx = context_windows(df[task.target].to_numpy(), np.asarray(idx), self.context_steps)
        _, q = self.tfm.forecast(horizon=task.horizon, inputs=[c for c in ctx])  # (n, H, 10): mean, q10..q90
        q = np.asarray(q, float)[:, :task.horizon, 1:]
        return to_task_quantiles(q, LEVELS, task.taus, tail=self.tail)

    def info(self):
        d = super().info()
        d["exogenous"] = []
        return d
