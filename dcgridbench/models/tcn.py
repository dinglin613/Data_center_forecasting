"""Quantile TCN: dilated causal convolutions over a lookback window of the
relative features, predicting the change from the current value for every
horizon and quantile at once. Trained with pinball loss, early-stopped on val.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dcgridbench.models.baselines import make_features


class _Block(nn.Module):
    def __init__(self, c, dilation, k=3, p=0.1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.c1 = nn.Conv1d(c, c, k, dilation=dilation)
        self.c2 = nn.Conv1d(c, c, k, dilation=dilation)
        self.act, self.drop = nn.GELU(), nn.Dropout(p)

    def forward(self, x):
        h = self.drop(self.act(self.c1(nn.functional.pad(x, (self.pad, 0)))))
        h = self.drop(self.act(self.c2(nn.functional.pad(h, (self.pad, 0)))))
        return x + h


class _Net(nn.Module):
    def __init__(self, c_in, H, K, width=64, levels=7):
        super().__init__()
        self.inp = nn.Conv1d(c_in, width, 1)
        self.blocks = nn.Sequential(*[_Block(width, 2 ** i) for i in range(levels)])
        self.head = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, H * K))
        self.H, self.K = H, K

    def forward(self, x):                     # x (B, L, C)
        h = self.blocks(self.inp(x.transpose(1, 2)))[:, :, -1]
        return self.head(h).view(-1, self.H, self.K)


class QuantileTCN:
    name = "qtcn"

    def __init__(self, lookback_hours=24, epochs=40, batch=512, lr=1e-3, patience=6, seed=0, device=None):
        self.lookback_hours, self.epochs, self.batch = lookback_hours, epochs, batch
        self.lr, self.patience, self.seed = lr, patience, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _prep(self, df, task):
        F = make_features(df, task, relative=True)
        F = F.ffill().fillna(0.0).to_numpy(np.float32)
        y = df[task.target].to_numpy(np.float32)
        H = task.horizon
        Y = np.full((len(y), H), np.nan, np.float32)
        for h in range(1, H + 1):
            Y[:-h, h - 1] = y[h:] - y[:-h]
        return F, Y, y

    def _windows(self, F, idx):
        L = self.L
        ar = np.arange(-L + 1, 1)
        return F[np.clip(idx[:, None] + ar[None], 0, None)]

    def fit(self, df, idx, task, val_idx=None):
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        self.L = int(self.lookback_hours * 60 / task.dt_min)
        F, Y, _ = self._prep(df, task)
        self.mu, self.sd = F[idx].mean(0), F[idx].std(0) + 1e-6
        self.ysd = float(np.nanstd(Y[idx])) + 1e-6
        Fn = (F - self.mu) / self.sd
        taus = torch.tensor(task.taus, device=self.device)
        self.net = _Net(F.shape[1], task.horizon, len(task.taus)).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        tr = idx[np.isfinite(Y[idx]).all(1)]
        va = val_idx[np.isfinite(Y[val_idx]).all(1)] if val_idx is not None else None
        Ft = torch.tensor(Fn, device=self.device)
        Yt = torch.tensor(np.nan_to_num(Y) / self.ysd, device=self.device)
        ar = torch.arange(-self.L + 1, 1, device=self.device)

        def batch_loss(b):
            bi = torch.tensor(b, device=self.device)
            x = Ft[(bi[:, None] + ar[None]).clamp(min=0)]
            d = Yt[bi][..., None] - self.net(x)
            return torch.maximum(taus * d, (taus - 1) * d).mean()

        best, bad, state = np.inf, 0, None
        for ep in range(self.epochs):
            self.net.train()
            order = np.random.permutation(tr)
            for s in range(0, len(order), self.batch):
                b = order[s:s + self.batch]
                opt.zero_grad(); loss = batch_loss(b); loss.backward(); opt.step()
            if va is None:
                continue
            self.net.eval()
            with torch.no_grad():
                vl = np.mean([batch_loss(va[s:s + 4096]).item() for s in range(0, len(va), 4096)])
            if vl < best - 1e-5:
                best, bad = vl, 0
                state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if state is not None:
            self.net.load_state_dict(state)
        self.best_val = best
        return self

    def predict(self, df, idx, task):
        F, _, y = self._prep(df, task)
        Ft = torch.tensor((F - self.mu) / self.sd, device=self.device)
        ar = torch.arange(-self.L + 1, 1, device=self.device)
        self.net.eval(); out = []
        with torch.no_grad():
            for s in range(0, len(idx), 4096):
                bi = torch.tensor(idx[s:s + 4096], device=self.device)
                out.append(self.net(Ft[(bi[:, None] + ar[None]).clamp(min=0)]).cpu().numpy())
        q = np.concatenate(out) * self.ysd + y[idx][:, None, None]
        return np.sort(q, axis=-1)
