"""Shared random-search space and its mapping onto model parameters (docs/protocol.md).

Every trained model gets the same N_TRIALS draws per task (same RNG stream), each
mapped to its own parameter names by the ``search:`` section of its config:

    search:
      lookback: lookback_hours            # shared name -> parameter name
      scaler: {key: scaling, values: {standard: true, robust: false}}
      size: {small: {...}, medium: {...}, large: {...}}   # parameter sets
      scheduler: null                     # not supported by this model
Zero-shot models use ``search: {grid: {context_steps: [...]}}`` instead.
"""
from __future__ import annotations

import itertools
import zlib

import numpy as np

N_TRIALS = 20
SEARCH_SEEDS = (0, 1)
FINAL_SEEDS = (0, 1, 2)
LOOKBACK_HOURS = {"mit-5min-1h": [6, 24, 72], "mit-15min-4h": [24, 72, 168], "mit-1h-24h": [48, 168, 336]}
SPACE = {
    "scheduler": [True, False],
    "calendar": [True],
    "target": ["level", "change"],
    "scaler": ["standard", "robust"],
    "size": ["small", "medium", "large"],
    "dropout": [0.0, 0.1, 0.2],
    "lr": (1e-4, 3e-3),  # log-uniform
}


def shared_trials(task_name: str, n: int = N_TRIALS, seed: int = 0) -> list[dict]:
    """The same ``n`` draws for every model on a task."""
    rng = np.random.default_rng([seed, zlib.crc32(task_name.encode())])
    out = []
    for _ in range(n):
        d = {k: v[rng.integers(len(v))] for k, v in SPACE.items() if isinstance(v, list)}
        lo, hi = SPACE["lr"]
        d["lr"] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        lb = LOOKBACK_HOURS[task_name]
        d["lookback"] = lb[rng.integers(len(lb))]
        out.append({k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()})
    return out


def to_override(search: dict, shared: dict) -> dict:
    """Model-specific parameter overrides for one shared draw."""
    over = {}
    for name, value in shared.items():
        rule = search.get(name)
        if rule is None:
            continue  # model does not support this knob
        if name == "size":
            over.update(rule[value])
        elif isinstance(rule, str):
            over[rule] = value
        else:
            over[rule["key"]] = rule["values"][value]
    return over


def trials_for(cfg: dict, task_name: str) -> list[dict]:
    """List of {"trial", "shared", "override"} for a config (random search or zero-shot grid)."""
    search = cfg.get("search") or {}
    if "grid" in search:
        keys = list(search["grid"])
        combos = itertools.product(*(search["grid"][k] for k in keys))
        return [{"trial": i, "shared": {}, "override": dict(zip(keys, c))} for i, c in enumerate(combos)]
    if not search:
        return []
    return [{"trial": i, "shared": s, "override": to_override(search, s)}
            for i, s in enumerate(shared_trials(task_name))]
