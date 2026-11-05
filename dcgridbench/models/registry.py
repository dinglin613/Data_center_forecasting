"""Model registry driven by YAML configs in ``configs/models/``.

A config names an adapter (or a ``module:Class`` path) and its parameters,
so adding a model needs a config file, not code changes:

    name: nhits                  # result key; defaults to the file name
    display_name: N-HiTS
    family: mlp
    adapter: neuralforecast      # see ADAPTERS, or class: pkg.module:Class
    params: {model: NHITS, lookback_hours: 168, max_steps: 2000}
    leaderboard: false           # optional: diagnostic config, never ranked
    task_params:                 # optional per-task overrides
      mit-5min-1h: {lookback_hours: 24}
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "models"

ADAPTERS = {
    "neuralforecast": "dcgridbench.models.external.neuralforecast_adapter:NeuralForecastModel",
    "chronos": "dcgridbench.models.external.chronos_adapter:ChronosModel",
    "gluonts_deepar": "dcgridbench.models.external.gluonts_adapter:GluonTSDeepAR",
    "builtin": None,
}
REQUIRED = ("family",)


def list_configs(config_dir: Path | str = CONFIG_DIR) -> list[str]:
    return sorted(p.stem for p in Path(config_dir).glob("*.yaml"))


def load_config(name_or_path: str, config_dir: Path | str = CONFIG_DIR) -> dict:
    p = Path(name_or_path)
    if p.suffix not in (".yaml", ".yml"):
        p = Path(config_dir) / f"{name_or_path}.yaml"
    cfg = yaml.safe_load(p.read_text()) or {}
    cfg.setdefault("name", p.stem)
    cfg.setdefault("params", {})
    cfg.setdefault("task_params", {})
    cfg["config_file"] = str(p)
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{p}: missing keys {missing}")
    if "class" not in cfg and cfg.get("adapter") not in ADAPTERS:
        raise ValueError(f"{p}: unknown adapter {cfg.get('adapter')!r}; use one of {list(ADAPTERS)} or class:")
    return cfg


def resolve_params(cfg: dict, task_name: str) -> dict:
    return {**cfg["params"], **(cfg["task_params"].get(task_name) or {})}


def _import(path: str):
    mod, _, attr = path.partition(":")
    return getattr(importlib.import_module(mod), attr)


def _builtin(cls_name: str):
    if cls_name == "qtcn":
        from dcgridbench.models.tcn import QuantileTCN
        return QuantileTCN
    from dcgridbench.models.baselines import MODELS
    return MODELS[cls_name]


def build(cfg: dict, task_name: str, seed: int = 0):
    """Instantiate the model described by ``cfg`` for one task and seed."""
    params = resolve_params(cfg, task_name)
    if "class" in cfg:
        cls = _import(cfg["class"])
    elif cfg["adapter"] == "builtin":
        cls = _builtin(params.pop("cls"))
    else:
        cls = _import(ADAPTERS[cfg["adapter"]])
    if cls.__init__ is object.__init__:
        return cls(**params)
    sig = inspect.signature(cls.__init__).parameters
    accepts_any = any(p.kind is p.VAR_KEYWORD for p in sig.values())
    if "seed" in sig or accepts_any:
        params["seed"] = seed
    if "name" in sig:
        params["name"] = cfg["name"]
    return cls(**params)


def fit_model(model, df, train_idx, task, val_idx):
    """Call ``fit`` with ``val_idx`` only if the model takes it (old baselines do not)."""
    if "val_idx" in inspect.signature(model.fit).parameters:
        return model.fit(df, train_idx, task, val_idx=val_idx)
    return model.fit(df, train_idx, task)
