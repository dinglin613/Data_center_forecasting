"""Standard result records: one JSON per (task, model, seed). See docs/result_schema.md."""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dcgridbench.metrics import scores as S

SCHEMA_VERSION = "1.0"
REGIME_NAMES = {0: "low", 1: "calm", 2: "active", 3: "volatile"}


def stage12_metrics(y, q, taus, reg=None) -> dict:
    """Stage 1 (forecast accuracy) and stage 2 (calibration) scores of quantiles ``q``."""
    taus = np.asarray(taus)
    lo, hi, md = taus.searchsorted(0.05), taus.searchsorted(0.95), taus.searchsorted(0.5)
    l50, h50 = taus.searchsorted(0.25), taus.searchsorted(0.75)
    out = {
        "stage1": {"crps_kw": S.crps_from_quantiles(y, q, taus),
                   "mae_kw": float(np.mean(np.abs(y - q[..., md]))),
                   "pinball_kw": S.pinball(y, q, taus)},
        "stage2": {"cov90": S.coverage(y, q[..., lo], q[..., hi]),
                   "cov50": S.coverage(y, q[..., l50], q[..., h50]),
                   "width90_kw": S.interval_width(q[..., lo], q[..., hi]),
                   "winkler90": S.winkler(y, q[..., lo], q[..., hi], 0.1)},
    }
    if reg is not None:
        g_all = [g for g in REGIME_NAMES if np.any(reg == g)]
        out["stage1"]["crps_by_regime"] = {REGIME_NAMES[g]: S.crps_from_quantiles(y[reg == g], q[reg == g], taus)
                                           for g in g_all}
        out["stage2"]["cov90_by_regime"] = {REGIME_NAMES[g]: S.coverage(y[reg == g], q[reg == g][..., lo],
                                                                        q[reg == g][..., hi]) for g in g_all}
        out["stage2"]["width90_by_regime"] = {REGIME_NAMES[g]: S.interval_width(q[reg == g][..., lo],
                                                                                q[reg == g][..., hi]) for g in g_all}
    return out


def git_commit(repo: Path | str = ".") -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except Exception:
        return None


def environment() -> dict:
    env = {"python": platform.python_version(), "host": platform.node()}
    try:
        import torch
        env["torch"] = torch.__version__
        env["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        env["device"] = "cpu"
    return env


def make_record(*, task, cfg, model_info, seed, splits, metrics, runtime, predictions_file, extra=None) -> dict:
    rec = {
        "schema_version": SCHEMA_VERSION,
        "task": task.name,
        "task_spec": {"resolution": task.resolution, "horizon": task.horizon, "stride": task.stride,
                      "taus": list(task.taus), "unit": "kW"},
        "model": {"name": cfg["name"], "display_name": cfg.get("display_name", cfg["name"]),
                  "family": cfg["family"], "adapter": cfg.get("adapter", cfg.get("class")),
                  "config_file": cfg.get("config_file"), "reference": cfg.get("reference"),
                  "leaderboard": bool(cfg.get("leaderboard", True)),
                  **model_info},
        "seed": int(seed),
        "splits": splits,
        "metrics": metrics,
        "runtime": runtime,
        "predictions_file": predictions_file,
        "environment": environment(),
        "code_commit": git_commit(Path(__file__).resolve().parents[2]),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        rec.update(extra)
    validate(rec)
    return rec


_REQ = {
    "schema_version": str, "task": str, "task_spec": dict, "model": dict, "seed": int, "splits": dict,
    "metrics": dict, "runtime": dict, "predictions_file": (str, type(None)), "environment": dict,
}
_REQ_MODEL = {"name": str, "family": str, "library": str, "library_version": (str, type(None)),
              "hyperparameters": dict, "zero_shot": bool}
_REQ_STAGE = {"stage1": ("crps_kw", "mae_kw"), "stage2": ("cov90", "width90_kw", "winkler90")}


def validate(rec: dict) -> None:
    """Raise ValueError if ``rec`` does not follow the schema."""
    def need(d, spec, where):
        for k, t in spec.items():
            if k not in d:
                raise ValueError(f"{where}: missing '{k}'")
            if not isinstance(d[k], t):
                raise ValueError(f"{where}.{k}: expected {t}, got {type(d[k]).__name__}")
    need(rec, _REQ, "record")
    need(rec["model"], _REQ_MODEL, "model")
    if "test" not in rec["metrics"]:
        raise ValueError("metrics: missing 'test'")
    for split, m in rec["metrics"].items():
        for stage, keys in _REQ_STAGE.items():
            for k in keys:
                v = m.get(stage, {}).get(k)
                if not isinstance(v, float) or not np.isfinite(v):
                    raise ValueError(f"metrics.{split}.{stage}.{k}: not a finite number")
    for k in ("fit_seconds", "predict_seconds"):
        if not isinstance(rec["runtime"].get(k), (int, float)):
            raise ValueError(f"runtime.{k}: missing")


def write(rec: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, default=_jsonable))
    return path


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
