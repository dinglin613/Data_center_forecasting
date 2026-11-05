"""Adapter layer: output shapes, quantile order, leakage, registry and result records.

Library-backed tests are skipped when the library is missing. Run on a compute
node (the login node hangs on OpenMP code):
    srun -A <account> -p gpu-dev --gres=gpu:1 $SCRATCH/dcgridbench_env/bin/python -m pytest -q tests
"""
import numpy as np
import pandas as pd
import pytest

from dcgridbench.data.windows import Task, origins, target_windows
from dcgridbench.leaderboard import results as R
from dcgridbench.models import registry
from dcgridbench.models.external.base import context_windows, to_task_quantiles

idx = pd.date_range("2021-03-01", "2021-03-25", freq="1h", tz="UTC", inclusive="left")
rng = np.random.default_rng(0)
hour = idx.hour.to_numpy()
df = pd.DataFrame({"power_w": 30 + 5 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.5, len(idx)),
                   "gpus_alloc": 300 + rng.normal(0, 5, len(idx)),
                   "jobs_running": 200 + rng.normal(0, 5, len(idx)),
                   "gpus_queued": 100 + rng.normal(0, 5, len(idx)),
                   "jobs_queued": 50 + rng.normal(0, 5, len(idx))}, index=idx)
task = Task("t", "1h", horizon=6, train=("2021-03-01", "2021-03-14"),
            val=("2021-03-15", "2021-03-18"), test=("2021-03-19", "2021-03-24"))
TR, VA, TE = (origins(df, task, s) for s in ("train", "val", "test"))
K = len(task.taus)


def _perturb_after(frame, pos):
    """Copy of ``frame`` with every row after position ``pos`` replaced by noise."""
    g = frame.copy()
    g.iloc[pos + 1:] = rng.normal(0, 100, g.iloc[pos + 1:].shape)
    return g


def _check_quantiles(q, n):
    assert q.shape == (n, task.horizon, K)
    assert np.isfinite(q).all()
    assert np.all(np.diff(q, axis=-1) >= -1e-9)


# ---------------------------------------------------------------- helpers

def test_context_windows_stop_at_origin():
    x = np.arange(10.0)
    w = context_windows(x, np.array([2, 9]), 4)
    assert np.isnan(w[0, :1]).all() and w[0, 1:].tolist() == [0, 1, 2]
    assert w[1].tolist() == [6, 7, 8, 9]
    w2 = context_windows(np.c_[x, -x], np.array([5]), 3)
    assert w2.shape == (1, 3, 2) and w2[0, -1].tolist() == [5, -5]


def test_to_task_quantiles():
    taus = np.array(task.taus)
    q = np.broadcast_to(taus, (2, 3, K)).copy()
    assert np.allclose(to_task_quantiles(q, taus, taus), q)
    levels = np.round(np.arange(0.1, 0.91, 0.1), 2)
    qb = np.broadcast_to(levels, (2, 3, 9)).copy()  # quantile = level
    clamp = to_task_quantiles(qb, levels, taus, tail="clamp")
    assert np.allclose(clamp[..., 0], 0.1) and np.allclose(clamp[..., 4], 0.5)
    from scipy.stats import norm  # normal quantiles must be reproduced exactly
    qn = np.broadcast_to(norm.ppf(levels), (1, 1, 9)).copy()
    gau = to_task_quantiles(qn, levels, taus, tail="gaussian")
    assert np.allclose(gau[0, 0, [0, 1, -2, -1]], norm.ppf(taus[[0, 1, -2, -1]]))
    crossed = qb[..., ::-1].copy()  # crossing input still gives sorted output
    assert np.all(np.diff(to_task_quantiles(crossed, levels, taus), axis=-1) >= 0)


def test_origins_stay_in_split():
    for s, o in (("train", TR), ("val", VA), ("test", TE)):
        lo, hi = getattr(task, s)
        assert df.index[o].min() >= pd.Timestamp(lo, tz="UTC")
        assert df.index[o + task.horizon].max() < pd.Timestamp(hi, tz="UTC") + pd.Timedelta(days=1)


# ---------------------------------------------------------------- registry + results

def test_registry_configs_load_and_build():
    names = registry.list_configs()
    for need in ("persistence", "qtcn", "nhits", "deepar", "tft", "patchtst", "chronos_bolt_base"):
        assert need in names
    for n in names:
        cfg = registry.load_config(n)
        assert cfg["name"] == n
        if n != "qtcn":  # needs torch at import
            registry.build(cfg, "mit-1h-24h", seed=1)
    m = registry.build(registry.load_config("nhits"), "mit-5min-1h", seed=3)
    assert m.lookback_hours == 24 and m.seed == 3


def test_result_record_roundtrip(tmp_path):
    cfg = registry.load_config("persistence")
    m = registry.fit_model(registry.build(cfg, "t"), df, TR, task, VA)
    Y = target_windows(df.power_w.to_numpy(), task.horizon)
    reg = rng.integers(0, 4, len(TE))
    q = m.predict(df, TE, task)
    met = R.stage12_metrics(Y[TE], q, task.taus, reg)
    assert set(met["stage2"]["cov90_by_regime"]) <= set(R.REGIME_NAMES.values())
    info = {"library": "dcgridbench", "library_version": None, "hyperparameters": {}, "zero_shot": False}
    rec = R.make_record(task=task, cfg=cfg, model_info=info, seed=0, splits={}, metrics={"test": met},
                        runtime={"fit_seconds": 0.0, "predict_seconds": 0.0}, predictions_file=None)
    R.write(rec, tmp_path / "r.json")
    bad = dict(rec, metrics={"test": {"stage1": {}, "stage2": {}}})
    with pytest.raises(ValueError):
        R.validate(bad)


# ---------------------------------------------------------------- NeuralForecast

@pytest.fixture(scope="module")
def nhits():
    pytest.importorskip("neuralforecast")
    from dcgridbench.models.external.neuralforecast_adapter import NeuralForecastModel
    kw = dict(model="NHITS", lookback_hours=24, max_steps=30, val_check_steps=10, batch_size=8,
              windows_batch_size=64, scaler_type="robust", accelerator="cpu", seed=0)
    return NeuralForecastModel(**kw).fit(df, TR, task, val_idx=VA), kw


def test_nf_segment_respects_splits():
    from dcgridbench.models.external.neuralforecast_adapter import NeuralForecastModel
    first, stop, val_size = NeuralForecastModel.segment(df, TR, VA, task)
    assert first == TR.min()
    assert stop - 1 == VA.max() + task.horizon            # last val target
    assert stop - val_size - 1 == TR.max() + task.horizon  # last train target
    assert stop <= TE.min()                                # nothing from test


def test_nf_shapes_and_order(nhits):
    m, _ = nhits
    assert m.exog == ["gpus_alloc", "jobs_running", "gpus_queued", "jobs_queued"]
    q = m.predict(df, TE, task)
    _check_quantiles(q, len(TE))
    shuffled = TE[::-1]
    assert np.allclose(m.predict(df, shuffled, task), q[::-1])


def test_nf_prediction_ignores_future(nhits):
    m, _ = nhits
    o = TE[len(TE) // 2]
    q = m.predict(df, np.array([o]), task)
    assert np.allclose(m.predict(_perturb_after(df, o), np.array([o]), task), q)


def test_nf_training_ignores_test_period(nhits):
    from dcgridbench.models.external.neuralforecast_adapter import NeuralForecastModel
    m, kw = nhits
    last_val_target = VA.max() + task.horizon
    m2 = NeuralForecastModel(**kw).fit(_perturb_after(df, last_val_target), TR, task, val_idx=VA)
    assert np.allclose(m.predict(df, VA, task), m2.predict(df, VA, task), atol=1e-5)


def test_nf_deepar_distribution_loss():
    pytest.importorskip("neuralforecast")
    from dcgridbench.models.external.neuralforecast_adapter import NeuralForecastModel
    m = NeuralForecastModel(model="DeepAR", lookback_hours=24, loss="distribution", distribution="StudentT",
                            max_steps=10, val_check_steps=5, batch_size=8, windows_batch_size=32,
                            lstm_hidden_size=16, accelerator="cpu").fit(df, TR, task, val_idx=VA)
    assert m.exog == []  # DeepAR takes no past-only exogenous inputs
    _check_quantiles(m.predict(df, TE[:20], task), 20)


# ---------------------------------------------------------------- Chronos

@pytest.fixture(scope="module")
def bolt():
    pytest.importorskip("chronos")
    from dcgridbench.models.external.chronos_adapter import ChronosModel
    try:
        return ChronosModel(checkpoint="amazon/chronos-bolt-tiny", context_steps=128).fit(df, TR, task)
    except OSError as e:  # no network / hub access
        pytest.skip(f"cannot load checkpoint: {e}")


def test_chronos_shapes_and_leakage(bolt):
    q = bolt.predict(df, TE, task)
    _check_quantiles(q, len(TE))
    o = TE[5]
    assert np.allclose(bolt.predict(_perturb_after(df, o), np.array([o]), task), q[5:6], atol=1e-4)
    with pytest.raises(NotImplementedError):
        bolt.sample(df, TE[:2], task)


# ---------------------------------------------------------------- GluonTS DeepAR

@pytest.fixture(scope="module")
def gdeepar():
    pytest.importorskip("gluonts.torch")
    from dcgridbench.models.external.gluonts_adapter import GluonTSDeepAR
    return GluonTSDeepAR(context_hours=24, lags_hours=[1, 2, 24], hidden_size=8, max_epochs=2,
                         num_batches_per_epoch=5, num_samples=50, accelerator="cpu").fit(df, TR, task, val_idx=VA)


def test_gluonts_deepar_shapes_samples_leakage(gdeepar):
    m = gdeepar
    assert m.train_stop_ - 1 == TR.max() + task.horizon  # last train target
    q = m.predict(df, TE, task)
    _check_quantiles(q, len(TE))
    s = m.sample(df, TE[:7], task, n_samples=30, seed=1)
    assert s.shape == (7, 30, task.horizon) and np.isfinite(s).all()
    o = TE[len(TE) // 2]
    a = m.sample(df, np.array([o]), task, 30, seed=2)
    b = m.sample(_perturb_after(df, o), np.array([o]), task, 30, seed=2)
    assert np.allclose(a, b)
