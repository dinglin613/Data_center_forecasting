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


# ---------------------------------------------------------------- Phase 2: calendar, change target, tuning, stages

def test_calendar_is_time_only():
    from dcgridbench.models.external.base import calendar
    c = calendar(df.index)
    assert c.shape == (len(df), 4) and np.allclose(c[:, 0] ** 2 + c[:, 1] ** 2, 1)
    assert np.allclose(calendar(df.index[24:48]), c[24:48])


def test_nf_change_target_calendar_and_leakage():
    pytest.importorskip("neuralforecast")
    from dcgridbench.models.external.neuralforecast_adapter import NeuralForecastModel
    m = NeuralForecastModel(model="NHITS", lookback_hours=24, target="change", calendar=True, exogenous=True,
                            max_steps=20, val_check_steps=10, batch_size=8, windows_batch_size=64,
                            scaler_type="standard", accelerator="cpu").fit(df, TR, task, val_idx=VA)
    assert m.futr == ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    q = m.predict(df, TE, task)
    _check_quantiles(q, len(TE))
    o = TE[len(TE) // 2]
    # future targets and scheduler values may change; calendar depends on time stamps only
    assert np.allclose(m.predict(_perturb_after(df, o), np.array([o]), task), q[len(TE) // 2:len(TE) // 2 + 1])
    # change target: the median moves one-for-one with the level of the input window
    g = df.copy()
    g["power_w"] += 50.0
    assert np.allclose(m.predict(g, np.array([o]), task)[..., 4], q[len(TE) // 2, :, 4] + 50.0, atol=1e-2)


def test_gluonts_change_target():
    pytest.importorskip("gluonts.torch")
    from dcgridbench.models.external.gluonts_adapter import GluonTSDeepAR
    m = GluonTSDeepAR(context_hours=24, lags_hours=[1, 2, 24], hidden_size=8, max_epochs=1,
                      num_batches_per_epoch=3, num_samples=20, target="change", calendar=True,
                      accelerator="cpu").fit(df, TR, task, val_idx=VA)
    s = m.sample(df, TE[:5], task, n_samples=20, seed=0)
    assert s.shape == (5, 20, task.horizon) and np.isfinite(s).all()
    o = TE[len(TE) // 2]
    assert np.allclose(m.sample(df, np.array([o]), task, 20, seed=3),
                       m.sample(_perturb_after(df, o), np.array([o]), task, 20, seed=3))


@pytest.fixture(scope="module")
def tfm():
    pytest.importorskip("timesfm")
    from dcgridbench.models.external.timesfm_adapter import TimesFMModel
    try:
        return TimesFMModel(context_steps=128, batch_size=32).fit(df, TR, task)
    except OSError as e:
        pytest.skip(f"cannot load checkpoint: {e}")


def test_timesfm_shapes_and_leakage(tfm):
    q = tfm.predict(df, TE, task)
    _check_quantiles(q, len(TE))
    o = TE[7]
    assert np.allclose(tfm.predict(_perturb_after(df, o), np.array([o]), task), q[7:8], atol=1e-3)


def test_tuning_same_draws_for_all_models():
    from dcgridbench.tuning import N_TRIALS, trials_for
    a = trials_for(registry.load_config("nhits"), "mit-1h-24h")
    b = trials_for(registry.load_config("deepar"), "mit-1h-24h")
    assert len(a) == len(b) == N_TRIALS
    assert all(x["shared"] == y["shared"] for x, y in zip(a, b))
    assert "exogenous" in a[0]["override"] and "context_hours" in b[0]["override"]
    assert all(t["shared"]["lookback"] in (48, 168, 336) for t in a)
    z = trials_for(registry.load_config("chronos_bolt_base"), "mit-1h-24h")
    assert [t["override"]["context_steps"] for t in z] == [512, 1024, 2048]
    for n in registry.list_configs():  # every override builds
        cfg = registry.load_config(n)
        for t in trials_for(cfg, "mit-5min-1h")[:3]:
            registry.build(cfg, "mit-5min-1h", overrides=t["override"])


def test_stage_scoring_on_stream():
    from dcgridbench.leaderboard.stages import score_stream
    m = registry.fit_model(registry.build(registry.load_config("persistence"), "t"), df, TR, task, VA)
    Y = target_windows(df.power_w.to_numpy(), task.horizon)
    stream = np.concatenate([VA, TE])
    st = {"q": m.predict(df, stream, task), "y": Y[stream], "t": stream, "n_val": len(VA),
          "reg": rng.integers(0, 4, len(stream))}
    native = rng.normal(30, 3, (len(TE), 10, task.horizon))
    r = score_stream(st, task, native=native, n_scen=20)
    assert set(r["calibration"]) == {"raw", "static", "regime", "online", "online_regime"}
    assert "native" in r["stage3"]["raw"] and "native" not in r["stage3"]["online_regime"]
    c = r["stage4"]["online_regime"]["block"]["schaake"]["0.1"]["cost"]
    assert np.isfinite(c) and c > 0
