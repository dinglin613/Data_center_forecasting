import numpy as np
import pandas as pd
from aidc_gridbench.data.windows import Task, origins, target_windows
from aidc_gridbench.models.baselines import Persistence, QuantileLGBM, SeasonalNaive
from aidc_gridbench.metrics.scores import coverage

idx = pd.date_range("2021-03-01", "2021-03-20", freq="1h", tz="UTC", inclusive="left")
rng = np.random.default_rng(0)
hour = idx.hour.to_numpy()
df = pd.DataFrame({"power_w": 100 + 20 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 3, len(idx)),
                   "gpus_alloc": 300 + rng.normal(0, 5, len(idx))}, index=idx)
task = Task("t", "1h", horizon=6, train=("2021-03-01", "2021-03-12"),
            val=("2021-03-13", "2021-03-15"), test=("2021-03-16", "2021-03-19"))


def test_origins_do_not_cross_split():
    for s in ("train", "val", "test"):
        o = origins(df, task, s)
        end = df.index[o + task.horizon]
        lo, hi = getattr(task, s)
        assert end.max() < pd.Timestamp(hi, tz="UTC") + pd.Timedelta(days=1)
        assert df.index[o].min() >= pd.Timestamp(lo, tz="UTC")


def test_target_windows():
    w = target_windows(np.arange(5.0), 2)
    assert w[0].tolist() == [1, 2] and np.isnan(w[4]).all()


def test_models_shapes_and_coverage():
    tr, te = origins(df, task, "train"), origins(df, task, "test")
    y = target_windows(df.power_w.to_numpy(), task.horizon)[te]
    for M in (Persistence, SeasonalNaive, lambda: QuantileLGBM(n_estimators=100, min_child_samples=5)):
        q = M().fit(df, tr, task).predict(df, te, task)
        assert q.shape == (len(te), 6, len(task.taus))
        assert np.all(np.diff(q, axis=-1) >= -1e-9)
        assert coverage(y, q[..., 0], q[..., -1]) > 0.7
