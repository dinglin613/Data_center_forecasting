import numpy as np
from aidc_gridbench.models.conformal import calibrate_online, calibrate_static

taus = np.array([0.05, 0.5, 0.95])


def test_static_fixes_shifted_forecast():
    rng = np.random.default_rng(0)
    y = rng.normal(5, 1, size=(4000, 2))
    q = np.broadcast_to(np.array([-1.645, 0, 1.645]), (4000, 2, 3)).copy()  # centred at 0, truth at 5
    out = calibrate_static(q[:2000], y[:2000], q[2000:], taus)
    cov = np.mean((y[2000:] >= out[..., 0]) & (y[2000:] <= out[..., 2]))
    assert abs(cov - 0.9) < 0.03


def test_online_is_causal():
    # targets after origin i must not change the correction applied at origin i
    rng = np.random.default_rng(1)
    n, H = 600, 3
    t = np.arange(n)
    y = rng.normal(size=(n, H)); q = np.zeros((n, H, 3))
    base = calibrate_online(t, y, q, taus, start=300, window=200, min_n=10)
    y2 = y.copy(); y2[450:] += 100.0          # tamper with the future of origin 449
    alt = calibrate_online(t, y2, q, taus, start=300, window=200, min_n=10)
    # origin 449 + h sees targets of origins <= 449 - h - 1 only, all untouched
    assert np.allclose(base[:150], alt[:150])
