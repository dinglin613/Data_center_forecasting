import numpy as np
from scipy.stats import norm
from dcgridbench.decisions.reserve import block_from_scenarios, per_step_from_quantiles
from dcgridbench.scenarios.generate import fit_copula, pit, sample

taus = np.array([0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975])
rng = np.random.default_rng(0)
n, H = 400, 6
# truth: random walk errors -> strongly correlated across horizon
y = np.cumsum(rng.normal(size=(n, H)), axis=1)
sd = np.sqrt(np.arange(1, H + 1))
q = norm.ppf(taus)[None, None, :] * sd[None, :, None] * np.ones((n, 1, 1))


def test_pit_uniform_for_true_marginals():
    u = pit(y, q, taus)
    assert abs(u.mean() - 0.5) < 0.03


def test_copula_recovers_dependence_and_shapes():
    C = fit_copula(y, q, taus)
    assert C.shape == (H, H) and C[0, -1] > 0.25
    s_ind = sample(q[:50], taus, 300, "independent")
    s_cop = sample(q[:50], taus, 300, "copula", corr=C)
    assert s_cop.shape == (50, 300, H)
    lag1 = lambda s: np.corrcoef(s[:, :, 0].ravel(), s[:, :, 1].ravel())[0, 1]
    assert lag1(s_cop) > lag1(s_ind) + 0.3


def test_bootstrap_and_reserve():
    med = q[..., 4]
    resid = y - med
    avail = np.full(50, n)
    s = sample(q[:50], taus, 100, "bootstrap", resid_hist=resid, resid_avail=avail)
    assert s.shape == (50, 100, H)
    r1 = per_step_from_quantiles(y, q, taus, 0.1, 1.0)
    assert abs(r1["shortfall_prob"] - 0.1) < 0.04
    r2 = block_from_scenarios(y[:50], s, med[:50], 0.1, 1.0)
    assert r2["reserve_kw"] > 0


def test_schaake_keeps_marginals_and_dependence():
    resid = y - q[..., 4]
    avail = np.full(50, n)
    s = sample(q[:50], taus, 200, "schaake", resid_hist=resid, resid_avail=avail)
    # marginals: median of each step equals the forecast median
    assert np.allclose(np.median(s, axis=1), q[:50, :, 4], atol=0.15 * sd)
    lag1 = np.corrcoef(s[:, :, 0].ravel(), s[:, :, 1].ravel())[0, 1]
    assert lag1 > 0.4
    s2 = sample(q[:50], taus, 200, "scaled_bootstrap", resid_hist=resid / 1.35, resid_avail=avail)
    assert s2.shape == (50, 200, H)
