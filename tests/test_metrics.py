import numpy as np
from dcgridbench.metrics.scores import (
    coverage, crps_from_quantiles, energy_score, grid_stats, pinball, scenario_stat_pit, variogram_score,
)

rng = np.random.default_rng(1)


def test_pinball_median_is_half_mae():
    y = rng.normal(size=(50, 4))
    q = np.zeros((50, 4, 1))
    assert np.isclose(pinball(y, q, [0.5]), 0.5 * np.abs(y).mean())


def test_crps_quantiles_matches_gaussian_closed_form():
    from scipy.stats import norm
    taus = (np.arange(199) + 0.5) / 199
    y = np.array([[0.3]])
    q = norm.ppf(taus)[None, None, :]
    exact = 0.3 * (2 * norm.cdf(0.3) - 1) + 2 * norm.pdf(0.3) - 1 / np.sqrt(np.pi)
    assert abs(crps_from_quantiles(y, q, taus) - exact) < 2e-3


def test_energy_score_1d_equals_crps_ensemble():
    y = rng.normal(size=(30, 1))
    s = rng.normal(size=(30, 40, 1))
    brute = np.mean(np.abs(s[..., 0] - y).mean(1)
                    - 0.5 * np.abs(s[:, :, None, 0] - s[:, None, :, 0]).mean((1, 2)))
    assert np.isclose(energy_score(y, s), brute)


def test_variogram_prefers_correct_dependence():
    n, S, H = 200, 100, 6
    y = np.cumsum(rng.normal(size=(n, H)), axis=1)                 # random walk
    good = np.cumsum(rng.normal(size=(n, S, H)), axis=2)
    bad = rng.normal(size=(n, S, H)) * np.sqrt(np.arange(1, H + 1))  # same marginals, no dependence
    assert variogram_score(y, good) < variogram_score(y, bad)


def test_grid_stats_and_coverage():
    t = np.array([[0, 2, 5, 5, 1.0]])
    g = grid_stats(t, threshold=4, dt_min=5)
    assert g["peak"][0] == 5 and g["ramp_up"][0] == 3 and g["ramp_down"][0] == 4
    assert g["dur_above"][0] == 10
    assert coverage(np.array([1.0, 3.0]), np.array([0.0, 0.0]), np.array([2.0, 2.0])) == 0.5


def test_pit_uniform_when_scenarios_from_truth():
    y = rng.normal(size=(2000, 8))
    s = rng.normal(size=(2000, 50, 8))
    pit = scenario_stat_pit(y, s, threshold=1.5, dt_min=5)
    for v in pit.values():
        assert abs(v.mean() - 0.5) < 0.03
