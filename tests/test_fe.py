"""Absorbed fixed effects must give exactly the dummy-variable OLS slopes."""
import numpy as np
import statsmodels.api as sm

from hintsight.fe import demean, fe_ols


def _dummies(g):
    _, inv = np.unique(g, return_inverse=True)
    return np.eye(inv.max() + 1)[inv]


def test_one_way_fe_matches_dummy_regression():
    rng = np.random.default_rng(1)
    n, G = 600, 30
    g = rng.integers(0, G, n)
    x = rng.normal(size=(n, 2)) + 0.5 * g[:, None] / G
    y = 1.5 * x[:, 0] - 0.7 * x[:, 1] + rng.normal(size=G)[g] + rng.normal(size=n)
    r = fe_ols(y, x, ["a", "b"], [g], cluster=g)
    beta = np.linalg.lstsq(np.column_stack([x, _dummies(g)]), y, rcond=None)[0][:2]
    assert np.allclose(r.coef, beta, atol=1e-8)


def test_two_way_fe_matches_dummy_regression():
    rng = np.random.default_rng(2)
    n = 800
    g1, g2 = rng.integers(0, 25, n), rng.integers(0, 15, n)
    x = rng.normal(size=n) + 0.1 * g1 - 0.2 * g2
    y = 0.4 * x + 0.3 * g1 - 0.5 * g2 + rng.normal(size=n)
    r = fe_ols(y, x, ["x"], [g1, g2], cluster=g1)
    D = np.column_stack([x, _dummies(g1), _dummies(g2)[:, 1:]])
    beta = np.linalg.lstsq(D, y, rcond=None)[0][0]
    assert abs(r.coef[0] - beta) < 1e-7


def test_cluster_se_matches_statsmodels_without_fe():
    rng = np.random.default_rng(3)
    n = 1000
    cl = rng.integers(0, 80, n)
    x = rng.normal(size=n) + rng.normal(size=80)[cl]
    y = 2 + 0.5 * x + rng.normal(size=80)[cl] + rng.normal(size=n)
    r = fe_ols(y, x, ["x"], [], cluster=cl)
    sm_fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type="cluster", cov_kwds={"groups": cl})
    assert np.isclose(r.coef[0], sm_fit.params[1])
    assert np.isclose(r.se[0], sm_fit.bse[1], rtol=1e-6)


def test_demean_is_idempotent_projection():
    rng = np.random.default_rng(4)
    g1, g2 = rng.integers(0, 10, 300), rng.integers(0, 7, 300)
    M = rng.normal(size=(300, 3))
    D = demean(M, [g1, g2])
    assert np.allclose(demean(D, [g1, g2]), D, atol=1e-8)
    for g in (g1, g2):
        for k in np.unique(g):
            assert np.allclose(D[g == k].mean(axis=0), 0, atol=1e-7)
