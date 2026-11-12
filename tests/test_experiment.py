import numpy as np
import pandas as pd

from hintsight import experiment as ex


def _students(n=2000, rho=0.6, seed=0):
    rng = np.random.default_rng(seed)
    skill = rng.normal(size=n)
    pre_p = 1 / (1 + np.exp(-skill))
    post_p = 1 / (1 + np.exp(-(rho * skill + rng.normal(scale=0.6, size=n))))
    pre_n = rng.integers(10, 60, n)
    post_n = pre_n + rng.integers(0, 2, n)
    pre_c, post_c = rng.binomial(pre_n, pre_p), rng.binomial(post_n, post_p)
    return pd.DataFrame({"pre_n": pre_n, "pre_correct": pre_c, "post_n": post_n, "post_correct": post_c,
                         "pre": pre_c / pre_n, "post": post_c / post_n, "class_id": np.arange(n) % 50})


def test_mde_and_sample_size_are_inverse():
    sd = 0.22
    n = ex.n_per_arm_for(sd, 0.02)
    assert ex.mde(sd, n) <= 0.02 < ex.mde(sd, n - 5)


def test_cuped_variance_reduction_equals_rho_squared():
    df = _students()
    y, x = df.post.to_numpy(), df.pre.to_numpy()
    rho = np.corrcoef(y, x)[0, 1]
    red = 1 - ex.cuped(y, x).var(ddof=1) / y.var(ddof=1)
    assert abs(red - rho ** 2) < 1e-9
    assert abs(ex.cuped(y, x).mean() - y.mean()) < 1e-12  # keeps the mean


def test_aa_false_positive_rate_near_alpha():
    r = ex.simulate(_students(), n_sims=600, seed=1)
    assert 0.025 < r["rejection_rate_naive"] < 0.08
    assert 0.025 < r["rejection_rate_cuped"] < 0.08
    assert abs(r["estimate_mean_cuped"]) < 0.003


def test_planted_lift_is_recovered_and_cuped_has_more_power():
    df = _students()
    q = ex.flip_prob_for_lift(df, 0.03)
    r = ex.simulate(df, n_sims=300, lift_flip_prob=q, seed=2)
    assert abs(r["mean_true_lift"] - 0.03) < 0.003
    assert abs(r["estimate_mean_cuped"] - r["mean_true_lift"]) < 0.003
    assert r["rejection_rate_cuped"] > r["rejection_rate_naive"]


def test_icc_recovers_known_value():
    rng = np.random.default_rng(3)
    k, m, icc = 300, 20, 0.3
    g = np.repeat(np.arange(k), m)
    v = rng.normal(scale=np.sqrt(icc), size=k)[g] + rng.normal(scale=np.sqrt(1 - icc), size=k * m)
    est, m0 = ex.icc_oneway(v, g)
    assert abs(est - icc) < 0.05 and abs(m0 - m) < 1e-9
