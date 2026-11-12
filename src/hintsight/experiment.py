"""Part 3: how big must a practice-feature A/B test be, and how much does CUPED shrink it?

Unit = student. Outcome = accuracy on the second half of their main problems ("post").
Covariate = accuracy on the first half ("pre"), which a real test would take from before launch.
Everything analytic is checked by re-randomizing the real students (A/A) and by planting a known
lift on real students (A/B). The planted lift is the only simulated quantity in this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
POWER = 0.80


def load_students(con, min_problems: int = 20) -> pd.DataFrame:
    df = con.execute(f"SELECT * FROM student_periods WHERE n_total >= {min_problems} ORDER BY user_id").df()
    df["pre"] = df["pre_correct"] / df["pre_n"]
    df["post"] = df["post_correct"] / df["post_n"]
    return df


def mde(sd: float, n_per_arm: int, alpha: float = ALPHA, power: float = POWER) -> float:
    """Minimum detectable difference in means for a two-arm, equal-size, two-sided z-test."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * sd * np.sqrt(2.0 / n_per_arm))


def n_per_arm_for(sd: float, effect: float, alpha: float = ALPHA, power: float = POWER) -> int:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(2 * (z * sd / effect) ** 2))


def cuped(y: np.ndarray, x: np.ndarray, theta: float | None = None) -> np.ndarray:
    """CUPED-adjusted outcome y - theta (x - mean x), theta = cov(y, x) / var(x) on pooled data."""
    if theta is None:
        theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean())


def welch_p(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def icc_oneway(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """ANOVA estimator of the intraclass correlation and the average cluster size m0."""
    df = pd.DataFrame({"v": values, "g": groups})
    g = df.groupby("g")["v"]
    k, N = g.ngroups, len(df)
    n_i = g.size().to_numpy()
    grand = df["v"].mean()
    msb = (n_i * (g.mean().to_numpy() - grand) ** 2).sum() / (k - 1)
    msw = ((df["v"] - g.transform("mean")) ** 2).sum() / (N - k)
    m0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    icc = (msb - msw) / (msb + (m0 - 1) * msw)
    return float(icc), float(m0)


def simulate(df: pd.DataFrame, n_sims: int = 2000, lift_flip_prob: float = 0.0, seed: int = 0) -> dict:
    """Randomly split real students 50/50 n_sims times.

    lift_flip_prob = 0 is an A/A test. Otherwise each treated student's wrong post-period answers are
    flipped to right with that probability (a planted, simulated lift on top of real data)."""
    rng = np.random.default_rng(seed)
    pre = df["pre"].to_numpy()
    post_correct = df["post_correct"].to_numpy().astype(np.int64)
    post_n = df["post_n"].to_numpy().astype(np.int64)
    n = len(df)
    p_naive, p_cuped, est_naive, est_cuped, true_lift = [], [], [], [], []
    for _ in range(n_sims):
        treat = np.zeros(n, bool)
        treat[rng.choice(n, n // 2, replace=False)] = True
        pc = post_correct.copy()
        if lift_flip_prob > 0:
            wrong = post_n[treat] - pc[treat]
            pc[treat] += rng.binomial(wrong, lift_flip_prob)
        y = pc / post_n
        ya = cuped(y, pre)
        p_naive.append(welch_p(y[treat], y[~treat]))
        p_cuped.append(welch_p(ya[treat], ya[~treat]))
        est_naive.append(y[treat].mean() - y[~treat].mean())
        est_cuped.append(ya[treat].mean() - ya[~treat].mean())
        true_lift.append(((pc - post_correct) / post_n)[treat].mean())
    p_naive, p_cuped = np.array(p_naive), np.array(p_cuped)
    return {
        "n_sims": n_sims, "n_per_arm": n // 2, "lift_flip_prob": lift_flip_prob,
        "mean_true_lift": float(np.mean(true_lift)),
        "rejection_rate_naive": float((p_naive < ALPHA).mean()),
        "rejection_rate_cuped": float((p_cuped < ALPHA).mean()),
        "estimate_mean_naive": float(np.mean(est_naive)),
        "estimate_mean_cuped": float(np.mean(est_cuped)),
        "estimate_sd_naive": float(np.std(est_naive, ddof=1)),
        "estimate_sd_cuped": float(np.std(est_cuped, ddof=1)),
        "p_values_naive": p_naive, "p_values_cuped": p_cuped,
    }


def flip_prob_for_lift(df: pd.DataFrame, lift: float) -> float:
    """Flip probability whose expected lift in mean post accuracy equals `lift`."""
    return float(lift / (1 - df["post"]).mean())


def analyze(con, n_sims: int = 2000, seed: int = 7) -> dict:
    df = load_students(con)
    sd = float(df["post"].std(ddof=1))
    rho = float(np.corrcoef(df["pre"], df["post"])[0, 1])
    sd_cuped = float(cuped(df["post"].to_numpy(), df["pre"].to_numpy()).std(ddof=1))
    n_arm = len(df) // 2
    target = 0.02
    n_naive = n_per_arm_for(sd, target)
    n_cu = n_per_arm_for(sd_cuped, target)
    mde_naive, mde_cuped = mde(sd, n_arm), mde(sd_cuped, n_arm)

    aa = simulate(df, n_sims=n_sims, seed=seed)
    # plant a lift equal to the CUPED MDE at this sample: analytic power should be ~80% for CUPED
    q = flip_prob_for_lift(df, mde_cuped)
    ab = simulate(df, n_sims=n_sims, lift_flip_prob=q, seed=seed + 1)
    z_a = stats.norm.ppf(1 - ALPHA / 2)
    se_naive, se_cuped = sd * np.sqrt(2 / n_arm), sd_cuped * np.sqrt(2 / n_arm)
    analytic_power_naive = float(stats.norm.cdf(ab["mean_true_lift"] / se_naive - z_a))
    analytic_power_cuped = float(stats.norm.cdf(ab["mean_true_lift"] / se_cuped - z_a))

    icc, m0 = icc_oneway(df["post"].to_numpy(), df["class_id"].to_numpy())
    deff = 1 + (m0 - 1) * icc
    icc_resid, _ = icc_oneway(cuped(df["post"].to_numpy(), df["pre"].to_numpy()), df["class_id"].to_numpy())

    def _clean(d):
        return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}

    return {
        "unit": "student with >= 20 main problems; pre = first half, post = second half (by order_id)",
        "n_students": int(len(df)), "n_per_arm_if_all_randomized": n_arm,
        "mean_post_accuracy": float(df["post"].mean()), "sd_post_accuracy": sd,
        "pre_post_correlation": rho, "sd_post_cuped": sd_cuped,
        "variance_reduction_cuped": float(1 - sd_cuped ** 2 / sd ** 2),
        "variance_reduction_theory_rho_sq": rho ** 2,
        "mde_pp_naive_at_available_n": 100 * mde_naive,
        "mde_pp_cuped_at_available_n": 100 * mde_cuped,
        "target_lift_pp": 100 * target,
        "n_per_arm_for_2pp_naive": n_naive, "n_per_arm_for_2pp_cuped": n_cu,
        "sample_size_saving_cuped": float(1 - n_cu / n_naive),
        "aa_test": _clean(aa),
        "ab_test_planted_lift_at_cuped_mde": _clean(ab) | {
            "analytic_power_naive": analytic_power_naive, "analytic_power_cuped": analytic_power_cuped,
            "note": "simulated: real students re-randomized; lift planted by flipping wrong post answers",
        },
        "class_randomization": {
            "n_classes": int(df["class_id"].nunique()), "icc_post_accuracy": icc,
            "avg_class_size_m0": m0, "design_effect": float(deff),
            "icc_after_cuped": icc_resid, "design_effect_after_cuped": float(1 + (m0 - 1) * icc_resid),
            "n_per_arm_for_2pp_class_randomized_naive": int(np.ceil(n_naive * deff)),
        },
        "_p_values": {"aa_naive": aa["p_values_naive"], "aa_cuped": aa["p_values_cuped"]},
    }
