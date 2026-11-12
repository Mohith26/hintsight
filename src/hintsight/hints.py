"""Part 1: do hints on problem t help a student get problem t+1 right?

Linear probability models with student-clustered standard errors. Each step of the ladder removes
one source of selection bias; the gap between the first and last rows is the bias of the naive read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .fe import fe_ols


def load_pairs(con) -> pd.DataFrame:
    return con.execute("SELECT * FROM pairs ORDER BY order_id").df()


def like_for_like(pairs: pd.DataFrame) -> pd.DataFrame:
    """Pairs where problem t was wrong. Treated = asked for a hint; control = typed a wrong answer
    with no hint. (ASSISTments scores any hint as wrong, and rows whose first action was the
    'break into steps' scaffold without a hint are neither group, so they are dropped.)"""
    wrong = pairs[pairs["correct"] == 0]
    keep = (wrong["hinted"] == 1) | (wrong["first_action"] == 0)
    return wrong[keep].copy()


def _controls(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["pre_acc"] = (df["pre_correct"] + 1) / (df["pre_n"] + 2)
    out["log_pre_n"] = np.log1p(df["pre_n"])
    out["log_opportunity"] = np.log1p(df["opportunity"])
    out["position"] = np.minimum(df["pos_in_assignment"], 20)
    return out


def estimate_ladder(pairs: pd.DataFrame) -> dict:
    res: dict = {}
    u = pairs["user_id"].to_numpy()

    # 1. naive: every pair, hint vs no hint
    r = fe_ols(pairs["next_correct"], pairs[["hinted"]], ["hinted"], [], cluster=u)
    res["1_naive_all_pairs"] = r.get("hinted") | {
        "next_correct_rate_hinted": float(pairs.loc[pairs.hinted == 1, "next_correct"].mean()),
        "next_correct_rate_not_hinted": float(pairs.loc[pairs.hinted == 0, "next_correct"].mean()),
    }

    d = like_for_like(pairs)
    du = d["user_id"].to_numpy()
    res["2_like_for_like_wrong_on_t"] = fe_ols(d["next_correct"], d[["hinted"]], ["hinted"], [], cluster=du).get("hinted") | {
        "next_correct_rate_hinted": float(d.loc[d.hinted == 1, "next_correct"].mean()),
        "next_correct_rate_wrong_no_hint": float(d.loc[d.hinted == 0, "next_correct"].mean()),
    }

    C = _controls(d)
    X = pd.concat([d[["hinted"]], C], axis=1)
    res["3_ability_adjusted_skill_fe"] = fe_ols(d["next_correct"], X, list(X.columns), [d["skill"].to_numpy()], cluster=du).get("hinted")

    X4 = pd.concat([d[["hinted"]], C[["log_opportunity", "position"]]], axis=1)
    res["4_student_fe"] = fe_ols(d["next_correct"], X4, list(X4.columns),
                                 [du, d["skill"].to_numpy()], cluster=du).get("hinted")

    ua = (d["user_id"].astype(str) + "_" + d["assignment_id"].astype(str)).to_numpy()
    X5 = d[["hinted"]].assign(position=C["position"])
    r5 = fe_ols(d["next_correct"], X5, list(X5.columns), [ua, d["next_problem_id"].to_numpy()], cluster=du)
    res["5_student_x_assignment_fe_next_problem_fe"] = r5.get("hinted")

    # placebo: a hint on t cannot cause the answer on t-1. A nonzero "effect" means within-session
    # struggle still selects who takes hints, so the causal reading of row 5 is an upper/lower bound.
    p = d.dropna(subset=["prev_correct"])
    pua = (p["user_id"].astype(str) + "_" + p["assignment_id"].astype(str)).to_numpy()
    Xp = p[["hinted"]].assign(position=np.minimum(p["pos_in_assignment"], 20))
    res["placebo_hint_on_t_vs_correct_on_t_minus_1"] = fe_ols(
        p["prev_correct"], Xp, list(Xp.columns), [pua, p["problem_id"].to_numpy()],
        cluster=p["user_id"].to_numpy()).get("hinted")

    # 6. same as 5, also holding fixed whether t-1 was right (momentum)
    X6 = Xp.assign(prev_correct=p["prev_correct"].astype(float))
    res["6_row5_plus_prev_correct"] = fe_ols(
        p["next_correct"], X6, list(X6.columns), [pua, p["next_problem_id"].to_numpy()],
        cluster=p["user_id"].to_numpy()).get("hinted")

    # 7. placebo-adjusted: change from t-1 to t+1, hinters vs non-hinters who both missed t
    #    (a difference-in-differences that nets out the pre-existing dip seen in the placebo)
    delta = p["next_correct"].astype(float) - p["prev_correct"].astype(float)
    res["7_placebo_adjusted_change_t_minus_1_to_t_plus_1"] = fe_ols(
        delta, Xp, list(Xp.columns), [pua], cluster=p["user_id"].to_numpy()).get("hinted")

    # dose: bottom-out hint (answer revealed) vs partial hints only, spec 5
    d2 = d.assign(partial_hint=((d.hinted == 1) & (d.bottom_hint == 0)).astype(int),
                  bottom_out_hint=((d.hinted == 1) & (d.bottom_hint == 1)).astype(int))
    X7 = d2[["partial_hint", "bottom_out_hint"]].assign(position=C["position"])
    r7 = fe_ols(d2["next_correct"], X7, list(X7.columns), [ua, d2["next_problem_id"].to_numpy()], cluster=du)
    res["5b_by_hint_depth"] = {"partial_hint": r7.get("partial_hint"), "bottom_out_hint": r7.get("bottom_out_hint")}

    naive = res["1_naive_all_pairs"]["estimate"]
    lfl = res["2_like_for_like_wrong_on_t"]["estimate"]
    final = res["5_student_x_assignment_fe_next_problem_fe"]["estimate"]
    res["summary"] = {
        "n_pairs_all": int(len(pairs)),
        "n_pairs_wrong_on_t": int(len(d)),
        "n_students_wrong_on_t": int(d["user_id"].nunique()),
        "share_hinted_among_wrong": float(d["hinted"].mean()),
        "naive_pp": 100 * naive,
        "like_for_like_pp": 100 * lfl,
        "within_student_assignment_pp": 100 * final,
        "naive_minus_within_pp": 100 * (naive - final),
        "like_for_like_minus_within_pp": 100 * (lfl - final),
        "share_of_naive_gap_removed_by_within": (naive - final) / naive,
        "placebo_pp": 100 * res["placebo_hint_on_t_vs_correct_on_t_minus_1"]["estimate"],
        "placebo_adjusted_pp": 100 * res["7_placebo_adjusted_change_t_minus_1_to_t_plus_1"]["estimate"],
        "pre_acc_hinters_vs_nonhinters": [
            float(_controls(d).loc[d.hinted == 1, "pre_acc"].mean()),
            float(_controls(d).loc[d.hinted == 0, "pre_acc"].mean()),
        ],
    }
    return res
