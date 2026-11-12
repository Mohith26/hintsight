"""Part 2: predict first-attempt correctness on the next problem (what an adaptive system needs).

All models are evaluated with 5-fold GroupKFold by student: no student is in both train and test,
every student feature uses only that student's earlier rows, and item difficulty is estimated from
training-fold students only (in SQL).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

EPS = 1e-6


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_attempts(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT h.order_id, h.user_id, h.problem_id, h.skill, h.correct,
               h.user_prior_n, h.user_prior_correct, h.user_prior_hinted,
               h.skill_prior_success, h.skill_prior_failure,
               t.user_n, t.user_correct
        FROM attempts_hist h
        JOIN (SELECT user_id, COUNT(*) AS user_n, SUM(correct) AS user_correct
              FROM attempts GROUP BY user_id) t USING (user_id)
        ORDER BY h.order_id
    """).df()
    return df.reset_index(drop=True)


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_curve(y, p, bins: int = 10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 50:
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def item_difficulty(con, train_users: np.ndarray, prior_strength: float = 5.0) -> tuple[dict, dict, float]:
    """Smoothed per-problem and per-skill accuracy from training students only (SQL)."""
    con.register("train_users_df", pd.DataFrame({"user_id": train_users}))
    g = con.execute("SELECT AVG(correct) FROM attempts WHERE user_id IN (SELECT user_id FROM train_users_df)").fetchone()[0]
    skill = con.execute(f"""
        SELECT skill, (SUM(correct) + {prior_strength} * {g}) / (COUNT(*) + {prior_strength}) AS acc
        FROM attempts WHERE user_id IN (SELECT user_id FROM train_users_df) GROUP BY skill
    """).df()
    item = con.execute(f"""
        WITH s AS (
            SELECT skill, (SUM(correct) + {prior_strength} * {g}) / (COUNT(*) + {prior_strength}) AS skill_acc
            FROM attempts WHERE user_id IN (SELECT user_id FROM train_users_df) GROUP BY skill
        )
        SELECT a.problem_id,
               (SUM(a.correct) + {prior_strength} * ANY_VALUE(s.skill_acc)) / (COUNT(*) + {prior_strength}) AS acc
        FROM attempts a JOIN s USING (skill)
        WHERE a.user_id IN (SELECT user_id FROM train_users_df)
        GROUP BY a.problem_id
    """).df()
    con.unregister("train_users_df")
    return dict(zip(item.problem_id, item.acc)), dict(zip(skill.skill, skill.acc)), float(g)


def item_logit(df: pd.DataFrame, item_acc: dict, skill_acc: dict, g: float) -> np.ndarray:
    acc = df["problem_id"].map(item_acc)
    acc = acc.fillna(df["skill"].map(skill_acc)).fillna(g)
    return logit(acc.to_numpy())


def item_logit_crossfit(con, train: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
    """Item difficulty for TRAINING rows, cross-fitted: each training student's rows get item
    accuracies computed from the other training students only. Without this the feature contains
    the row's own label, the model over-trusts it, and test predictions come out over-confident."""
    out = np.empty(len(train))
    for itr, ite in GroupKFold(n_splits=n_splits).split(train, groups=train["user_id"].to_numpy()):
        ia, sa, g = item_difficulty(con, train.iloc[itr]["user_id"].unique())
        out[ite] = item_logit(train.iloc[ite], ia, sa, g)
    return out


def student_features(df: pd.DataFrame, leaky: bool = False) -> np.ndarray:
    if leaky:
        # the classic mistake: the student's accuracy over ALL their rows except this one,
        # i.e. it peeks at future answers
        n = df["user_n"] - 1
        c = df["user_correct"] - df["correct"]
        acc = (c + 1) / (n + 2)
        cnt = np.log1p(n)
    else:
        acc = (df["user_prior_correct"] + 1) / (df["user_prior_n"] + 2)
        cnt = np.log1p(df["user_prior_n"])
    hint_rate = (df["user_prior_hinted"] + 0.5) / (df["user_prior_n"] + 2)
    return np.column_stack([logit(acc.to_numpy()), cnt.to_numpy(), logit(hint_rate.to_numpy())])


class PFA:
    """Performance factors analysis: per-skill intercept + per-skill weights on log prior successes
    and failures, fit as one sparse logistic regression. Extra dense columns are optional."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def _design(self, df, extra):
        codes = df["skill"].map(self.skill_index).fillna(len(self.skill_index)).astype(int).to_numpy()
        k = len(self.skill_index) + 1
        n = len(df)
        rows = np.arange(n)
        s = np.log1p(df["skill_prior_success"].to_numpy(float))
        f = np.log1p(df["skill_prior_failure"].to_numpy(float))
        onehot = sp.csr_matrix((np.ones(n), (rows, codes)), shape=(n, k))
        succ = sp.csr_matrix((s, (rows, codes)), shape=(n, k))
        fail = sp.csr_matrix((f, (rows, codes)), shape=(n, k))
        # shared slopes help skills with little data
        blocks = [onehot, succ, fail, sp.csr_matrix(np.column_stack([s, f]))]
        if extra is not None:
            blocks.append(sp.csr_matrix(extra))
        return sp.hstack(blocks).tocsr()

    def fit(self, df, y, extra=None):
        self.skill_index = {s: i for i, s in enumerate(sorted(df["skill"].unique()))}
        self.model = LogisticRegression(C=self.C, max_iter=2000)
        self.model.fit(self._design(df, extra), y)
        return self

    def predict_proba(self, df, extra=None):
        return self.model.predict_proba(self._design(df, extra))[:, 1]


def elo_fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Online Elo / Rasch: P(correct) = sigmoid(theta_student - b_item), uncertainty-damped updates.
    Item difficulties are learned on training students and frozen; test students start at 0 and
    update after every answer, so each prediction uses only their earlier answers."""

    def k(n):
        return 0.4 / (1 + 0.05 * n)

    b: dict = {}
    nb: dict = {}
    theta: dict = {}
    nt: dict = {}
    for u, i, y in zip(train["user_id"].to_numpy(), train["problem_id"].to_numpy(), train["correct"].to_numpy()):
        p = sigmoid(theta.get(u, 0.0) - b.get(i, 0.0))
        theta[u] = theta.get(u, 0.0) + k(nt.get(u, 0)) * (y - p)
        b[i] = b.get(i, 0.0) - k(nb.get(i, 0)) * (y - p)
        nt[u] = nt.get(u, 0) + 1
        nb[i] = nb.get(i, 0) + 1
    # unseen items: mean difficulty of their skill's items in train
    skill_b = pd.Series(b).groupby(train.drop_duplicates("problem_id").set_index("problem_id")["skill"]).mean().to_dict()
    theta_t: dict = {}
    nt_t: dict = {}
    out = np.empty(len(test))
    for j, (u, i, s, y) in enumerate(zip(test["user_id"].to_numpy(), test["problem_id"].to_numpy(),
                                         test["skill"].to_numpy(), test["correct"].to_numpy())):
        bi = b.get(i, skill_b.get(s, 0.0))
        p = sigmoid(theta_t.get(u, 0.0) - bi)
        out[j] = p
        theta_t[u] = theta_t.get(u, 0.0) + k(nt_t.get(u, 0)) * (y - p)
        nt_t[u] = nt_t.get(u, 0) + 1
    return out


def evaluate(con, df: pd.DataFrame | None = None, n_splits: int = 5) -> dict:
    if df is None:
        df = load_attempts(con)
    y = df["correct"].to_numpy()
    groups = df["user_id"].to_numpy()
    names = ["base_rate", "item_difficulty_only", "elo_online_rasch", "pfa",
             "pfa_item_student_history",
             "pfa_item_student_history_in_sample_item_encoding", "LEAKY_pfa_item_student_full_accuracy"]
    oof = {m: np.zeros(len(df)) for m in names}
    folds = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(df, y, groups)):
        dtr, dte = df.iloc[tr], df.iloc[te]
        assert not set(dtr.user_id) & set(dte.user_id)
        item_acc, skill_acc, g = item_difficulty(con, dtr["user_id"].unique())
        il_te = item_logit(dte, item_acc, skill_acc, g)
        il_tr_naive = item_logit(dtr, item_acc, skill_acc, g)  # contains each row's own label
        il_tr = item_logit_crossfit(con, dtr)

        oof["base_rate"][te] = y[tr].mean()

        lr = LogisticRegression().fit(il_tr[:, None], y[tr])
        oof["item_difficulty_only"][te] = lr.predict_proba(il_te[:, None])[:, 1]

        oof["elo_online_rasch"][te] = elo_fit_predict(dtr, dte)

        oof["pfa"][te] = PFA().fit(dtr, y[tr]).predict_proba(dte)

        ex_tr = np.column_stack([il_tr, student_features(dtr)])
        ex_te = np.column_stack([il_te, student_features(dte)])
        oof["pfa_item_student_history"][te] = PFA().fit(dtr, y[tr], ex_tr).predict_proba(dte, ex_te)

        nx_tr = np.column_stack([il_tr_naive, student_features(dtr)])
        oof["pfa_item_student_history_in_sample_item_encoding"][te] = PFA().fit(dtr, y[tr], nx_tr).predict_proba(dte, ex_te)

        lx_tr = np.column_stack([il_tr, student_features(dtr, leaky=True)])
        lx_te = np.column_stack([il_te, student_features(dte, leaky=True)])
        oof["LEAKY_pfa_item_student_full_accuracy"][te] = PFA().fit(dtr, y[tr], lx_tr).predict_proba(dte, lx_te)

        row = {"fold": fold, "n_test": int(len(te)), "n_test_students": int(dte.user_id.nunique())}
        for m in names:
            p = oof[m][te]
            row[m] = {"auc": float(roc_auc_score(y[te], p)) if m != "base_rate" else 0.5,
                      "log_loss": float(log_loss(y[te], np.clip(p, EPS, 1 - EPS))),
                      "brier": float(brier_score_loss(y[te], p)),
                      "ece": expected_calibration_error(y[te], p)}
        folds.append(row)
        print(f"fold {fold}: " + ", ".join(f"{m}={row[m]['auc']:.4f}" for m in names))

    summary = {}
    for m in names:
        s = {}
        for metric in ["auc", "log_loss", "brier", "ece"]:
            vals = np.array([f[m][metric] for f in folds])
            s[metric + "_mean"] = float(vals.mean())
            s[metric + "_sd"] = float(vals.std(ddof=1))
        s["pooled_oof_auc"] = float(roc_auc_score(y, oof[m])) if m != "base_rate" else 0.5
        s["pooled_oof_ece"] = expected_calibration_error(y, oof[m])
        s["calibration_curve"] = calibration_curve(y, oof[m])
        summary[m] = s
    best, base = summary["pfa_item_student_history"], summary["item_difficulty_only"]
    return {
        "n_rows": int(len(df)), "n_students": int(df.user_id.nunique()), "n_problems": int(df.problem_id.nunique()),
        "n_skills": int(df.skill.nunique()), "base_rate_correct": float(y.mean()), "n_splits": n_splits,
        "split": "GroupKFold by user_id (no student in both train and test)",
        "models": summary, "folds": folds,
        "headline": {
            "best_model": "pfa_item_student_history",
            "best_auc": best["auc_mean"], "item_only_auc": base["auc_mean"],
            "auc_gain_over_item_only": best["auc_mean"] - base["auc_mean"],
            "best_ece": best["ece_mean"], "item_only_ece": base["ece_mean"],
            "best_brier": best["brier_mean"], "base_rate_brier": summary["base_rate"]["brier_mean"],
            "in_sample_item_encoding_ece": summary["pfa_item_student_history_in_sample_item_encoding"]["ece_mean"],
            "in_sample_item_encoding_auc": summary["pfa_item_student_history_in_sample_item_encoding"]["auc_mean"],
            "elo_auc": summary["elo_online_rasch"]["auc_mean"], "pfa_auc": summary["pfa"]["auc_mean"],
            "leaky_auc": summary["LEAKY_pfa_item_student_full_accuracy"]["auc_mean"],
            "leakage_auc_inflation": summary["LEAKY_pfa_item_student_full_accuracy"]["auc_mean"] - best["auc_mean"],
        },
    }
