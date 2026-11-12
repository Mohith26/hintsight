import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from hintsight import predict


def test_ece_zero_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=200_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert predict.expected_calibration_error(y, p) < 0.01
    assert predict.expected_calibration_error(y, np.clip(p + 0.2, 0, 1)) > 0.15


def test_item_difficulty_uses_only_training_students(con):
    users = [r[0] for r in con.execute("SELECT DISTINCT user_id FROM attempts ORDER BY 1").fetchall()]
    train = np.array(users[:30])
    item_acc, _, _ = predict.item_difficulty(con, train)
    # changing the labels of a test student must not move any item estimate
    con.execute("CREATE TEMP TABLE backup AS SELECT * FROM attempts")
    con.execute(f"UPDATE attempts SET correct = 1 - correct WHERE user_id = {users[-1]}")
    item_acc2, _, _ = predict.item_difficulty(con, train)
    con.execute("DELETE FROM attempts; INSERT INTO attempts SELECT * FROM backup; DROP TABLE backup")
    assert item_acc == item_acc2


def test_honest_student_features_ignore_the_future():
    df = pd.DataFrame({"user_prior_n": [0, 1, 2], "user_prior_correct": [0, 1, 1], "user_prior_hinted": [0, 0, 1],
                       "user_n": [3, 3, 3], "user_correct": [2, 2, 2], "correct": [1, 0, 1]})
    honest = predict.student_features(df)
    df2 = df.assign(user_correct=0, correct=0)  # change the future, keep the past
    assert np.allclose(honest, predict.student_features(df2))
    assert not np.allclose(predict.student_features(df, leaky=True), predict.student_features(df2, leaky=True))


def _rasch_data(n_students=300, n_items=40, per=30, seed=0):
    rng = np.random.default_rng(seed)
    theta, b = rng.normal(size=n_students), rng.normal(size=n_items)
    rows, o = [], 0
    for u in range(n_students):
        for _ in range(per):
            i = int(rng.integers(n_items))
            o += 1
            rows.append({"order_id": o, "user_id": u, "problem_id": i, "skill": str(i % 4),
                         "correct": int(rng.random() < 1 / (1 + np.exp(-(theta[u] - b[i]))))})
    return pd.DataFrame(rows)


def test_elo_recovers_signal_on_rasch_data():
    df = _rasch_data()
    tr, te = df[df.user_id < 200], df[df.user_id >= 200]
    p = predict.elo_fit_predict(tr, te)
    assert roc_auc_score(te.correct, p) > 0.70


def test_pfa_learns_from_prior_successes():
    rng = np.random.default_rng(1)
    n = 5000
    s, f = rng.integers(0, 10, n), rng.integers(0, 10, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * np.log1p(s) - 0.8 * np.log1p(f))))).astype(int)
    df = pd.DataFrame({"skill": rng.choice(["a", "b"], n), "skill_prior_success": s, "skill_prior_failure": f})
    m = predict.PFA().fit(df, y)
    assert roc_auc_score(y, m.predict_proba(df)) > 0.65
