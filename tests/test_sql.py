"""The SQL feature tables must never let a row see its own label or the future."""
import numpy as np


def test_main_problems_only(con, raw):
    n = con.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    assert n == int((raw.original == 1).sum())


def test_untagged_skill_falls_back_to_problem_set(con):
    skills = {r[0] for r in con.execute("SELECT DISTINCT skill FROM attempts").fetchall()}
    assert "seq_4" in skills and None not in skills


def test_prior_counts_exclude_current_row(con):
    df = con.execute("SELECT * FROM attempts_hist ORDER BY user_id, order_id").df()
    for _, g in df.groupby("user_id"):
        expected_n = np.arange(len(g))
        expected_c = np.concatenate([[0], np.cumsum(g["correct"].to_numpy())[:-1]])
        assert (g["user_prior_n"].to_numpy() == expected_n).all()
        assert (g["user_prior_correct"].to_numpy() == expected_c).all()
    # per-skill success + failure = number of earlier attempts on that skill
    for _, g in df.groupby(["user_id", "skill"]):
        tot = (g["skill_prior_success"] + g["skill_prior_failure"]).to_numpy()
        assert (tot == np.arange(len(g))).all()


def test_pairs_are_consecutive_within_assignment(con):
    p = con.execute("SELECT * FROM pairs").df()
    a = con.execute("SELECT * FROM attempts").df()
    # each assignment with k main problems yields k-1 pairs
    k = a.groupby(["user_id", "assignment_id"]).size()
    assert len(p) == int((k - 1).sum())
    look = a.set_index("order_id")
    nxt = a.sort_values("order_id").groupby(["user_id", "assignment_id"])["order_id"].shift(-1)
    nxt_map = dict(zip(a.sort_values("order_id")["order_id"], nxt))
    for _, r in p.sample(50, random_state=0).iterrows():
        nid = nxt_map[r.order_id]
        assert look.loc[nid, "problem_id"] == r.next_problem_id
        assert look.loc[nid, "correct"] == r.next_correct


def test_pre_assignment_ability_uses_only_earlier_assignments(con):
    df = con.execute("""
        SELECT p.user_id, p.assignment_id, p.pre_n, s.start_order,
               (SELECT COUNT(*) FROM attempts a WHERE a.user_id = p.user_id AND a.order_id < s.start_order) AS check_n
        FROM pre_assignment_ability p JOIN assignment_start s USING (user_id, assignment_id)""").df()
    assert (df.pre_n == df.check_n).all()
    assert (df.groupby("user_id").pre_n.min() == 0).all()


def test_student_periods_split_in_half(con):
    s = con.execute("SELECT * FROM student_periods").df()
    assert ((s.pre_n + s.post_n) == s.n_total).all()
    assert ((s.post_n - s.pre_n).isin([0, 1])).all()
