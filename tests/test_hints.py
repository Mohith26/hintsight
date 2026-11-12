"""On data with a known hint effect and built-in selection bias, the naive comparison must be
badly wrong and the within-student estimate must recover the truth."""
import numpy as np
import pandas as pd

from hintsight.hints import estimate_ladder, like_for_like


def simulate_pairs(true_effect=0.05, n_students=1500, per_student=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_students):
        ability = rng.normal()
        for k in range(per_student):
            a = u * 10 + k // 10
            # weak students miss more and, when they miss, take hints more often
            p_right = 1 / (1 + np.exp(-(ability + 0.3)))
            correct = int(rng.random() < p_right)
            hinted = 0 if correct else int(rng.random() < 1 / (1 + np.exp(2 * ability)))
            p_next = np.clip(1 / (1 + np.exp(-ability)) + true_effect * hinted, 0, 1)
            rows.append({
                "order_id": u * 1000 + k, "user_id": u, "assignment_id": a, "class_id": u % 20, "skill": str(k // 10),
                "problem_id": k, "next_problem_id": k + 1, "correct": correct, "hinted": hinted,
                "bottom_hint": int(hinted and rng.random() < 0.5), "first_action": int(hinted),
                "next_correct": int(rng.random() < p_next), "prev_correct": int(rng.random() < p_right) if k % 10 else np.nan,
                "pos_in_assignment": k % 10 + 1, "opportunity": k % 10, "pre_n": k, "pre_correct": int(k * p_right),
            })
    return pd.DataFrame(rows)


def test_naive_is_biased_and_within_student_recovers_truth():
    pairs = simulate_pairs(true_effect=0.05)
    r = estimate_ladder(pairs)
    naive = r["1_naive_all_pairs"]["estimate"]
    within = r["4_student_fe"]["estimate"]
    assert naive < -0.05  # selection makes a helpful hint look harmful
    assert abs(within - 0.05) < 0.02
    lo, hi = r["4_student_fe"]["ci95"]
    assert lo < 0.05 < hi


def test_null_effect_is_reported_as_null():
    r = estimate_ladder(simulate_pairs(true_effect=0.0, seed=5))
    lo, hi = r["4_student_fe"]["ci95"]
    assert lo < 0 < hi


def test_like_for_like_drops_correct_and_scaffold_rows():
    df = pd.DataFrame({"correct": [1, 0, 0, 0], "hinted": [0, 1, 0, 0], "first_action": [0, 1, 0, 2]})
    out = like_for_like(df)
    assert list(out.index) == [1, 2]
