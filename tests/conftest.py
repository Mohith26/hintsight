import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hintsight import db  # noqa: E402


def make_raw(n_students=60, n_assignments=4, per_assignment=6, seed=0) -> pd.DataFrame:
    """Small synthetic log with the same columns as the real ASSISTments file."""
    rng = np.random.default_rng(seed)
    rows = []
    order = 1000
    ability = rng.normal(0, 1, n_students)
    for u in range(n_students):
        for a in range(n_assignments):
            skill = a + 1
            for k in range(per_assignment):
                order += int(rng.integers(1, 5))
                prob = 100 * skill + int(rng.integers(0, 10))
                p = 1 / (1 + np.exp(-(ability[u] - 0.2 * skill + 0.3 * k)))
                hinted = rng.random() < 0.3 * (1 - p)
                correct = 0 if hinted else int(rng.random() < p)
                rows.append({
                    "order_id": order, "assignment_id": 10 * u + a, "user_id": u, "assistment_id": prob,
                    "problem_id": prob, "original": 1, "correct": correct, "attempt_count": 1,
                    "ms_first_response": 1000, "tutor_mode": "tutor", "answer_type": "algebra",
                    "sequence_id": skill, "student_class_id": u % 5, "position": 1, "type": "MasterySection",
                    "base_sequence_id": skill, "skill_id": str(skill) if skill != 4 else None,
                    "skill_name": f"s{skill}", "teacher_id": 1, "school_id": 1,
                    "hint_count": int(hinted) * 2, "hint_total": 3, "overlap_time": 1000, "template_id": prob,
                    "first_action": 1 if hinted else 0, "bottom_hint": 1 if hinted else None,
                    "opportunity": k + 1, "opportunity_original": k + 1,
                })
            # one scaffolding row that is not a main problem
            rows.append(rows[-1] | {"order_id": order + 1, "original": 0})
            order += 2
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def raw():
    return make_raw()


@pytest.fixture(scope="session")
def con(raw):
    c = duckdb.connect()
    db.build_tables(c, raw)
    return c
