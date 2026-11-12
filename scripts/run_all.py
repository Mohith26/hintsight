"""Reproduce every number in results/results.json and every figure in figures/."""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hintsight import db, experiment, figures, hints, predict  # noqa: E402


def main(rebuild: bool = False) -> None:
    t0 = time.time()
    con = db.connect(rebuild=rebuild)
    data = con.execute("""
        SELECT COUNT(*) AS main_problem_attempts, COUNT(DISTINCT user_id) AS students,
               COUNT(DISTINCT problem_id) AS problems, COUNT(DISTINCT skill) AS skills,
               COUNT(DISTINCT class_id) AS classes, COUNT(DISTINCT school_id) AS schools,
               AVG(correct) AS share_correct_first_try, AVG(hinted) AS share_with_hint
        FROM attempts""").df().iloc[0].to_dict()
    raw_rows = con.execute("SELECT COUNT(*) FROM raw").fetchone()[0]

    print("part 1: hints")
    h = hints.estimate_ladder(hints.load_pairs(con))
    print("part 2: prediction")
    p = predict.evaluate(con)
    print("part 3: experiment design")
    e = experiment.analyze(con, n_sims=2000, seed=7)
    pvals = e.pop("_p_values")

    (ROOT / "figures").mkdir(exist_ok=True)
    figures.hint_ladder(h, ROOT / "figures" / "fig1_hint_effect_ladder.png")
    figures.calibration(p, ROOT / "figures" / "fig2_prediction_calibration.png")
    figures.power(e, pvals, ROOT / "figures" / "fig3_experiment_power.png")

    results = {
        "dataset": {
            "name": "ASSISTments 2009-2010 skill-builder (corrected, one row per student-problem)",
            "sha256": db.DATA_SHA256, "raw_rows": int(raw_rows),
            **{k: (float(v) if k.startswith("share") else int(v)) for k, v in data.items()},
        },
        "part1_hints": h,
        "part2_prediction": p,
        "part3_experiment_design": e,
        "run": {"seconds": round(time.time() - t0, 1), "python": platform.python_version()},
    }
    out = ROOT / "results" / "results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out} in {results['run']['seconds']}s")


if __name__ == "__main__":
    main(rebuild="--rebuild" in sys.argv)
