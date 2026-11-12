# Findings memo: hints, predicting the next answer, and sizing experiments

**For:** curriculum and product team. **Data:** 275,458 first attempts by 4,217 middle-school students on a public math-practice platform (ASSISTments, 2009-10). Every number below comes from `results/results.json`.

## 1. Do hints help students get the next problem right? Not that we can detect.

* **The raw numbers look alarming.** Right after taking a hint, students got the next problem right 39% of the time, versus 72% after not taking one: a 33-point gap.
* **Most of that gap is about who takes hints, not what hints do.** Students who took hints were already weaker (58% accuracy before the problem set started, versus 66% for students who answered wrong without a hint). Comparing each student only with themselves, inside the same problem set and on the same next problem, shrinks the gap to 8.6 points. About three quarters of the raw gap was selection.
* **The rest looks like selection too.** A hint on a problem cannot change the answer to the problem *before* it, yet hint-takers were already 6.1 points worse on that earlier problem. Net of that pre-existing dip, the hint "effect" is -1.6 points, with a 95% range of -3.7 to +0.6. That is statistically indistinguishable from zero.
* **Answer-revealing hints** (the last hint shows the answer) go with worse next-problem results than partial hints (-10.3 vs -5.9 points under the same comparison). That fits students clicking through to the answer when they have stopped trying, but it is the same kind of observational comparison, so treat it as a lead, not a finding.

**Recommendation:** do not cut or expand hints based on log comparisons like the raw one; they mostly measure which students struggle. If hint design matters, run an experiment (see 3). A useful first test: delay the answer-revealing hint.

## 2. Can we predict whether a student will get the next problem right? Yes, well enough to drive problem selection.

* Scored on students the model never saw, the best model (a skill-practice model plus problem difficulty and the student's own track record) ranks right versus wrong answers correctly 77% of the time (AUC 0.770). Knowing only how hard each problem is gets 68% (0.681). A guess at the overall average gets 50%.
* Its probabilities are honest: when it says 70%, students are right about 70% of the time (average calibration error 0.7 points).
* Two traps that make models look better than they are, both measured here: (a) estimating problem difficulty from the same answers the model trains on made the probabilities five times less trustworthy (calibration error 3.8 points instead of 0.7); (b) letting the model peek at a student's later answers raised the score to 0.778, a gain that would vanish in production.

## 3. How big does an A/B test on a practice feature need to be?

* Student accuracy varies a lot (standard deviation 22 points), so detecting a **2-point** lift with the usual standards (80% power, 5% false-positive rate) needs **1,959 students per group**.
* Adjusting each student's result for how they were doing before the test (CUPED) removes 37% of that noise and cuts the requirement to **1,231 per group**, 37% fewer students for the same answer.
* We checked this on the real students rather than trusting formulas: across 2,000 fake experiments where nothing changed, the method raised a false alarm 4.75% of the time (target 5%). With a 2.2-point improvement planted in the data, it caught it 80.8% of the time versus 60.9% without the adjustment, matching the formulas (80.1% and 60.3%).
* **If whole classes must be randomized together, plan for about 7 times more students.** Students in the same class score very alike (intraclass correlation 0.40), so a 2-point test would need about 14,400 students per group. Randomize individual students whenever the product allows it.

## What this data cannot tell us

These are 2009-10 middle-school students doing teacher-assigned work, not self-selected AoPS students; there are no timestamps; and we only see the very next problem, not learning a week later. See `DATA.md` for the full list and how each gap limits the conclusions above.
