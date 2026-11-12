# Hintsight: spec

## Problem

An online math-practice platform logs every problem a student attempts, whether they asked for a hint, and whether they got it right. Curriculum teams ask three kinds of questions of that log:

1. **Does a feature work?** ("Do hints help students get the next problem right?") The naive comparison is biased because students choose to take hints, and the students who take hints are the ones who are struggling.
2. **What will this student do next?** An adaptive system (Alcumus-style) has to predict whether a student will get the next problem right so it can pick the next problem.
3. **How big does an experiment need to be?** Before running an A/B test on a practice feature, the team needs a minimum detectable effect computed on the real variance of student accuracy, and a way to shrink that variance.

## Why it matters to AoPS Data Science

AoPS runs Alcumus (adaptive problem practice), online courses and Beast Academy. The internship posting asks for SQL + Python, hypothesis testing and regression, models evaluated with attention to their assumptions, experiment analysis, separating real effects from noise and bias, data sense, reproducible code, and plain-English findings. Each part of this project is one of those tasks done on real student data.

## Data (real, public)

ASSISTments 2009-2010 skill-builder data, corrected version (one row per student-problem). ASSISTments is a free middle-school math practice platform run out of WPI. Skill-builder problem sets are mastery-learning sets: a student keeps getting problems on one skill until three in a row are correct.

* Source: https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010
* Mirror used for the download script: Hugging Face `OloriBern/assistments-dataset`, file `skill_builder_data_corrected.csv` (64.4 MB, SHA-256 `162ef8d2...e5da`, checked by the loader).
* Loaded into DuckDB; all feature work (next-problem pairing, running prior counts, item statistics) is done in SQL.

Nothing in the headline results is simulated. Part 3 uses simulation in one specific, labeled way: A/A and A/B tests are drawn by randomly re-splitting the real students, and the A/B "treatment effect" is planted on top of real data to check power.

## Design

### Part 1: do hints help on the next problem? (causal-style)

Unit: a pair of consecutive main problems (t, t+1) by the same student inside the same skill-builder assignment. Treatment: the student requested at least one hint on problem t. Outcome: first-attempt correctness on t+1.

Estimate ladder (each row is a regression with student-clustered standard errors):

1. Naive: hint vs no hint on t, all pairs.
2. Like-for-like: only pairs where problem t was answered wrong (ASSISTments scores any hint use as wrong, so "no hint" otherwise includes all correct answers).
3. + ability adjustment: student prior accuracy before this assignment, opportunity count, skill fixed effects.
4. Student fixed effects (each student compared with themselves).
5. Student x assignment fixed effects + next-problem fixed effects + position (same student, same skill set, same next problem).

Reported: each estimate with a 95% CI and how far the naive estimate was from the most controlled one. A null is reported as a null.

### Part 2: predict next-answer correctness (adaptive system)

Every main-problem attempt, in order. Models:

* Baseline A: majority/base rate.
* Baseline B: item difficulty only (smoothed accuracy of the problem, computed on training students only).
* Elo / online Rasch: student ability and item difficulty updated after each answer; item difficulty learned on training students, then frozen.
* PFA (performance factors analysis): logistic regression on skill + per-skill prior successes and failures.
* PFA + item difficulty + student prior history.

Split: 5-fold GroupKFold by student, so no student appears in both train and test. All student features use only that student's past rows. Metrics: AUC, log loss, Brier, expected calibration error (ECE, 10 bins). A deliberately leaky variant (student accuracy computed over all of their rows, including the future) is reported to show how much leakage inflates AUC.

### Part 3: experiment design on real variance

Unit: student (students with at least 20 main problems). Pre-period = the first half of their problems, post-period = second half. Outcome = post-period accuracy.

* Analytic minimum detectable effect (alpha 0.05 two-sided, power 0.8) at the available sample and the sample needed for a 2-point lift.
* CUPED: adjust the outcome with pre-period accuracy; variance reduction = 1 - rho^2.
* Validation: 2,000 simulated A/A tests (random re-splits of real students) to check false-positive rate is about 5% for both estimators; then simulated A/B tests with a planted lift to compare empirical power with the analytic prediction.
* Class-level design effect (ICC by class) for the case where the team must randomize whole classes.

## The 3 headline metrics

1. **Bias of the naive hint estimate**: naive estimate minus within-student estimate, in percentage points, with 95% CIs on both.
2. **Grouped-split AUC and ECE** of the best model vs the item-difficulty-only baseline (and the base rate).
3. **CUPED variance reduction and sample-size saving**, with the A/A false-positive rate and empirical power that validate it.

## Baselines

Naive difference (Part 1); base rate and item-difficulty-only model (Part 2); unadjusted difference-in-means t-test (Part 3).
