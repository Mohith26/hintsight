# Where this data came from, and what it leaves out

## Source

ASSISTments 2009-2010 skill-builder data, corrected version (one row per student-problem), published by the ASSISTments team at Worcester Polytechnic Institute:
https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010

`scripts/download_data.sh` fetches a byte-identical mirror (SHA-256 checked) from Hugging Face. The raw file is 346,860 rows; 275,458 are main problems (the rest are scaffolding sub-questions) from 4,217 students in 250 classes at 75 schools.

How it was collected: middle-school teachers assigned "skill builder" problem sets as classwork or homework. A student keeps getting problems on one skill until they answer three in a row correctly on the first try (there is also a daily cap). `correct` is first-attempt correctness, and **any hint request marks the problem wrong**.

## What it leaves out, and what that does to the conclusions

| Missing | Consequence for this analysis |
|---|---|
| Timestamps (the corrected file has none) | Order comes from `order_id`. "Pre" and "post" in the experiment section are halves of each student's own work, not calendar periods, so the 37% variance reduction is an estimate of what pre-launch data would give, not a measurement of it. |
| Why a student asked for a hint, and what they did with it (action-level logs) | Part 1 can only compare students who asked with students who did not. Even within the same student and problem set, the placebo shows hint-takers were already doing worse one problem earlier, so momentary struggle still selects who takes hints. The honest result is "no detectable benefit on the next problem", not "hints do not work". |
| Random assignment of hints | Every Part 1 estimate is observational. Only an experiment that randomizes hint availability (Part 3 sizes one) can give a causal answer. |
| Learning beyond the next problem | The outcome is the very next first attempt. Hints could help retention days later; this data cannot see that. |
| Who the students are (age, grade, prior math level, demographics) | Results are for 2009-10 Massachusetts-area middle-school classwork. AoPS students are self-selected, more advanced, and mostly not doing teacher-assigned work, so effect sizes and variances may not transfer. |
| Students who stopped | Mastery sets end after three correct in a row, and students who quit leave no outcome. Pairs exist only when a next problem was served, so the data over-represents students who were still struggling. |
| Skill tags for 18% of rows | Untagged rows use the problem set as their skill. |
| Teacher choice of problem sets | Classes worked on different sets, which is part of why accuracy is so correlated within a class (ICC 0.40). |
| Anything about what the hints said | Hint quality varies by problem; the estimate averages over all of it. |

## Known data quirks handled in code

* The file is not valid UTF-8 (Windows-1252 bytes inside `answer_text`); it is read as Latin-1 and `answer_text` is dropped.
* The original release duplicated rows for multi-skill problems; this project uses the corrected release, which collapses them (`skill_id` like `1_13`).
* Rows whose first action was "break this problem into steps" without a hint are neither hint-takers nor plain wrong answers and are left out of the Part 1 comparison.
