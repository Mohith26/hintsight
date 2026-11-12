-- Builds the analysis tables from the raw ASSISTments skill-builder log.
-- Input table: raw (one row per student x problem, as shipped in skill_builder_data_corrected.csv)

-- 1. Main-problem attempts in order. order_id is the id of the original problem log
--    and increases with time, so it orders a student's work.
CREATE OR REPLACE TABLE attempts AS
SELECT
    order_id,
    user_id,
    assignment_id,
    problem_id,
    student_class_id AS class_id,
    teacher_id,
    school_id,
    -- 18% of rows have no skill tag; fall back to the problem set (one skill per skill builder)
    COALESCE(CAST(skill_id AS VARCHAR), 'seq_' || CAST(sequence_id AS VARCHAR)) AS skill,
    CAST(correct AS INTEGER) AS correct,
    CAST(hint_count > 0 AS INTEGER) AS hinted,
    COALESCE(CAST(bottom_hint AS INTEGER), 0) AS bottom_hint,
    CAST(first_action AS INTEGER) AS first_action,
    hint_count,
    attempt_count,
    ms_first_response
FROM raw
WHERE original = 1;

-- 2. Running history features, using only rows strictly before the current one.
CREATE OR REPLACE TABLE attempts_hist AS
SELECT
    a.*,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_id) AS seq_in_user,
    COUNT(*) OVER w_user_prior AS user_prior_n,
    COALESCE(SUM(correct) OVER w_user_prior, 0) AS user_prior_correct,
    COALESCE(SUM(hinted) OVER w_user_prior, 0) AS user_prior_hinted,
    COALESCE(SUM(correct) OVER w_skill_prior, 0) AS skill_prior_success,
    COUNT(*) OVER w_skill_prior - COALESCE(SUM(correct) OVER w_skill_prior, 0) AS skill_prior_failure,
    ROW_NUMBER() OVER (PARTITION BY user_id, assignment_id ORDER BY order_id) AS pos_in_assignment
FROM attempts a
WINDOW
    w_user_prior AS (PARTITION BY user_id ORDER BY order_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    w_skill_prior AS (PARTITION BY user_id, skill ORDER BY order_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);

-- 3. Student accuracy before the current assignment started (ability proxy for Part 1).
CREATE OR REPLACE TABLE assignment_start AS
SELECT user_id, assignment_id, MIN(order_id) AS start_order
FROM attempts GROUP BY user_id, assignment_id;

CREATE OR REPLACE TABLE pre_assignment_ability AS
SELECT s.user_id, s.assignment_id,
       COUNT(a.order_id) AS pre_n,
       COALESCE(SUM(a.correct), 0) AS pre_correct
FROM assignment_start s
LEFT JOIN attempts a
  ON a.user_id = s.user_id AND a.order_id < s.start_order
GROUP BY s.user_id, s.assignment_id;

-- 4. Consecutive pairs (t, t+1) inside the same student x assignment (Part 1).
CREATE OR REPLACE TABLE pairs AS
WITH ordered AS (
    SELECT h.*,
           LEAD(problem_id) OVER w AS next_problem_id,
           LEAD(correct) OVER w AS next_correct,
           LEAD(order_id) OVER w AS next_order_id,
           LAG(correct) OVER w AS prev_correct
    FROM attempts_hist h
    WINDOW w AS (PARTITION BY user_id, assignment_id ORDER BY order_id)
)
SELECT o.order_id, o.user_id, o.assignment_id, o.class_id, o.skill,
       o.problem_id, o.next_problem_id,
       o.correct, o.hinted, o.bottom_hint, o.first_action,
       o.next_correct, o.prev_correct,
       o.pos_in_assignment,
       o.skill_prior_success + o.skill_prior_failure AS opportunity,
       p.pre_n, p.pre_correct
FROM ordered o
JOIN pre_assignment_ability p USING (user_id, assignment_id)
WHERE o.next_problem_id IS NOT NULL;

-- 5. Per-student pre/post split for experiment design (Part 3).
CREATE OR REPLACE TABLE student_periods AS
WITH numbered AS (
    SELECT user_id, class_id, correct,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_id) AS k,
           COUNT(*) OVER (PARTITION BY user_id) AS n
    FROM attempts
),
main_class AS (  -- the class a student did most problems in (ties -> lowest id)
    SELECT user_id, class_id FROM (
        SELECT user_id, class_id,
               ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC, class_id) AS r
        FROM attempts GROUP BY user_id, class_id
    ) WHERE r = 1
)
SELECT n.user_id,
       ANY_VALUE(m.class_id) AS class_id,
       MAX(n) AS n_total,
       SUM(CASE WHEN k <= n / 2 THEN 1 ELSE 0 END) AS pre_n,
       SUM(CASE WHEN k <= n / 2 THEN correct ELSE 0 END) AS pre_correct,
       SUM(CASE WHEN k > n / 2 THEN 1 ELSE 0 END) AS post_n,
       SUM(CASE WHEN k > n / 2 THEN correct ELSE 0 END) AS post_correct
FROM numbered n JOIN main_class m USING (user_id)
GROUP BY n.user_id;
