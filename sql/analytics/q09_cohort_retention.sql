-- Q09 · Monthly cohort retention
-- Question: Of the customers who first bought in a given month, what share buy again 1, 2, 3 ... months later?
-- Techniques: DISTINCT customer-month activity, MIN() cohort assignment, INNER JOIN, date_diff in months,
--             FIRST_VALUE() OVER (PARTITION BY cohort) as cohort size
-- Note: cohorts start Jan-2010. The Dec-2009 "cohort" would include customers who existed before the data (left-censoring).
WITH purchases AS (
    SELECT DISTINCT customer_id,
           CAST(DATE_TRUNC('month', invoice_date) AS DATE) AS activity_month
    FROM core.fact_line
    WHERE customer_id IS NOT NULL
      AND NOT is_cancel
),
cohorts AS (
    SELECT customer_id, MIN(activity_month) AS cohort_month
    FROM purchases
    GROUP BY customer_id
),
activity AS (
    SELECT c.cohort_month,
           date_diff('month', c.cohort_month, p.activity_month) AS months_since_first,
           COUNT(DISTINCT p.customer_id)                        AS active_customers
    FROM purchases AS p
    INNER JOIN cohorts AS c
            ON c.customer_id = p.customer_id
    GROUP BY 1, 2
)
SELECT cohort_month,
       months_since_first,
       active_customers,
       FIRST_VALUE(active_customers) OVER (PARTITION BY cohort_month ORDER BY months_since_first) AS cohort_size,
       ROUND(100.0 * active_customers
             / FIRST_VALUE(active_customers) OVER (PARTITION BY cohort_month ORDER BY months_since_first), 1) AS retention_pct
FROM activity
WHERE cohort_month >= DATE '2010-01-01'
ORDER BY cohort_month, months_since_first;
