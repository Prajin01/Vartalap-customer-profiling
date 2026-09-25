-- Q06 · Repeat-purchase rate
-- Question: What share of new customers place a second order (at all, and within 90 days), by market and acquisition quarter?
-- Techniques: ROW_NUMBER() to sequence orders, LEFT JOIN (NULL = never repeated), CASE WHEN market, date_diff, FILTER, HAVING
-- Notes: Dec-2009 is excluded: its "first" orders include customers who existed before the data starts (left-censoring).
--        Later quarters had less time to repeat (right-censoring), so compare repeat_within_90d_pct, not repeat_rate_pct,
--        and treat the last quarter with caution.
WITH orders AS (
    SELECT customer_id,
           invoice_no,
           invoice_date,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY invoice_ts, invoice_no) AS order_seq
    FROM core.fact_invoice
    WHERE customer_id IS NOT NULL
      AND NOT is_cancel
),
first_order  AS (SELECT customer_id, invoice_date AS first_date  FROM orders WHERE order_seq = 1),
second_order AS (SELECT customer_id, invoice_date AS second_date FROM orders WHERE order_seq = 2),
customers AS (
    SELECT fo.customer_id,
           CASE WHEN d.country = 'United Kingdom' THEN 'UK' ELSE 'International' END      AS market,
           CAST(year(fo.first_date) AS VARCHAR) || '-Q' || CAST(quarter(fo.first_date) AS VARCHAR) AS first_quarter,
           so.second_date,
           date_diff('day', fo.first_date, so.second_date)                                 AS days_to_second
    FROM first_order AS fo
    INNER JOIN core.dim_customer AS d
            ON d.customer_id = fo.customer_id
    LEFT JOIN second_order AS so
           ON so.customer_id = fo.customer_id
)
SELECT market,
       first_quarter,
       COUNT(*)                                                                   AS new_customers,
       COUNT(second_date)                                                         AS repeated,
       ROUND(100.0 * COUNT(second_date) / COUNT(*), 1)                            AS repeat_rate_pct,
       ROUND(100.0 * COUNT(*) FILTER (WHERE days_to_second <= 90) / COUNT(*), 1)  AS repeat_within_90d_pct,
       median(days_to_second)                                                     AS median_days_to_second
FROM customers
WHERE first_quarter <> '2009-Q4'
GROUP BY market, first_quarter
HAVING COUNT(*) >= 30
ORDER BY market, first_quarter;
