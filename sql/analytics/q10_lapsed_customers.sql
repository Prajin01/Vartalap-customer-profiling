-- Q10 · Lapsed and at-risk customers
-- Question: As of the last data date, which customers have gone quiet relative to their OWN usual buying rhythm, and how valuable are they?
-- Techniques: LAG() for inter-order gaps, median(), CROSS JOIN reference date, INNER + LEFT JOIN, CASE WHEN status rules, GREATEST(), NULLIF
-- Notes: descriptive rule, not the ML label. 'lapsed'  = days since last order > max(3 x own median gap, 90)
--                                            'at_risk' = days since last order > max(1.5 x own median gap, 45)
--        Customers with fewer than 3 orders have no reliable rhythm -> 'insufficient_history'.
--        Business use: the lapsed/at-risk list ranked by value is a re-engagement target list.
WITH ref AS (
    SELECT MAX(invoice_date) AS ref_date FROM core.fact_line
),
orders AS (
    SELECT customer_id,
           invoice_date,
           date_diff('day',
                     LAG(invoice_date) OVER (PARTITION BY customer_id ORDER BY invoice_ts, invoice_no),
                     invoice_date) AS gap_days
    FROM core.fact_invoice
    WHERE customer_id IS NOT NULL
      AND NOT is_cancel
),
rhythm AS (
    SELECT customer_id,
           COUNT(*)          AS n_orders,
           MAX(invoice_date) AS last_order_date,
           median(gap_days)  AS median_gap_days
    FROM orders
    GROUP BY customer_id
),
lifetime AS (
    SELECT customer_id, SUM(line_amount) AS lifetime_net_revenue
    FROM core.fact_line
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
),
classified AS (
    SELECT r.customer_id,
           d.country,
           r.n_orders,
           r.last_order_date,
           date_diff('day', r.last_order_date, ref.ref_date)                    AS days_since_last,
           r.median_gap_days,
           ROUND(date_diff('day', r.last_order_date, ref.ref_date)
                 / NULLIF(r.median_gap_days, 0), 1)                            AS overdue_ratio,
           ROUND(l.lifetime_net_revenue, 2)                                     AS lifetime_net_revenue,
           CASE
               WHEN r.n_orders < 3 THEN 'insufficient_history'
               WHEN date_diff('day', r.last_order_date, ref.ref_date) > GREATEST(3.0 * r.median_gap_days, 90) THEN 'lapsed'
               WHEN date_diff('day', r.last_order_date, ref.ref_date) > GREATEST(1.5 * r.median_gap_days, 45) THEN 'at_risk'
               ELSE 'active'
           END                                                                  AS status
    FROM rhythm AS r
    CROSS JOIN ref
    INNER JOIN core.dim_customer AS d
            ON d.customer_id = r.customer_id
    LEFT JOIN lifetime AS l
           ON l.customer_id = r.customer_id
)
SELECT *
FROM classified
ORDER BY CASE status WHEN 'lapsed' THEN 1 WHEN 'at_risk' THEN 2 WHEN 'active' THEN 3 ELSE 4 END,
         lifetime_net_revenue DESC;
