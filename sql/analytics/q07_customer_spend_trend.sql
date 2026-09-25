-- Q07 · Spend trend per customer
-- Question: For each customer, how does monthly spend evolve: gap since the previous active month, change vs that month,
--           the next active month, running lifetime spend and a rolling average?
-- Techniques: monthly aggregation, LAG() and LEAD() over a named WINDOW, date_diff in months,
--             running SUM() OVER (ROWS UNBOUNDED PRECEDING), rolling AVG() OVER (ROWS 2 PRECEDING), PARTITION BY customer
-- Note: one row per customer-month with activity. The notebook plots one customer from this table.
WITH customer_month AS (
    SELECT customer_id,
           CAST(DATE_TRUNC('month', invoice_date) AS DATE)             AS month,
           SUM(line_amount)                                            AS net_spend,
           COUNT(DISTINCT invoice_no) FILTER (WHERE NOT is_cancel)     AS n_orders
    FROM core.fact_line
    WHERE customer_id IS NOT NULL
    GROUP BY 1, 2
)
SELECT customer_id,
       month,
       ROUND(net_spend, 2)                                       AS net_spend,
       n_orders,
       LAG(month) OVER w                                         AS prev_active_month,
       date_diff('month', LAG(month) OVER w, month)              AS months_since_prev,
       ROUND(net_spend - LAG(net_spend) OVER w, 2)               AS change_vs_prev,
       LEAD(month) OVER w                                        AS next_active_month,
       ROUND(SUM(net_spend) OVER (PARTITION BY customer_id ORDER BY month
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) AS cumulative_spend,
       ROUND(AVG(net_spend) OVER (PARTITION BY customer_id ORDER BY month
             ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)       AS rolling_3_active_month_avg
FROM customer_month
WINDOW w AS (PARTITION BY customer_id ORDER BY month)
ORDER BY customer_id, month;
