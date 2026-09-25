-- Q03 · Weekly revenue and best weeks
-- Question: What does the weekly revenue curve look like, and which weeks were the strongest in each year?
-- Techniques: INNER JOIN to dim_date, weekly aggregation (ISO week start), LAG() week-over-week change, DENSE_RANK() OVER (PARTITION BY year)
WITH weekly AS (
    SELECT d.week_start,
           SUM(f.line_amount)                                           AS net_revenue,
           COUNT(DISTINCT f.invoice_no) FILTER (WHERE NOT f.is_cancel)  AS n_orders
    FROM core.fact_line AS f
    INNER JOIN core.dim_date AS d
            ON d.date_key = f.invoice_date
    GROUP BY d.week_start
)
SELECT week_start,
       year(week_start)                                                  AS year,
       ROUND(net_revenue, 2)                                             AS net_revenue,
       n_orders,
       ROUND(net_revenue - LAG(net_revenue) OVER (ORDER BY week_start), 2) AS wow_change,
       DENSE_RANK() OVER (PARTITION BY year(week_start)
                          ORDER BY net_revenue DESC)                     AS rank_in_year
FROM weekly
ORDER BY week_start;
