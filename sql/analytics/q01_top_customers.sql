-- Q01 · Top customers
-- Question: Which 10 customers generate the most net revenue, and how concentrated is revenue among them?
-- Techniques: CTE, INNER JOIN, GROUP BY, FILTER aggregate, RANK(), SUM() OVER () share, running SUM() OVER (ORDER BY ...)
-- Note: net revenue = sales minus cancellations. Only identified customers.
WITH customer_revenue AS (
    SELECT f.customer_id,
           SUM(f.line_amount)                                           AS net_revenue,
           COUNT(DISTINCT f.invoice_no) FILTER (WHERE NOT f.is_cancel)  AS n_orders
    FROM core.fact_line AS f
    WHERE f.customer_id IS NOT NULL
    GROUP BY f.customer_id
),
ranked AS (
    SELECT cr.customer_id,
           d.country,
           cr.n_orders,
           ROUND(cr.net_revenue, 2)                                      AS net_revenue,
           RANK() OVER (ORDER BY cr.net_revenue DESC)                    AS revenue_rank,
           ROUND(100.0 * cr.net_revenue / SUM(cr.net_revenue) OVER (), 2) AS pct_of_customer_revenue,
           ROUND(100.0 * SUM(cr.net_revenue) OVER (
                     ORDER BY cr.net_revenue DESC, cr.customer_id
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                 / SUM(cr.net_revenue) OVER (), 2)                      AS cumulative_pct
    FROM customer_revenue AS cr
    INNER JOIN core.dim_customer AS d
            ON d.customer_id = cr.customer_id
)
SELECT *
FROM ranked
WHERE revenue_rank <= 10
ORDER BY revenue_rank;
