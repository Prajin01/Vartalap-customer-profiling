-- Q02 · Monthly revenue
-- Question: How do sales, cancellations, orders and active customers move month by month, and how fast is revenue growing?
-- Techniques: DATE_TRUNC monthly aggregation, FILTER aggregates, LAG() for MoM and YoY growth, AVG() OVER (ROWS 2 PRECEDING) moving average, scalar subquery
-- Note: Dec-2011 contains data only up to 9 Dec, so it is flagged as a partial month. Includes guest (no customer ID) revenue.
WITH monthly AS (
    SELECT DATE_TRUNC('month', invoice_date)                          AS month,
           SUM(line_amount) FILTER (WHERE NOT is_cancel)              AS gross_sales,
           -SUM(line_amount) FILTER (WHERE is_cancel)                 AS cancelled_value,
           SUM(line_amount)                                           AS net_revenue,
           COUNT(DISTINCT invoice_no) FILTER (WHERE NOT is_cancel)    AS n_orders,
           COUNT(DISTINCT customer_id) FILTER (WHERE NOT is_cancel)   AS active_customers
    FROM core.fact_line
    GROUP BY 1
)
SELECT CAST(month AS DATE)                                               AS month,
       ROUND(gross_sales, 2)                                             AS gross_sales,
       ROUND(COALESCE(cancelled_value, 0), 2)                            AS cancelled_value,
       ROUND(net_revenue, 2)                                             AS net_revenue,
       n_orders,
       active_customers,
       ROUND(gross_sales / NULLIF(n_orders, 0), 2)                       AS avg_order_value,
       ROUND(100.0 * (net_revenue - LAG(net_revenue) OVER (ORDER BY month))
             / NULLIF(LAG(net_revenue) OVER (ORDER BY month), 0), 1)     AS mom_growth_pct,
       ROUND(100.0 * (net_revenue - LAG(net_revenue, 12) OVER (ORDER BY month))
             / NULLIF(LAG(net_revenue, 12) OVER (ORDER BY month), 0), 1) AS yoy_growth_pct,
       ROUND(AVG(net_revenue) OVER (ORDER BY month
             ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2)               AS net_revenue_3m_avg,
       month = (SELECT MAX(DATE_TRUNC('month', invoice_date)) FROM core.fact_line) AS is_partial_month
FROM monthly
ORDER BY month;
