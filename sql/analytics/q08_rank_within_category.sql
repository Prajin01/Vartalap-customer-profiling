-- Q08 · Top products within each category
-- Question: Which 3 products lead each product group by revenue, and how do their revenue, reach and unit ranks compare?
-- Techniques: INNER JOIN fact_line to dim_product, ROW_NUMBER() vs RANK() vs DENSE_RANK() OVER (PARTITION BY product_group),
--             share within group via SUM() OVER (PARTITION BY ...)
-- Notes: ROW_NUMBER never ties (unique 1,2,3), RANK skips after ties (1,1,3), DENSE_RANK does not skip (1,1,2).
--        product_group is the heuristic keyword taxonomy; 'other' and 'unknown' are excluded here.
WITH product_sales AS (
    SELECT p.product_group,
           p.stock_code,
           p.description,
           SUM(f.line_amount)            AS net_revenue,
           SUM(f.quantity)               AS net_units,
           COUNT(DISTINCT f.customer_id) AS n_customers
    FROM core.fact_line AS f
    INNER JOIN core.dim_product AS p
            ON p.stock_code = f.stock_code
    GROUP BY p.product_group, p.stock_code, p.description
),
ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY product_group ORDER BY net_revenue DESC, stock_code) AS revenue_rank,
           RANK()       OVER (PARTITION BY product_group ORDER BY n_customers DESC)             AS reach_rank,
           DENSE_RANK() OVER (PARTITION BY product_group ORDER BY net_units DESC)               AS units_dense_rank,
           ROUND(100.0 * net_revenue / SUM(net_revenue) OVER (PARTITION BY product_group), 2)   AS pct_of_group_revenue
    FROM product_sales
)
SELECT product_group,
       revenue_rank,
       reach_rank,
       units_dense_rank,
       stock_code,
       description,
       ROUND(net_revenue, 2) AS net_revenue,
       net_units,
       n_customers,
       pct_of_group_revenue
FROM ranked
WHERE revenue_rank <= 3
  AND product_group NOT IN ('other', 'unknown')
ORDER BY product_group, revenue_rank;
