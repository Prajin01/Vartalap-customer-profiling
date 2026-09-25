-- =====================================================================
-- core_summary.sql - post-build checks appended to the DQ report.
-- =====================================================================

-- name: core_row_counts
-- title: Core table row counts
SELECT 'fact_line'    AS table_name, COUNT(*) AS n_rows FROM core.fact_line
UNION ALL SELECT 'fact_invoice', COUNT(*) FROM core.fact_invoice
UNION ALL SELECT 'dim_customer', COUNT(*) FROM core.dim_customer
UNION ALL SELECT 'dim_product',  COUNT(*) FROM core.dim_product
UNION ALL SELECT 'dim_date',     COUNT(*) FROM core.dim_date;

-- name: revenue_reconciliation
-- title: Revenue reconciliation staging -> fact_line -> fact_invoice (must match)
SELECT (SELECT ROUND(SUM(line_amount), 2)    FROM staging.lines WHERE include_in_core) AS staging_amount,
       (SELECT ROUND(SUM(line_amount), 2)    FROM core.fact_line)                      AS fact_line_amount,
       (SELECT ROUND(SUM(invoice_amount), 2) FROM core.fact_invoice)                   AS fact_invoice_amount;

-- name: product_group_coverage
-- title: Product-group taxonomy coverage (sales lines only)
SELECT p.product_group,
       COUNT(DISTINCT p.stock_code)                                              AS n_products,
       COUNT(f.line_id)                                                          AS n_sale_lines,
       ROUND(100.0 * COUNT(f.line_id) / SUM(COUNT(f.line_id)) OVER (), 1)        AS pct_lines,
       ROUND(100.0 * SUM(f.line_amount) / SUM(SUM(f.line_amount)) OVER (), 1)    AS pct_revenue
FROM core.dim_product p
LEFT JOIN core.fact_line f
       ON f.stock_code = p.stock_code
      AND NOT f.is_cancel
GROUP BY p.product_group
ORDER BY pct_revenue DESC;
