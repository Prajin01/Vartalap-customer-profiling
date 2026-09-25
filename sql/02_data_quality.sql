-- =====================================================================
-- 02_data_quality.sql
-- Named queries rendered into reports/data_quality.md by build_warehouse.py.
-- Each block: `-- name:` id, `-- title:` heading, then ONE SELECT.
-- =====================================================================

-- name: row_counts
-- title: Rows and date range per source sheet
SELECT source_sheet,
       COUNT(*)          AS n_rows,
       MIN(invoice_date) AS first_ts,
       MAX(invoice_date) AS last_ts
FROM raw.transactions
GROUP BY source_sheet
UNION ALL
SELECT 'TOTAL', COUNT(*), MIN(invoice_date), MAX(invoice_date)
FROM raw.transactions
ORDER BY source_sheet;

-- name: null_profile
-- title: Missing values per raw column
SELECT COUNT(*)                       AS n_rows,
       COUNT(*) - COUNT(invoice)      AS null_invoice,
       COUNT(*) - COUNT(stock_code)   AS null_stock_code,
       COUNT(*) - COUNT(description)  AS null_description,
       COUNT(*) - COUNT(quantity)     AS null_quantity,
       COUNT(*) - COUNT(invoice_date) AS null_invoice_date,
       COUNT(*) - COUNT(price)        AS null_price,
       COUNT(*) - COUNT(customer_id)  AS null_customer_id,
       ROUND(100.0 * (COUNT(*) - COUNT(customer_id)) / COUNT(*), 2) AS pct_null_customer_id
FROM raw.transactions;

-- name: missing_customer_impact
-- title: Lines and sales value with vs without a customer ID (sales = non-cancelled, qty>0, price>0)
WITH s AS (
    SELECT has_customer,
           COUNT(*) AS n_lines,
           SUM(line_amount) FILTER (WHERE NOT is_cancel AND quantity > 0 AND unit_price > 0) AS sales_amount
    FROM staging.lines
    GROUP BY has_customer
)
SELECT has_customer,
       n_lines,
       ROUND(100.0 * n_lines / SUM(n_lines) OVER (), 2)           AS pct_lines,
       ROUND(sales_amount, 2)                                     AS sales_amount,
       ROUND(100.0 * sales_amount / SUM(sales_amount) OVER (), 2) AS pct_sales_amount
FROM s
ORDER BY has_customer DESC;

-- name: invoice_types
-- title: Invoice types by prefix
SELECT CASE
           WHEN is_cancel                             THEN 'cancellation (C-prefix)'
           WHEN is_adjustment                         THEN 'adjustment (A-prefix)'
           WHEN regexp_matches(invoice_no, '^[0-9]+$') THEN 'regular'
           ELSE 'other'
       END                          AS invoice_type,
       COUNT(*)                     AS n_lines,
       COUNT(DISTINCT invoice_no)   AS n_invoices,
       ROUND(SUM(line_amount), 2)   AS total_amount
FROM staging.lines
GROUP BY 1
ORDER BY n_lines DESC;

-- name: quantity_price_anomalies
-- title: Quantity and price anomalies
SELECT COUNT(*) FILTER (WHERE NOT is_cancel AND quantity <= 0)                     AS sale_lines_qty_le_0,
       COUNT(*) FILTER (WHERE NOT is_cancel AND quantity <= 0 AND NOT has_customer) AS of_which_without_customer,
       COUNT(*) FILTER (WHERE is_cancel AND quantity >= 0)                         AS cancel_lines_qty_ge_0,
       COUNT(*) FILTER (WHERE unit_price = 0)                                      AS price_eq_0,
       COUNT(*) FILTER (WHERE unit_price < 0)                                      AS price_lt_0
FROM staging.lines;

-- name: exact_duplicates
-- title: Exact duplicate lines (cross-sheet overlap vs within-sheet)
WITH d AS (
    SELECT *,
           FIRST_VALUE(source_sheet) OVER (
               PARTITION BY invoice_no, stock_code, description, quantity,
                            invoice_ts, unit_price, customer_id, country
               ORDER BY raw_row_id
           ) AS first_sheet
    FROM staging.lines
)
SELECT CASE WHEN first_sheet <> source_sheet THEN 'cross_sheet_overlap'
            ELSE 'within_sheet' END AS duplicate_type,
       COUNT(*)                     AS n_duplicate_lines,
       COUNT(DISTINCT invoice_no)   AS n_invoices_affected
FROM d
WHERE is_exact_dup
GROUP BY 1;

-- name: non_product_codes
-- title: Non-product stock codes (top 40 by line count) - review these
SELECT stock_code,
       ANY_VALUE(description)     AS example_description,
       COUNT(*)                   AS n_lines,
       ROUND(SUM(line_amount), 2) AS total_amount
FROM staging.lines
WHERE NOT is_product
GROUP BY stock_code
ORDER BY n_lines DESC
LIMIT 40;

-- name: integrity_checks
-- title: Referential / consistency checks (0 is ideal for the first and last)
SELECT
    (SELECT COUNT(*) FROM (SELECT invoice_no FROM staging.lines GROUP BY invoice_no
                           HAVING COUNT(DISTINCT COALESCE(customer_id, -1)) > 1)) AS invoices_with_multiple_customers,
    (SELECT COUNT(*) FROM (SELECT customer_id FROM staging.lines WHERE has_customer GROUP BY customer_id
                           HAVING COUNT(DISTINCT country) > 1))                    AS customers_with_multiple_countries,
    (SELECT COUNT(*) FROM (SELECT stock_code FROM staging.lines WHERE description IS NOT NULL GROUP BY stock_code
                           HAVING COUNT(DISTINCT description) > 1))                AS products_with_multiple_descriptions,
    (SELECT COUNT(*) FROM (SELECT invoice_no FROM staging.lines GROUP BY invoice_no
                           HAVING COUNT(DISTINCT invoice_ts) > 1))                 AS invoices_with_multiple_timestamps;

-- name: value_distributions
-- title: Distributions of included sale lines (min / p1 / median / p99 / max)
SELECT 'quantity' AS metric,
       CAST(MIN(quantity) AS DOUBLE) AS min_v,
       quantile_cont(quantity, 0.01) AS p01,
       quantile_cont(quantity, 0.50) AS p50,
       quantile_cont(quantity, 0.99) AS p99,
       CAST(MAX(quantity) AS DOUBLE) AS max_v
FROM staging.lines WHERE include_in_core AND NOT is_cancel
UNION ALL
SELECT 'unit_price', CAST(MIN(unit_price) AS DOUBLE),
       quantile_cont(CAST(unit_price AS DOUBLE), 0.01), quantile_cont(CAST(unit_price AS DOUBLE), 0.50),
       quantile_cont(CAST(unit_price AS DOUBLE), 0.99), CAST(MAX(unit_price) AS DOUBLE)
FROM staging.lines WHERE include_in_core AND NOT is_cancel
UNION ALL
SELECT 'line_amount', CAST(MIN(line_amount) AS DOUBLE),
       quantile_cont(CAST(line_amount AS DOUBLE), 0.01), quantile_cont(CAST(line_amount AS DOUBLE), 0.50),
       quantile_cont(CAST(line_amount AS DOUBLE), 0.99), CAST(MAX(line_amount) AS DOUBLE)
FROM staging.lines WHERE include_in_core AND NOT is_cancel;

-- name: cleaning_funnel
-- title: Cleaning funnel - why each raw line was kept or excluded
SELECT exclusion_reason,
       COUNT(*)                                          AS n_lines,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_lines
FROM staging.lines
GROUP BY exclusion_reason
ORDER BY n_lines DESC;

-- name: customer_overview
-- title: Repeat behaviour of identified customers (included, non-cancelled invoices)
WITH inv AS (
    SELECT customer_id, COUNT(DISTINCT invoice_no) AS n_invoices
    FROM staging.lines
    WHERE include_in_core AND has_customer AND NOT is_cancel
    GROUP BY customer_id
)
SELECT COUNT(*)                                                    AS customers_with_purchase,
       COUNT(*) FILTER (WHERE n_invoices >= 2)                     AS repeat_customers,
       ROUND(100.0 * COUNT(*) FILTER (WHERE n_invoices >= 2) / COUNT(*), 1) AS pct_repeat,
       median(n_invoices)                                          AS median_invoices,
       quantile_cont(n_invoices, 0.90)                             AS p90_invoices,
       MAX(n_invoices)                                             AS max_invoices
FROM inv;
