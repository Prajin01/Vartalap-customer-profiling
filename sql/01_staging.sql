-- =====================================================================
-- 01_staging.sql
-- Purpose : Type every raw column and FLAG problems. No rows are dropped
--           here, so data quality is measurable and every exclusion is
--           visible in one column: exclusion_reason.
-- Input   : raw.transactions  (loaded from Parquet by build_warehouse.py)
-- Output  : staging.lines     (1 row per raw row)
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS staging;

CREATE OR REPLACE TABLE staging.lines AS
WITH typed AS (
    SELECT
        raw_row_id,
        source_sheet,
        UPPER(TRIM(CAST(invoice AS VARCHAR)))            AS invoice_no,
        UPPER(TRIM(CAST(stock_code AS VARCHAR)))         AS stock_code,
        NULLIF(UPPER(TRIM(CAST(description AS VARCHAR))), '') AS description,
        CAST(quantity AS INTEGER)                        AS quantity,
        CAST(invoice_date AS TIMESTAMP)                  AS invoice_ts,
        CAST(price AS DECIMAL(18, 3))                    AS unit_price,
        CAST(customer_id AS BIGINT)                      AS customer_id,
        NULLIF(TRIM(CAST(country AS VARCHAR)), '')       AS country
    FROM raw.transactions
),
flagged AS (
    SELECT
        *,
        CAST(quantity * unit_price AS DECIMAL(18, 3))    AS line_amount,
        invoice_no LIKE 'C%'                             AS is_cancel,
        invoice_no LIKE 'A%'                             AS is_adjustment,
        -- Real products: 5 digits + optional 1-2 letter variant (e.g. 85123A).
        -- Everything else (POST, M, DOT, BANK CHARGES, gift vouchers...) is a
        -- service/admin code. The DQ report lists them for review.
        regexp_matches(stock_code, '^[0-9]{5}[A-Z]{0,2}$') AS is_product,
        customer_id IS NOT NULL                          AS has_customer,
        -- Exact duplicate = identical on every business column. Catches the
        -- overlap between the two workbook sheets AND repeated system rows.
        (ROW_NUMBER() OVER w) > 1                        AS is_exact_dup
    FROM typed
    WINDOW w AS (
        PARTITION BY invoice_no, stock_code, description, quantity,
                     invoice_ts, unit_price, customer_id, country
        ORDER BY raw_row_id
    )
)
SELECT
    *,
    CASE
        WHEN is_exact_dup                                THEN 'exact_duplicate'
        WHEN NOT is_product                              THEN 'non_product_code'
        WHEN is_adjustment                               THEN 'adjustment_invoice'
        WHEN unit_price IS NULL OR unit_price <= 0       THEN 'non_positive_price'
        WHEN NOT is_cancel AND quantity <= 0             THEN 'non_positive_qty_sale'
        WHEN is_cancel AND quantity >= 0                 THEN 'non_negative_qty_cancel'
        ELSE 'included'
    END                                                  AS exclusion_reason,
    (
        NOT is_exact_dup AND is_product AND NOT is_adjustment
        AND unit_price > 0
        AND ((NOT is_cancel AND quantity > 0) OR (is_cancel AND quantity < 0))
    )                                                    AS include_in_core
FROM flagged;
