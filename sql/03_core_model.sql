-- =====================================================================
-- 03_core_model.sql
-- Purpose : Star schema built from included staging lines.
--   fact_line     1 row per clean invoice line (sales AND cancellations)
--   fact_invoice  1 row per invoice
--   dim_customer  1 row per identified customer (full-history, DESCRIPTIVE)
--   dim_product   1 row per stock code, canonical description + product_group
--   dim_date      1 row per calendar day in the data range
--
-- LEAKAGE NOTE: dim_customer / dim_product summarise the FULL history. They
-- are for analytics and profiles only. Model features come exclusively from
-- the point-in-time macro customer_features(cutoff) (Step 4).
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS core;

-- ---------- fact_line ----------
CREATE OR REPLACE TABLE core.fact_line AS
SELECT raw_row_id                 AS line_id,
       invoice_no,
       customer_id,
       stock_code,
       invoice_ts,
       CAST(invoice_ts AS DATE)   AS invoice_date,
       quantity,
       unit_price,
       line_amount,
       is_cancel
FROM staging.lines
WHERE include_in_core;

-- ---------- fact_invoice ----------
CREATE OR REPLACE TABLE core.fact_invoice AS
SELECT invoice_no,
       MAX(customer_id)                 AS customer_id,   -- tests assert 1 customer per invoice
       MIN(invoice_ts)                  AS invoice_ts,
       CAST(MIN(invoice_ts) AS DATE)    AS invoice_date,
       BOOL_OR(is_cancel)               AS is_cancel,
       COUNT(*)                         AS n_lines,
       COUNT(DISTINCT stock_code)       AS n_products,
       SUM(quantity)                    AS total_quantity,
       SUM(line_amount)                 AS invoice_amount
FROM core.fact_line
GROUP BY invoice_no;

-- ---------- dim_product ----------
CREATE OR REPLACE TABLE core.dim_product AS
WITH desc_counts AS (
    SELECT stock_code, description, COUNT(*) AS n
    FROM staging.lines
    WHERE include_in_core AND description IS NOT NULL
    GROUP BY stock_code, description
),
canonical AS (
    SELECT stock_code,
           description,
           COUNT(*) OVER (PARTITION BY stock_code)                          AS n_descriptions,
           ROW_NUMBER() OVER (PARTITION BY stock_code ORDER BY n DESC, description) AS rn
    FROM desc_counts
),
products AS (
    SELECT DISTINCT stock_code FROM core.fact_line
)
SELECT p.stock_code,
       c.description,
       COALESCE(c.n_descriptions, 0) AS n_descriptions,
       -- Heuristic keyword taxonomy v1. First match wins, so order matters.
       -- Coverage is reported in reports/data_quality.md.
       CASE
           WHEN c.description IS NULL THEN 'unknown'
           WHEN regexp_matches(c.description, '\b(CHRISTMAS|XMAS|ADVENT|SANTA|REINDEER|SNOWFLAKE|NOEL)\b')
               THEN 'seasonal_christmas'
           WHEN regexp_matches(c.description, '\b(CANDLES?|T-LIGHTS?|TLIGHTS?|LANTERNS?|LIGHTS?|LAMPS?)\b')
               THEN 'candles_lighting'
           WHEN regexp_matches(c.description, '\b(MUGS?|CUPS?|PLATES?|BOWLS?|TEAPOT|TEACUP|CAKE|CAKESTAND|BAKING|JARS?|LUNCH BOX|TINS?|NAPKINS?|CUTLERY|SPOONS?|TRAY|COASTERS?|GLASS|BOTTLE|APRON|TEA TOWELS?|JUG|CAKE CASES)\b')
               THEN 'kitchen_dining'
           WHEN regexp_matches(c.description, '\b(BAGS?|STORAGE|BASKETS?|BOX|BOXES|HAMPER)\b')
               THEN 'bags_storage'
           WHEN regexp_matches(c.description, '\b(CARDS?|WRAP|PAPER|RIBBONS?|TAGS?|NOTEBOOK|PENCILS?|PENS?|STICKERS?|ENVELOPES?|CHALK|ERASERS?|STATIONERY)\b')
               THEN 'stationery_giftwrap'
           WHEN regexp_matches(c.description, '\b(TOYS?|GAMES?|DOLLY|DOLLS?|JIGSAW|CHILDRENS?|KIDS|PLAYHOUSE|SPACEBOY|DINOSAUR|PUZZLES?|CRAYONS?|SKIPPING ROPE)\b')
               THEN 'toys_kids'
           WHEN regexp_matches(c.description, '\b(NECKLACE|EARRINGS?|BRACELET|RING|HAIR|BROOCH|PURSE|WALLET|UMBRELLA)\b')
               THEN 'jewellery_accessories'
           WHEN regexp_matches(c.description, '\b(GARDEN|PARASOL|BIRD|PLANT|PLANTER|WATERING|PICNIC|HEN HOUSE)\b')
               THEN 'garden_outdoor'
           WHEN regexp_matches(c.description, '\b(SIGN|FRAMES?|CLOCK|MIRROR|CUSHION|DOORMAT|HOOKS?|DECORATION|ORNAMENT|HEARTS?|BUNTING|WREATH|VASE|DOORSTOP|DRAWER|CABINET|WALL|HANGING|CHIME|THROW|BLANKET|QUILT)\b')
               THEN 'home_decor'
           ELSE 'other'
       END AS product_group
FROM products p
LEFT JOIN canonical c
       ON c.stock_code = p.stock_code
      AND c.rn = 1;

-- ---------- dim_customer ----------
CREATE OR REPLACE TABLE core.dim_customer AS
WITH country_rank AS (
    -- Customers can appear with >1 country: keep the most frequent,
    -- tie-break on most recent, then alphabetical (deterministic).
    SELECT customer_id,
           country,
           ROW_NUMBER() OVER (
               PARTITION BY customer_id
               ORDER BY COUNT(*) DESC, MAX(invoice_ts) DESC, country
           ) AS rn,
           COUNT(*) OVER (PARTITION BY customer_id) AS n_countries
    FROM staging.lines
    WHERE include_in_core AND has_customer
    GROUP BY customer_id, country
),
activity AS (
    SELECT customer_id,
           MIN(invoice_ts) FILTER (WHERE NOT is_cancel)               AS first_purchase_ts,
           MAX(invoice_ts) FILTER (WHERE NOT is_cancel)               AS last_purchase_ts,
           COUNT(DISTINCT invoice_no) FILTER (WHERE NOT is_cancel)    AS n_purchase_invoices,
           COUNT(DISTINCT invoice_no) FILTER (WHERE is_cancel)        AS n_cancel_invoices
    FROM core.fact_line
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
)
SELECT a.customer_id,
       cr.country,
       cr.n_countries,
       a.first_purchase_ts,          -- observed first purchase: left-censored at data start
       a.last_purchase_ts,
       a.n_purchase_invoices,
       a.n_cancel_invoices,
       a.n_purchase_invoices > 0     AS has_purchase
FROM activity a
LEFT JOIN country_rank cr
       ON cr.customer_id = a.customer_id
      AND cr.rn = 1;

-- ---------- dim_date ----------
CREATE OR REPLACE TABLE core.dim_date AS
WITH bounds AS (
    SELECT CAST(MIN(invoice_ts) AS DATE) AS d0,
           CAST(MAX(invoice_ts) AS DATE) AS d1
    FROM core.fact_line
),
days AS (
    SELECT CAST(UNNEST(generate_series(CAST(d0 AS TIMESTAMP),
                                       CAST(d1 AS TIMESTAMP),
                                       INTERVAL 1 DAY)) AS DATE) AS date_key
    FROM bounds
)
SELECT date_key,
       year(date_key)                AS year,
       quarter(date_key)             AS quarter,
       month(date_key)               AS month,
       strftime(date_key, '%Y-%m')   AS year_month,
       CAST(date_trunc('month', date_key) AS DATE) AS month_start,
       CAST(date_trunc('week', date_key) AS DATE)  AS week_start,   -- ISO week (Monday)
       isodow(date_key)              AS iso_dow,                    -- 1 = Mon ... 7 = Sun
       dayname(date_key)             AS day_name,
       isodow(date_key) >= 6         AS is_weekend
FROM days;
