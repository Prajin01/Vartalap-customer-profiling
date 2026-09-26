-- =====================================================================
-- 04_features_macro.sql
-- customer_features(cutoff): POINT-IN-TIME customer feature table.
--
-- Leakage rules (enforced by tests/test_features.py):
--   * Only rows with invoice_date <  cutoff are read (strictly before).
--   * Population = customers with >=1 purchase in [cutoff-365d, cutoff).
--   * Product "typical price" is also computed from data before the cutoff.
--   * country is returned for AUDIT ONLY and is excluded from the model.
-- Labels live in a separate macro (05) and read only rows >= cutoff.
--
-- Usage: SELECT * FROM customer_features(DATE '2011-06-01');
-- =====================================================================
CREATE OR REPLACE MACRO customer_features(cutoff) AS TABLE
WITH params AS (
    SELECT CAST(cutoff AS DATE)                                  AS t,
           CAST(CAST(cutoff AS DATE) - INTERVAL 90 DAY  AS DATE) AS t90,
           CAST(CAST(cutoff AS DATE) - INTERVAL 180 DAY AS DATE) AS t180,
           CAST(CAST(cutoff AS DATE) - INTERVAL 365 DAY AS DATE) AS t365
),
-- ---------- inputs, truncated at the cutoff ----------
lines AS (
    SELECT f.customer_id, f.invoice_no, f.stock_code, f.invoice_ts, f.invoice_date,
           f.quantity, f.unit_price, f.line_amount, f.is_cancel,
           p.t, p.t90, p.t180, p.t365
    FROM core.fact_line AS f
    CROSS JOIN params AS p
    WHERE f.customer_id IS NOT NULL
      AND f.invoice_date < p.t
),
orders AS (
    SELECT i.customer_id, i.invoice_no, i.invoice_ts, i.invoice_date,
           i.invoice_amount, i.n_products, i.total_quantity,
           p.t, p.t90, p.t180, p.t365
    FROM core.fact_invoice AS i
    CROSS JOIN params AS p
    WHERE i.customer_id IS NOT NULL
      AND NOT i.is_cancel
      AND i.invoice_date < p.t
),
population AS (
    SELECT DISTINCT customer_id
    FROM orders
    WHERE invoice_date >= t365
),
-- ---------- L2/L3: order-level behaviour ----------
order_stats AS (
    SELECT customer_id,
           date_diff('day', MAX(invoice_date), ANY_VALUE(t))                       AS recency_days,
           date_diff('day', MIN(invoice_date), ANY_VALUE(t))                       AS tenure_days,
           COUNT(*)                                                                AS frequency_total,
           COUNT(*) FILTER (WHERE invoice_date >= t365)                            AS frequency_365d,
           COUNT(*) FILTER (WHERE invoice_date >= t90)                             AS frequency_90d,
           COUNT(*) FILTER (WHERE invoice_date >= t180 AND invoice_date < t90)     AS frequency_prev_90d,
           CAST(SUM(invoice_amount) FILTER (WHERE invoice_date >= t365) AS DOUBLE)
               / NULLIF(COUNT(*) FILTER (WHERE invoice_date >= t365), 0)           AS aov_365d,
           median(CAST(invoice_amount AS DOUBLE))                                  AS median_order_value,
           AVG(CAST(n_products AS DOUBLE))                                         AS avg_products_per_order,
           median(CAST(total_quantity AS DOUBLE))                                  AS median_units_per_order,
           AVG(CASE WHEN hour(invoice_ts) < 12 THEN 1.0 ELSE 0.0 END)              AS morning_order_share,
           COALESCE(CAST(SUM(invoice_amount) FILTER (WHERE month(invoice_date) >= 10) AS DOUBLE), 0)
               / NULLIF(CAST(SUM(invoice_amount) AS DOUBLE), 0)                    AS q4_spend_share
    FROM orders
    GROUP BY customer_id
),
-- ---------- L2: spend, cancellations, basket ----------
spend_agg AS (
    SELECT customer_id,
           CAST(COALESCE(SUM(line_amount) FILTER (WHERE NOT is_cancel), 0) AS DOUBLE)                        AS gross_spend_total,
           CAST(COALESCE(SUM(line_amount) FILTER (WHERE NOT is_cancel AND invoice_date >= t365), 0) AS DOUBLE) AS gross_spend_365d,
           CAST(COALESCE(SUM(line_amount) FILTER (WHERE NOT is_cancel AND invoice_date >= t90), 0) AS DOUBLE)  AS gross_spend_90d,
           CAST(COALESCE(SUM(line_amount) FILTER (WHERE NOT is_cancel AND invoice_date >= t180
                                                   AND invoice_date < t90), 0) AS DOUBLE)                    AS gross_spend_prev_90d,
           CAST(COALESCE(-SUM(line_amount) FILTER (WHERE is_cancel), 0) AS DOUBLE)                           AS cancelled_value_total,
           COUNT(DISTINCT invoice_no) FILTER (WHERE is_cancel)                                               AS n_cancel_invoices,
           COUNT(DISTINCT stock_code) FILTER (WHERE NOT is_cancel)                                           AS distinct_products,
           median(CAST(quantity AS DOUBLE)) FILTER (WHERE NOT is_cancel)                                     AS median_line_quantity,
           CAST(SUM(line_amount) FILTER (WHERE NOT is_cancel) AS DOUBLE)
               / NULLIF(CAST(SUM(quantity) FILTER (WHERE NOT is_cancel) AS DOUBLE), 0)                       AS avg_unit_price_paid
    FROM lines
    GROUP BY customer_id
),
-- ---------- L2: price paid vs typical price (typical = median before cutoff, all buyers) ----------
product_price AS (
    SELECT f.stock_code, median(CAST(f.unit_price AS DOUBLE)) AS typical_price
    FROM core.fact_line AS f
    CROSS JOIN params AS p
    WHERE NOT f.is_cancel
      AND f.invoice_date < p.t
    GROUP BY f.stock_code
),
price AS (
    SELECT l.customer_id,
           CAST(SUM(l.line_amount) AS DOUBLE)
               / NULLIF(SUM(CAST(l.quantity AS DOUBLE) * pp.typical_price), 0) AS price_index
    FROM lines AS l
    INNER JOIN product_price AS pp
            ON pp.stock_code = l.stock_code
    WHERE NOT l.is_cancel
    GROUP BY l.customer_id
),
-- ---------- L2: product-mix diversity ----------
group_spend AS (
    SELECT l.customer_id, dp.product_group, CAST(SUM(l.line_amount) AS DOUBLE) AS spend
    FROM lines AS l
    INNER JOIN core.dim_product AS dp
            ON dp.stock_code = l.stock_code
    WHERE NOT l.is_cancel
    GROUP BY l.customer_id, dp.product_group
),
mix AS (
    SELECT customer_id,
           COUNT(*)                                                                     AS distinct_product_groups,
           -SUM(share * ln(share))                                                      AS product_group_entropy,
           MAX(share)                                                                   AS top_group_share,
           COALESCE(SUM(share) FILTER (WHERE product_group = 'seasonal_christmas'), 0) AS christmas_share
    FROM (
        SELECT customer_id, product_group,
               spend / SUM(spend) OVER (PARTITION BY customer_id) AS share
        FROM group_spend
    ) AS g
    GROUP BY customer_id
),
-- ---------- L2: re-ordering of previously bought products ----------
sku_repeat AS (
    SELECT customer_id,
           AVG(CASE WHEN invoice_ts > first_ts THEN 1.0 ELSE 0.0 END) AS repeat_sku_share
    FROM (
        SELECT customer_id, invoice_ts,
               MIN(invoice_ts) OVER (PARTITION BY customer_id, stock_code) AS first_ts
        FROM lines
        WHERE NOT is_cancel
    ) AS x
    GROUP BY customer_id
),
-- ---------- L3: purchase rhythm (gaps between distinct order days) ----------
gaps AS (
    SELECT customer_id,
           date_diff('day',
                     LAG(invoice_date) OVER (PARTITION BY customer_id ORDER BY invoice_date),
                     invoice_date) AS gap_days
    FROM (SELECT DISTINCT customer_id, invoice_date FROM orders) AS d
),
rhythm AS (
    SELECT customer_id,
           AVG(gap_days)                                         AS mean_gap_days,
           median(gap_days)                                      AS median_gap_days,
           stddev_samp(gap_days) / NULLIF(AVG(gap_days), 0)      AS gap_cv
    FROM gaps
    GROUP BY customer_id
),
-- ---------- L3: 12 x 30-day spend windows -> activity and relative trend ----------
buckets AS (
    SELECT UNNEST(range(0, 12)) AS bucket          -- 0 = most recent 30 days
),
bucket_spend AS (
    SELECT customer_id,
           (date_diff('day', invoice_date, t) - 1) // 30 AS bucket,
           CAST(SUM(line_amount) AS DOUBLE)              AS spend
    FROM lines
    WHERE NOT is_cancel
      AND date_diff('day', invoice_date, t) <= 360
    GROUP BY 1, 2
),
trend AS (
    SELECT pop.customer_id,
           COUNT(bs.spend)                                                     AS active_windows_12m,
           regr_slope(COALESCE(bs.spend, 0.0), CAST(11 - b.bucket AS DOUBLE))
               / NULLIF(AVG(COALESCE(bs.spend, 0.0)), 0)                       AS spend_slope_rel_12m
    FROM population AS pop
    CROSS JOIN buckets AS b
    LEFT JOIN bucket_spend AS bs
           ON bs.customer_id = pop.customer_id
          AND bs.bucket = b.bucket
    GROUP BY pop.customer_id
)
-- ---------- final: one row per customer in the active population ----------
SELECT pop.customer_id,
       (SELECT t FROM params)                                              AS cutoff_date,
       dc.country                                                          AS country,   -- AUDIT ONLY
       -- recency / frequency / monetary
       o.recency_days,
       o.tenure_days,
       o.frequency_total,
       o.frequency_365d,
       o.frequency_90d,
       s.gross_spend_total,
       s.gross_spend_365d,
       s.gross_spend_90d,
       -- basket
       o.aov_365d,
       o.median_order_value,
       o.avg_products_per_order,
       o.median_units_per_order,
       s.median_line_quantity,
       s.avg_unit_price_paid,
       pr.price_index,
       -- product mix
       s.distinct_products,
       mx.distinct_product_groups,
       mx.product_group_entropy,
       mx.top_group_share,
       mx.christmas_share,
       sr.repeat_sku_share,
       -- cancellations
       CAST(s.n_cancel_invoices AS DOUBLE) / o.frequency_total             AS cancel_rate,
       s.cancelled_value_total / NULLIF(s.gross_spend_total, 0)            AS cancel_value_share,
       -- rhythm
       rh.mean_gap_days,
       rh.median_gap_days,
       rh.gap_cv,
       o.recency_days / NULLIF(rh.median_gap_days, 0)                      AS overdue_ratio,
       -- trend and timing
       ln((s.gross_spend_90d + 1.0) / (s.gross_spend_prev_90d + 1.0))      AS spend_trend_90d_log_ratio,
       o.frequency_90d - o.frequency_prev_90d                              AS order_trend_90d,
       tr.active_windows_12m,
       tr.spend_slope_rel_12m,
       o.q4_spend_share,
       o.morning_order_share
FROM population AS pop
INNER JOIN order_stats AS o  ON o.customer_id  = pop.customer_id
INNER JOIN spend_agg   AS s  ON s.customer_id  = pop.customer_id
LEFT JOIN price        AS pr ON pr.customer_id = pop.customer_id
LEFT JOIN mix          AS mx ON mx.customer_id = pop.customer_id
LEFT JOIN sku_repeat   AS sr ON sr.customer_id = pop.customer_id
LEFT JOIN rhythm       AS rh ON rh.customer_id = pop.customer_id
LEFT JOIN trend        AS tr ON tr.customer_id = pop.customer_id
LEFT JOIN core.dim_customer AS dc ON dc.customer_id = pop.customer_id;
