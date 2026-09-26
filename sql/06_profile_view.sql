-- Step 8 - Customer profile base: one row per active customer at the scoring cutoff.
-- Joins the point-in-time features (L1-L3), the segment and propensity (L4) and the
-- top SHAP drivers (L5), and adds percentile ranks so every number has context.
-- {segment_id_expr} and {segment_name_col} are filled in by src/profile_engine.py from
-- information_schema, so the view does not depend on the exact column names of Step 5.
CREATE OR REPLACE VIEW marts.customer_profile_base AS
WITH pct AS (
    SELECT customer_id,
           PERCENT_RANK() OVER (ORDER BY gross_spend_365d)   AS pct_spend_365d,
           PERCENT_RANK() OVER (ORDER BY frequency_365d)     AS pct_frequency_365d,
           PERCENT_RANK() OVER (ORDER BY aov_365d)           AS pct_aov_365d,
           PERCENT_RANK() OVER (ORDER BY distinct_products)  AS pct_distinct_products,
           PERCENT_RANK() OVER (ORDER BY active_windows_12m) AS pct_active_windows_12m,
           PERCENT_RANK() OVER (ORDER BY recency_days DESC)  AS pct_recency      -- 1 = most recent buyer
    FROM marts.customer_snapshot_current
),
driver_text AS (
    SELECT customer_id,
           STRING_AGG(label, '; ' ORDER BY rank) FILTER (WHERE direction = 'raises') AS drivers_raising,
           STRING_AGG(label, '; ' ORDER BY rank) FILTER (WHERE direction = 'lowers') AS drivers_lowering
    FROM marts.customer_drivers
    GROUP BY customer_id
)
SELECT f.*,
       CAST(f.cutoff_date AS DATE) - CAST(ROUND(f.recency_days) AS INTEGER) AS last_purchase_date,
       CAST(f.cutoff_date AS DATE) - CAST(ROUND(f.tenure_days) AS INTEGER)  AS first_purchase_date,
       CASE WHEN TRIM(f.country) = 'United Kingdom' THEN 'UK' ELSE 'International' END AS market,
       {segment_id_expr}          AS segment_id,
       s."{segment_name_col}"     AS segment_name,
       p.propensity_90d,
       p.propensity_decile,
       p.predicted_buyer,
       pct.pct_spend_365d, pct.pct_frequency_365d, pct.pct_aov_365d,
       pct.pct_distinct_products, pct.pct_active_windows_12m, pct.pct_recency,
       d.drivers_raising,
       d.drivers_lowering
FROM marts.customer_snapshot_current AS f
LEFT JOIN marts.customer_segments   AS s   ON s.customer_id   = f.customer_id
LEFT JOIN marts.customer_propensity AS p   ON p.customer_id   = f.customer_id
LEFT JOIN pct                              ON pct.customer_id = f.customer_id
LEFT JOIN driver_text               AS d   ON d.customer_id   = f.customer_id;
