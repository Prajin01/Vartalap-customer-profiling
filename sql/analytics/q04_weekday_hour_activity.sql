-- Q04 · When do customers order?
-- Question: On which weekdays and at which times of day are orders placed? (Useful for timing outreach messages.)
-- Techniques: INNER JOIN fact_invoice to dim_date, CASE WHEN hour bands, hour() extraction, GROUP BY alias, share via SUM(COUNT(*)) OVER ()
-- Note: timestamps are the retailer's system time (UK). Weekdays with no rows simply do not appear.
WITH orders AS (
    SELECT fi.invoice_no,
           fi.invoice_ts,
           fi.invoice_amount,
           d.iso_dow,
           d.day_name
    FROM core.fact_invoice AS fi
    INNER JOIN core.dim_date AS d
            ON d.date_key = fi.invoice_date
    WHERE NOT fi.is_cancel
)
SELECT iso_dow,
       day_name,
       CASE
           WHEN hour(invoice_ts) < 10 THEN '1_before_10h'
           WHEN hour(invoice_ts) < 13 THEN '2_10h_to_13h'
           WHEN hour(invoice_ts) < 16 THEN '3_13h_to_16h'
           ELSE                            '4_16h_onwards'
       END                                                       AS hour_band,
       COUNT(*)                                                  AS n_orders,
       ROUND(SUM(invoice_amount), 2)                             AS revenue,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)        AS pct_of_orders
FROM orders
GROUP BY iso_dow, day_name, hour_band
ORDER BY iso_dow, hour_band;
