-- =====================================================================
-- 05_labels_macro.sql
-- purchase_labels(cutoff, horizon_days): the prediction TARGET.
--
--   Population : identical to customer_features(cutoff)
--                (>=1 purchase in [cutoff-365d, cutoff)).
--   Label      : purchased_in_horizon = 1 if >=1 non-cancelled invoice
--                in [cutoff, cutoff + horizon_days), else 0.
--   Reads ONLY rows on/after the cutoff for the label - never features.
--   label_window_complete = the whole window is inside the data range.
--
-- Usage: SELECT * FROM purchase_labels(DATE '2011-06-01', 90);
-- =====================================================================
CREATE OR REPLACE MACRO purchase_labels(cutoff, horizon_days) AS TABLE
WITH params AS (
    SELECT CAST(cutoff AS DATE)                                                     AS t,
           CAST(CAST(cutoff AS DATE) + to_days(CAST(horizon_days AS INTEGER)) AS DATE) AS t_end,
           CAST(CAST(cutoff AS DATE) - INTERVAL 365 DAY AS DATE)                    AS t365
),
population AS (
    SELECT DISTINCT i.customer_id
    FROM core.fact_invoice AS i
    CROSS JOIN params AS p
    WHERE i.customer_id IS NOT NULL
      AND NOT i.is_cancel
      AND i.invoice_date >= p.t365
      AND i.invoice_date <  p.t
),
future_orders AS (
    SELECT i.customer_id, COUNT(*) AS n_future_orders
    FROM core.fact_invoice AS i
    CROSS JOIN params AS p
    WHERE i.customer_id IS NOT NULL
      AND NOT i.is_cancel
      AND i.invoice_date >= p.t
      AND i.invoice_date <  p.t_end
    GROUP BY i.customer_id
),
future_spend AS (
    SELECT f.customer_id, CAST(SUM(f.line_amount) AS DOUBLE) AS future_net_spend
    FROM core.fact_line AS f
    CROSS JOIN params AS p
    WHERE f.customer_id IS NOT NULL
      AND f.invoice_date >= p.t
      AND f.invoice_date <  p.t_end
    GROUP BY f.customer_id
)
SELECT pop.customer_id,
       p.t                                                           AS cutoff_date,
       p.t_end                                                       AS label_end_date,
       CASE WHEN fo.customer_id IS NOT NULL THEN 1 ELSE 0 END        AS purchased_in_horizon,
       COALESCE(fo.n_future_orders, 0)                               AS n_future_orders,
       COALESCE(fs.future_net_spend, 0.0)                            AS future_net_spend,
       p.t_end <= (SELECT CAST(MAX(invoice_date) + INTERVAL 1 DAY AS DATE) FROM core.fact_line)
                                                                     AS label_window_complete
FROM population AS pop
CROSS JOIN params AS p
LEFT JOIN future_orders AS fo ON fo.customer_id = pop.customer_id
LEFT JOIN future_spend  AS fs ON fs.customer_id = pop.customer_id;
