-- Q05 · RFM segmentation (rule-based baseline)
-- Question: How do customers split by Recency, Frequency and Monetary value, and how much revenue does each RFM segment hold?
-- Techniques: CROSS JOIN reference date, date math (date_diff, INTERVAL), HAVING, NTILE(5) quintile scoring, CASE WHEN segment rules, SUM(SUM()) OVER () share
-- Note: descriptive, full-history snapshot as of the day after the last invoice. This is the rule-based
--       baseline that the ML segmentation (Step 5) is compared against. Ties broken by customer_id for determinism.
WITH ref AS (
    SELECT CAST(MAX(invoice_date) + INTERVAL 1 DAY AS DATE) AS ref_date
    FROM core.fact_line
),
base AS (
    SELECT f.customer_id,
           date_diff('day', MAX(f.invoice_date) FILTER (WHERE NOT f.is_cancel), r.ref_date) AS recency_days,
           COUNT(DISTINCT f.invoice_no) FILTER (WHERE NOT f.is_cancel)                      AS frequency,
           SUM(f.line_amount)                                                               AS monetary
    FROM core.fact_line AS f
    CROSS JOIN ref AS r
    WHERE f.customer_id IS NOT NULL
    GROUP BY f.customer_id, r.ref_date
    HAVING COUNT(DISTINCT f.invoice_no) FILTER (WHERE NOT f.is_cancel) > 0
       AND SUM(f.line_amount) > 0
),
scored AS (
    SELECT *,
           NTILE(5) OVER (ORDER BY recency_days DESC, customer_id) AS r_score,  -- 5 = most recent
           NTILE(5) OVER (ORDER BY frequency ASC,  customer_id)    AS f_score,  -- 5 = most frequent
           NTILE(5) OVER (ORDER BY monetary ASC,   customer_id)    AS m_score   -- 5 = highest value
    FROM base
),
segmented AS (
    SELECT *,
           CASE
               WHEN r_score >= 4 AND f_score >= 4 THEN 'Champions'
               WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal'
               WHEN r_score >= 4 AND f_score <= 2 THEN 'Recent low-frequency'
               WHEN r_score <= 2 AND f_score >= 3 THEN 'At risk (previously frequent)'
               WHEN r_score <= 2 AND f_score <= 2 THEN 'Hibernating'
               ELSE                                    'Needs attention'
           END AS rfm_segment
    FROM scored
)
SELECT rfm_segment,
       COUNT(*)                                                        AS n_customers,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)              AS pct_customers,
       ROUND(median(recency_days), 0)                                  AS median_recency_days,
       ROUND(median(frequency), 1)                                     AS median_frequency,
       ROUND(median(monetary), 2)                                      AS median_monetary,
       ROUND(100.0 * SUM(monetary) / SUM(SUM(monetary)) OVER (), 1)    AS pct_revenue
FROM segmented
GROUP BY rfm_segment
ORDER BY pct_revenue DESC;
