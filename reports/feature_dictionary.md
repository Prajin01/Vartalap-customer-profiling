# Feature Dictionary

All features are computed in SQL by the point-in-time macro `customer_features(cutoff)` (`sql/04_features_macro.sql`) using only data **strictly before the cutoff**. `sql_cte` names the CTE inside that file where each feature is calculated.

Layers: L1 raw attribute · L2 aggregated behaviour · L3 temporal. L4 (segment, propensity) and L5 (explanations) are added by later steps.

**Excluded from modelling:** `country` (audit only - potential nationality proxy), `customer_id`, `cutoff_date`.

33 model features.

| feature | layer | family | sql_cte | definition | rationale |
| --- | --- | --- | --- | --- | --- |
| recency_days | L3 | RFM | order_stats | Days from last purchase to the cutoff. | Most established predictor of repeat buying. |
| tenure_days | L1 | RFM | order_stats | Days from first observed purchase to the cutoff (left-censored at Dec 2009). | Separates new from established accounts. |
| frequency_total | L2 | RFM | order_stats | Purchase invoices before the cutoff. | Long-run engagement. |
| frequency_365d | L2 | RFM | order_stats | Purchase invoices in the last 365 days. | Recent engagement on a comparable window. |
| frequency_90d | L3 | RFM | order_stats | Purchase invoices in the last 90 days. | Short-term engagement, same length as the label window. |
| gross_spend_total | L2 | Monetary | spend_agg | Sum of non-cancelled line value before the cutoff (GBP). | Customer value; gross so cancellations are captured separately. |
| gross_spend_365d | L2 | Monetary | spend_agg | Gross spend in the last 365 days. | Recent value. |
| gross_spend_90d | L3 | Monetary | spend_agg | Gross spend in the last 90 days. | Short-term value. |
| aov_365d | L2 | Basket | order_stats | Average order value over the last 365 days. | Order size distinguishes bulk/trade buyers from small buyers. |
| median_order_value | L2 | Basket | order_stats | Median order value, all history before cutoff. | Robust to one-off very large orders. |
| avg_products_per_order | L2 | Basket | order_stats | Mean distinct products per order. | Basket breadth. |
| median_units_per_order | L2 | Basket | order_stats | Median total units per order. | Bulk-buying indicator. |
| median_line_quantity | L2 | Basket | spend_agg | Median quantity per purchased line. | Bulk-buying at line level (wholesale pattern). |
| avg_unit_price_paid | L2 | Price | spend_agg | Gross spend / units bought. | Price level of the products the customer buys. |
| price_index | L2 | Price | price | Spend / (units x each product's median price before cutoff). <1 = pays below typical. | Captures quantity-tier pricing; typical price is point-in-time, so no leakage. |
| distinct_products | L2 | Diversity | spend_agg | Distinct stock codes bought. | Range of the relationship. |
| distinct_product_groups | L2 | Diversity | mix | Distinct product groups (heuristic taxonomy). | Category breadth. |
| product_group_entropy | L2 | Diversity | mix | Shannon entropy of spend shares across product groups. | Specialist (low) vs generalist (high) buyer. |
| top_group_share | L2 | Diversity | mix | Share of spend in the customer's largest product group. | Concentration of the basket. |
| christmas_share | L2 | Seasonality | mix | Share of spend on Christmas-themed products. | Seasonal buying orientation. |
| repeat_sku_share | L2 | Loyalty | sku_repeat | Share of purchased lines re-ordering a product bought on an earlier invoice. | Replenishment behaviour - re-orderers tend to return. |
| cancel_rate | L2 | Cancellations | final select | Cancellation invoices / purchase invoices. | Observed cancellation behaviour (not a judgement of intent). |
| cancel_value_share | L2 | Cancellations | final select | Cancelled value / gross spend. | Size of cancellations relative to purchases. |
| mean_gap_days | L3 | Rhythm | rhythm | Mean days between distinct order days. | Typical purchase cycle. |
| median_gap_days | L3 | Rhythm | rhythm | Median days between distinct order days. | Robust purchase cycle. |
| gap_cv | L3 | Rhythm | rhythm | Std / mean of gaps (NULL with < 3 order days). | Regular (low) vs irregular (high) buyer. |
| overdue_ratio | L3 | Rhythm | final select | recency_days / median_gap_days. | Silence relative to the customer's own rhythm. |
| spend_trend_90d_log_ratio | L3 | Trend | final select | ln((spend last 90d + 1) / (spend previous 90d + 1)). | Growing (>0) vs declining (<0) spend; symmetric and zero-safe. |
| order_trend_90d | L3 | Trend | final select | Orders last 90d minus orders in the previous 90d. | Change in order frequency. |
| active_windows_12m | L3 | Trend | trend | Number of the last twelve 30-day windows with a purchase. | Consistency of activity over the year. |
| spend_slope_rel_12m | L3 | Trend | trend | Regression slope of 30-day spend over 12 windows / mean window spend. | Scale-free long-run trend. |
| q4_spend_share | L3 | Seasonality | order_stats | Share of order value placed in Oct-Dec. | Seasonal (Christmas-stock) buyer vs year-round buyer. |
| morning_order_share | L3 | Timing | order_stats | Share of orders placed before 12:00. | Observed ordering-time habit. |
