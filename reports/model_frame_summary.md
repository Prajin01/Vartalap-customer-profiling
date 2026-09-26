# Model Frame Summary

Target: `purchased_in_horizon` = at least one purchase in the 90 days after the cutoff, for customers with at least one purchase in the 365 days before it.

## Snapshots per cutoff

| split | cutoff_date | label_end_date | customers | positives | positive_rate_pct | window_complete |
| --- | --- | --- | --- | --- | --- | --- |
| train | 2010-06-01 00:00:00 | 2010-08-30 00:00:00 | 2,684 | 1,334.00 | 49.70 | 1 |
| train | 2010-07-01 00:00:00 | 2010-09-29 00:00:00 | 2,951 | 1,442.00 | 48.90 | 1 |
| train | 2010-08-01 00:00:00 | 2010-10-30 00:00:00 | 3,136 | 1,707.00 | 54.40 | 1 |
| train | 2010-09-01 00:00:00 | 2010-11-30 00:00:00 | 3,299 | 1,924.00 | 58.30 | 1 |
| train | 2010-10-01 00:00:00 | 2010-12-30 00:00:00 | 3,538 | 1,888.00 | 53.40 | 1 |
| train | 2010-11-01 00:00:00 | 2011-01-30 00:00:00 | 3,913 | 1,733.00 | 44.30 | 1 |
| train | 2010-12-01 00:00:00 | 2011-03-01 00:00:00 | 4,239 | 1,407.00 | 33.20 | 1 |
| train | 2011-01-01 00:00:00 | 2011-04-01 00:00:00 | 4,205 | 1,390.00 | 33.10 | 1 |
| train | 2011-02-01 00:00:00 | 2011-05-02 00:00:00 | 4,221 | 1,460.00 | 34.60 | 1 |
| train | 2011-03-01 00:00:00 | 2011-05-30 00:00:00 | 4,260 | 1,571.00 | 36.90 | 1 |
| valid | 2011-06-01 00:00:00 | 2011-08-30 00:00:00 | 4,324 | 1,584.00 | 36.60 | 1 |
| test | 2011-09-01 00:00:00 | 2011-11-30 00:00:00 | 4,324 | 2,142.00 | 49.50 | 1 |

## Feature missingness and spread (train split)

Missing values are expected for rhythm features when a customer has too few order days.

| feature | pct_missing | p01 | median | p99 | max |
| --- | --- | --- | --- | --- | --- |
| recency_days | 0.00 | 1.00 | 69.00 | 338.00 | 365.00 |
| tenure_days | 0.00 | 6.00 | 203.00 | 450.00 | 455.00 |
| frequency_total | 0.00 | 1.00 | 2.00 | 29.00 | 198.00 |
| frequency_365d | 0.00 | 1.00 | 2.00 | 28.00 | 180.00 |
| frequency_90d | 0.00 | 0.00 | 1.00 | 9.00 | 80.00 |
| gross_spend_total | 0.00 | 38.25 | 641.38 | 19,073.76 | 359,699.83 |
| gross_spend_365d | 0.00 | 36.25 | 621.54 | 17,642.89 | 321,329.74 |
| gross_spend_90d | 0.00 | 0.00 | 178.21 | 6,250.00 | 114,475.32 |
| aov_365d | 0.00 | 31.20 | 287.49 | 1,953.93 | 25,784.32 |
| median_order_value | 0.00 | 30.30 | 285.38 | 1,851.24 | 11,880.84 |
| avg_products_per_order | 0.00 | 1.00 | 17.25 | 86.11 | 230.00 |
| median_units_per_order | 0.00 | 8.00 | 144.00 | 1,180.00 | 87,167.00 |
| median_line_quantity | 0.00 | 1.00 | 6.00 | 200.00 | 4,320.00 |
| avg_unit_price_paid | 0.00 | 0.45 | 1.85 | 7.95 | 295.00 |
| price_index | 0.00 | 0.54 | 0.99 | 1.04 | 2.31 |
| distinct_products | 0.00 | 1.00 | 35.00 | 360.00 | 1,785.00 |
| distinct_product_groups | 0.00 | 1.00 | 7.00 | 10.00 | 10.00 |
| product_group_entropy | 0.00 | 0.00 | 1.58 | 2.06 | 2.21 |
| top_group_share | 0.00 | 0.20 | 0.37 | 1.00 | 1.00 |
| christmas_share | 0.00 | 0.00 | 0.00 | 0.37 | 1.00 |
| repeat_sku_share | 0.00 | 0.00 | 0.06 | 0.68 | 0.96 |
| cancel_rate | 0.00 | 0.00 | 0.00 | 1.50 | 5.00 |
| cancel_value_share | 0.00 | 0.00 | 0.00 | 0.41 | 5.00 |
| mean_gap_days | 37.50 | 4.13 | 55.50 | 290.00 | 418.00 |
| median_gap_days | 37.50 | 3.50 | 50.00 | 290.00 | 418.00 |
| gap_cv | 57.60 | 0.03 | 0.62 | 1.49 | 2.14 |
| overdue_ratio | 37.50 | 0.02 | 0.85 | 33.97 | 363.00 |
| spend_trend_90d_log_ratio | 0.00 | -7.34 | 0.00 | 7.48 | 11.25 |
| order_trend_90d | 0.00 | -4.00 | 0.00 | 5.00 | 58.00 |
| active_windows_12m | 0.00 | 1.00 | 2.00 | 11.00 | 12.00 |
| spend_slope_rel_12m | 0.20 | -0.46 | 0.13 | 0.46 | 0.46 |
| q4_spend_share | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |
| morning_order_share | 0.00 | 0.00 | 0.25 | 1.00 | 1.00 |

## Current snapshot

Scoring cutoff 2011-12-10: 4,261 active customers (used for segmentation and profiles).
