"""Step 4 - Point-in-time features and labels.

Creates the SQL macros customer_features(cutoff) and purchase_labels(cutoff, horizon),
then materialises:
  marts.model_frame                one row per (customer, cutoff) with features + label + split
  marts.customer_snapshot_current  features as of the day after the last invoice
                                   (for segmentation and the profile engine)
and writes reports/feature_dictionary.md, reports/model_frame_summary.md,
models/feature_schema.json.

Run:  python -m src.features
"""
from __future__ import annotations

import json

import pandas as pd

from src.build_warehouse import df_to_markdown
from src.config import load_config, path, sql_path
from src.db import connect, run_sql_file

ID_COLUMNS = ["customer_id", "cutoff_date"]
AUDIT_COLUMNS = ["country"]          # returned by the macro, never used for modelling
LABEL = "purchased_in_horizon"

# name, layer, family, SQL CTE, definition, rationale
FEATURES = [
    ("recency_days", "L3", "RFM", "order_stats", "Days from last purchase to the cutoff.",
     "Most established predictor of repeat buying."),
    ("tenure_days", "L1", "RFM", "order_stats", "Days from first observed purchase to the cutoff (left-censored at Dec 2009).",
     "Separates new from established accounts."),
    ("frequency_total", "L2", "RFM", "order_stats", "Purchase invoices before the cutoff.",
     "Long-run engagement."),
    ("frequency_365d", "L2", "RFM", "order_stats", "Purchase invoices in the last 365 days.",
     "Recent engagement on a comparable window."),
    ("frequency_90d", "L3", "RFM", "order_stats", "Purchase invoices in the last 90 days.",
     "Short-term engagement, same length as the label window."),
    ("gross_spend_total", "L2", "Monetary", "spend_agg", "Sum of non-cancelled line value before the cutoff (GBP).",
     "Customer value; gross so cancellations are captured separately."),
    ("gross_spend_365d", "L2", "Monetary", "spend_agg", "Gross spend in the last 365 days.",
     "Recent value."),
    ("gross_spend_90d", "L3", "Monetary", "spend_agg", "Gross spend in the last 90 days.",
     "Short-term value."),
    ("aov_365d", "L2", "Basket", "order_stats", "Average order value over the last 365 days.",
     "Order size distinguishes bulk/trade buyers from small buyers."),
    ("median_order_value", "L2", "Basket", "order_stats", "Median order value, all history before cutoff.",
     "Robust to one-off very large orders."),
    ("avg_products_per_order", "L2", "Basket", "order_stats", "Mean distinct products per order.",
     "Basket breadth."),
    ("median_units_per_order", "L2", "Basket", "order_stats", "Median total units per order.",
     "Bulk-buying indicator."),
    ("median_line_quantity", "L2", "Basket", "spend_agg", "Median quantity per purchased line.",
     "Bulk-buying at line level (wholesale pattern)."),
    ("avg_unit_price_paid", "L2", "Price", "spend_agg", "Gross spend / units bought.",
     "Price level of the products the customer buys."),
    ("price_index", "L2", "Price", "price", "Spend / (units x each product's median price before cutoff). <1 = pays below typical.",
     "Captures quantity-tier pricing; typical price is point-in-time, so no leakage."),
    ("distinct_products", "L2", "Diversity", "spend_agg", "Distinct stock codes bought.",
     "Range of the relationship."),
    ("distinct_product_groups", "L2", "Diversity", "mix", "Distinct product groups (heuristic taxonomy).",
     "Category breadth."),
    ("product_group_entropy", "L2", "Diversity", "mix", "Shannon entropy of spend shares across product groups.",
     "Specialist (low) vs generalist (high) buyer."),
    ("top_group_share", "L2", "Diversity", "mix", "Share of spend in the customer's largest product group.",
     "Concentration of the basket."),
    ("christmas_share", "L2", "Seasonality", "mix", "Share of spend on Christmas-themed products.",
     "Seasonal buying orientation."),
    ("repeat_sku_share", "L2", "Loyalty", "sku_repeat", "Share of purchased lines re-ordering a product bought on an earlier invoice.",
     "Replenishment behaviour - re-orderers tend to return."),
    ("cancel_rate", "L2", "Cancellations", "final select", "Cancellation invoices / purchase invoices.",
     "Observed cancellation behaviour (not a judgement of intent)."),
    ("cancel_value_share", "L2", "Cancellations", "final select", "Cancelled value / gross spend.",
     "Size of cancellations relative to purchases."),
    ("mean_gap_days", "L3", "Rhythm", "rhythm", "Mean days between distinct order days.",
     "Typical purchase cycle."),
    ("median_gap_days", "L3", "Rhythm", "rhythm", "Median days between distinct order days.",
     "Robust purchase cycle."),
    ("gap_cv", "L3", "Rhythm", "rhythm", "Std / mean of gaps (NULL with < 3 order days).",
     "Regular (low) vs irregular (high) buyer."),
    ("overdue_ratio", "L3", "Rhythm", "final select", "recency_days / median_gap_days.",
     "Silence relative to the customer's own rhythm."),
    ("spend_trend_90d_log_ratio", "L3", "Trend", "final select", "ln((spend last 90d + 1) / (spend previous 90d + 1)).",
     "Growing (>0) vs declining (<0) spend; symmetric and zero-safe."),
    ("order_trend_90d", "L3", "Trend", "final select", "Orders last 90d minus orders in the previous 90d.",
     "Change in order frequency."),
    ("active_windows_12m", "L3", "Trend", "trend", "Number of the last twelve 30-day windows with a purchase.",
     "Consistency of activity over the year."),
    ("spend_slope_rel_12m", "L3", "Trend", "trend", "Regression slope of 30-day spend over 12 windows / mean window spend.",
     "Scale-free long-run trend."),
    ("q4_spend_share", "L3", "Seasonality", "order_stats", "Share of order value placed in Oct-Dec.",
     "Seasonal (Christmas-stock) buyer vs year-round buyer."),
    ("morning_order_share", "L3", "Timing", "order_stats", "Share of orders placed before 12:00.",
     "Observed ordering-time habit."),
]
MODEL_FEATURES = [f[0] for f in FEATURES]


def horizon_days() -> int:
    return int(load_config()["modelling"]["horizon_days"])


def cutoff_schedule() -> dict[str, list[pd.Timestamp]]:
    m = load_config()["modelling"]
    return {
        "train": list(pd.date_range(m["train_cutoff_first"], m["train_cutoff_last"], freq="MS")),
        "valid": [pd.Timestamp(m["valid_cutoff"])],
        "test": [pd.Timestamp(m["test_cutoff"])],
    }


def create_macros(con) -> None:
    run_sql_file(con, sql_path("features"))
    run_sql_file(con, sql_path("labels"))


def features_at(con, cutoff) -> pd.DataFrame:
    c = pd.Timestamp(cutoff).strftime("%Y-%m-%d")
    return con.execute(f"SELECT * FROM customer_features(DATE '{c}')").df()


def _frame_sql(cutoff: pd.Timestamp, horizon: int, split: str) -> str:
    c = cutoff.strftime("%Y-%m-%d")
    return (
        f"SELECT '{split}' AS split, f.*, l.label_end_date, l.{LABEL}, l.n_future_orders, "
        f"l.future_net_spend, l.label_window_complete "
        f"FROM customer_features(DATE '{c}') AS f "
        f"INNER JOIN purchase_labels(DATE '{c}', {horizon}) AS l ON l.customer_id = f.customer_id"
    )


def build_model_frame(con) -> pd.DataFrame:
    h = horizon_days()
    parts = [_frame_sql(c, h, split) for split, cs in cutoff_schedule().items() for c in cs]
    con.execute("CREATE OR REPLACE TABLE marts.model_frame AS\n" + "\nUNION ALL\n".join(parts))
    summary = con.execute(f"""
        SELECT split,
               cutoff_date,
               MAX(label_end_date)                        AS label_end_date,
               COUNT(*)                                   AS customers,
               SUM({LABEL})                               AS positives,
               ROUND(100.0 * AVG({LABEL}), 1)             AS positive_rate_pct,
               BOOL_AND(label_window_complete)            AS window_complete
        FROM marts.model_frame
        GROUP BY split, cutoff_date
        ORDER BY cutoff_date
    """).df()
    if not summary["window_complete"].all():
        raise RuntimeError("A label window runs past the end of the data - check cutoffs in config.yaml")
    return summary


def build_current_snapshot(con):
    scoring_cutoff = con.execute(
        "SELECT CAST(MAX(invoice_date) + INTERVAL 1 DAY AS DATE) FROM core.fact_line").fetchone()[0]
    con.execute("CREATE OR REPLACE TABLE marts.customer_snapshot_current AS "
                f"SELECT * FROM customer_features(DATE '{scoring_cutoff}')")
    n = con.execute("SELECT COUNT(*) FROM marts.customer_snapshot_current").fetchone()[0]
    return scoring_cutoff, n


def feature_profile(con) -> pd.DataFrame:
    """Missingness and spread of each feature on the TRAIN split."""
    df = con.execute("SELECT * FROM marts.model_frame WHERE split = 'train'").df()
    rows = []
    for name in MODEL_FEATURES:
        s = pd.to_numeric(df[name], errors="coerce")
        rows.append({"feature": name,
                     "pct_missing": round(100 * s.isna().mean(), 1),
                     "p01": round(s.quantile(0.01), 3), "median": round(s.median(), 3),
                     "p99": round(s.quantile(0.99), 3), "max": round(s.max(), 3)})
    return pd.DataFrame(rows)


def write_docs(summary: pd.DataFrame, profile: pd.DataFrame, scoring_cutoff, n_current: int) -> None:
    reports, models = path("reports_dir"), path("models_dir")
    models.mkdir(parents=True, exist_ok=True)

    fd = pd.DataFrame(FEATURES, columns=["feature", "layer", "family", "sql_cte", "definition", "rationale"])
    (reports / "feature_dictionary.md").write_text(
        "# Feature Dictionary\n\n"
        "All features are computed in SQL by the point-in-time macro `customer_features(cutoff)` "
        "(`sql/04_features_macro.sql`) using only data **strictly before the cutoff**. "
        "`sql_cte` names the CTE inside that file where each feature is calculated.\n\n"
        "Layers: L1 raw attribute · L2 aggregated behaviour · L3 temporal. "
        "L4 (segment, propensity) and L5 (explanations) are added by later steps.\n\n"
        "**Excluded from modelling:** `country` (audit only - potential nationality proxy), "
        "`customer_id`, `cutoff_date`.\n\n"
        f"{len(fd)} model features.\n\n" + df_to_markdown(fd) + "\n",
        encoding="utf-8")

    (reports / "model_frame_summary.md").write_text(
        "# Model Frame Summary\n\n"
        f"Target: `{LABEL}` = at least one purchase in the {horizon_days()} days after the cutoff, "
        "for customers with at least one purchase in the 365 days before it.\n\n"
        "## Snapshots per cutoff\n\n" + df_to_markdown(summary) +
        "\n\n## Feature missingness and spread (train split)\n\n"
        "Missing values are expected for rhythm features when a customer has too few order days.\n\n"
        + df_to_markdown(profile) +
        f"\n\n## Current snapshot\n\nScoring cutoff {scoring_cutoff}: {n_current:,} active customers "
        "(used for segmentation and profiles).\n",
        encoding="utf-8")

    schema = {
        "model_features": MODEL_FEATURES,
        "id_columns": ID_COLUMNS,
        "audit_columns_excluded_from_model": AUDIT_COLUMNS,
        "label": LABEL,
        "horizon_days": horizon_days(),
        "population": "customers with >=1 purchase in the 365 days before the cutoff",
        "feature_window": "invoice_date < cutoff",
        "label_window": "cutoff <= invoice_date < cutoff + horizon_days",
        "cutoffs": {k: [c.strftime('%Y-%m-%d') for c in v] for k, v in cutoff_schedule().items()},
        "sql_sources": {"features": "sql/04_features_macro.sql", "labels": "sql/05_labels_macro.sql"},
        "features": [{"name": f[0], "layer": f[1], "family": f[2], "sql_cte": f[3], "definition": f[4]}
                     for f in FEATURES],
    }
    (models / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")


def main() -> None:
    con = connect()
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS marts")
        print("1/4 creating feature and label macros ...")
        create_macros(con)
        print("2/4 building marts.model_frame (stacked monthly cutoffs) ...")
        summary = build_model_frame(con)
        print(summary.to_string(index=False))
        print("3/4 building marts.customer_snapshot_current ...")
        scoring_cutoff, n_current = build_current_snapshot(con)
        print(f"    scoring cutoff {scoring_cutoff}: {n_current:,} active customers")
        print("4/4 writing feature dictionary, summary and schema ...")
        profile = feature_profile(con)
    finally:
        con.close()
    write_docs(summary, profile, scoring_cutoff, n_current)
    print("Done -> reports/feature_dictionary.md, reports/model_frame_summary.md, models/feature_schema.json")


if __name__ == "__main__":
    main()
