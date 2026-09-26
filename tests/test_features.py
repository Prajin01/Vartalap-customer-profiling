"""Leakage and consistency tests for the point-in-time features and labels (Step 4)."""
import duckdb
import pandas as pd
import pytest

from src.config import path, sql_path
from src.db import run_sql_file
from src.features import LABEL, MODEL_FEATURES, cutoff_schedule, horizon_days

CUTOFF = "2011-03-01"


def test_split_label_windows_never_reach_later_splits():
    s, h = cutoff_schedule(), pd.Timedelta(days=horizon_days())
    assert max(s["train"]) + h <= s["valid"][0]
    assert s["valid"][0] + h <= s["test"][0]


def test_features_ignore_future_data(con):
    """Gold-standard leakage test: delete every row on/after the cutoff, rebuild the
    macro on the truncated copy, and require identical features."""
    q = f"SELECT * FROM customer_features(DATE '{CUTOFF}') ORDER BY customer_id"
    full = con.execute(q).df()

    mem = duckdb.connect()
    mem.execute(f"ATTACH '{path('duckdb').as_posix()}' AS wh (READ_ONLY)")
    mem.execute("CREATE SCHEMA core")
    mem.execute(f"CREATE TABLE core.fact_line AS SELECT * FROM wh.core.fact_line WHERE invoice_date < DATE '{CUTOFF}'")
    mem.execute(f"CREATE TABLE core.fact_invoice AS SELECT * FROM wh.core.fact_invoice WHERE invoice_date < DATE '{CUTOFF}'")
    mem.execute("CREATE TABLE core.dim_product AS SELECT * FROM wh.core.dim_product")
    mem.execute("CREATE TABLE core.dim_customer AS SELECT * FROM wh.core.dim_customer")
    mem.execute("DETACH wh")
    run_sql_file(mem, sql_path("features"))
    truncated = mem.execute(q).df()
    mem.close()

    pd.testing.assert_frame_equal(full, truncated, check_exact=False, rtol=1e-9)


def test_recency_inside_active_window(con):
    lo, hi = con.execute(
        f"SELECT MIN(recency_days), MAX(recency_days) FROM customer_features(DATE '{CUTOFF}')").fetchone()
    assert lo >= 1 and hi <= 365


def test_features_and_labels_share_population(con):
    only_one_side = con.execute(f"""
        SELECT COUNT(*) FROM customer_features(DATE '{CUTOFF}') AS f
        FULL OUTER JOIN purchase_labels(DATE '{CUTOFF}', 90) AS l ON l.customer_id = f.customer_id
        WHERE f.customer_id IS NULL OR l.customer_id IS NULL
    """).fetchone()[0]
    assert only_one_side == 0


def test_labels_match_direct_query(con):
    h = horizon_days()
    macro = set(con.execute(
        f"SELECT customer_id FROM purchase_labels(DATE '{CUTOFF}', {h}) WHERE {LABEL} = 1").df()["customer_id"])
    direct = set(con.execute(f"""
        SELECT DISTINCT customer_id FROM core.fact_invoice
        WHERE customer_id IS NOT NULL AND NOT is_cancel
          AND invoice_date >= DATE '{CUTOFF}'
          AND invoice_date <  CAST(DATE '{CUTOFF}' + INTERVAL {h} DAY AS DATE)
          AND customer_id IN (
              SELECT customer_id FROM core.fact_invoice
              WHERE NOT is_cancel
                AND invoice_date >= CAST(DATE '{CUTOFF}' - INTERVAL 365 DAY AS DATE)
                AND invoice_date <  DATE '{CUTOFF}')
    """).df()["customer_id"])
    assert macro == direct


def test_model_frame_is_complete_and_unique(con):
    df = con.execute("SELECT * FROM marts.model_frame").df()
    assert set(MODEL_FEATURES) <= set(df.columns)
    assert df["label_window_complete"].all()
    assert not df.duplicated(["customer_id", "cutoff_date"]).any()
    assert set(df["split"]) == {"train", "valid", "test"}


def test_country_is_not_a_model_feature():
    assert "country" not in MODEL_FEATURES


@pytest.mark.parametrize("split", ["valid", "test"])
def test_each_split_has_both_classes(con, split):
    rate = con.execute(f"SELECT AVG({LABEL}) FROM marts.model_frame WHERE split = '{split}'").fetchone()[0]
    assert 0.05 < rate < 0.95
