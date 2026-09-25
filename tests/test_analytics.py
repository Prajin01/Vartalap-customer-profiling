"""Every analytics query runs, returns rows, and key results are internally consistent."""
import pytest

from src.analytics import list_queries, run_query


def test_ten_queries_exist():
    assert len(list_queries()) >= 10


@pytest.mark.parametrize("name", list_queries())
def test_query_runs_and_returns_rows(con, name):
    assert len(run_query(con, name)) > 0


def test_top_customers_ranked_descending(con):
    df = run_query(con, "q01_top_customers")
    assert df["revenue_rank"].min() == 1
    assert df["net_revenue"].is_monotonic_decreasing


def test_monthly_revenue_reconciles_to_fact_line(con):
    monthly_total = run_query(con, "q02_monthly_revenue")["net_revenue"].sum()
    fact_total = float(con.execute("SELECT SUM(line_amount) FROM core.fact_line").fetchone()[0])
    assert abs(float(monthly_total) - fact_total) < 1.0  # per-month rounding only


def test_rfm_revenue_shares_sum_to_100(con):
    df = run_query(con, "q05_rfm_segments")
    assert abs(df["pct_revenue"].sum() - 100) < 0.5
    assert abs(df["pct_customers"].sum() - 100) < 0.5


def test_cohort_month_zero_is_100_percent(con):
    df = run_query(con, "q09_cohort_retention")
    assert (df.loc[df["months_since_first"] == 0, "retention_pct"] == 100).all()


def test_lapsed_status_values_are_known(con):
    statuses = set(run_query(con, "q10_lapsed_customers")["status"])
    assert statuses <= {"lapsed", "at_risk", "active", "insufficient_history"}
