"""Step 6 checks: calendar feature, metric helpers, split order, saved model and scores."""
import numpy as np
import pandas as pd
import pytest

from src.config import path
from src.propensity import (AUDIT_COLUMN, CAL_FEATURE, FEATURES, add_calendar_feature,
                            best_f1_threshold, check_time_order, expected_calibration_error,
                            peak_share, precision_at_top, predict_proba)

SEP_NOV = [9, 10, 11]


@pytest.mark.parametrize("cutoff, expected", [
    ("2011-09-01", 1.0),        # Sep 1 - Nov 29: all peak days  (test cutoff)
    ("2011-06-01", 0.0),        # Jun 1 - Aug 29: none           (valid cutoff)
    ("2010-08-01", 59 / 90),    # Aug 1 - Oct 29: 30 Sep + 29 Oct
    ("2010-10-01", 61 / 90),    # Oct 1 - Dec 29: 31 Oct + 30 Nov
    ("2011-12-10", 0.0),        # scoring cutoff
])
def test_peak_share_known_windows(cutoff, expected):
    assert peak_share(cutoff, 90, SEP_NOV) == pytest.approx(expected)


def test_calendar_feature_depends_only_on_cutoff():
    df = pd.DataFrame({"cutoff_date": pd.to_datetime(["2011-09-01"] * 3), "purchased_in_horizon": [0, 1, 0]})
    out = add_calendar_feature(df, 90, SEP_NOV)
    assert out[CAL_FEATURE].nunique() == 1   # same for every customer: no customer-level future info


def test_audit_column_is_not_a_model_feature():
    assert AUDIT_COLUMN not in FEATURES
    assert not {"customer_id", "cutoff_date", "purchased_in_horizon"} & set(FEATURES)


def test_metric_helpers():
    assert precision_at_top([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 0.5) == 0.5
    t = best_f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 < t <= 0.8
    assert expected_calibration_error([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_split_windows_do_not_overlap(con):
    df = con.execute("SELECT split, cutoff_date, label_end_date FROM marts.model_frame").df()
    check_time_order(df)   # raises if any label window runs into the next split


@pytest.fixture
def bundle():
    f = path("models_dir") / "propensity.joblib"
    if not f.exists():
        pytest.skip("run `python -m src.propensity` first")
    import joblib
    return joblib.load(f)


def test_saved_model_scores_are_probabilities(con, bundle):
    assert AUDIT_COLUMN not in bundle["features"]
    assert 0 < bundle["threshold"] < 1
    snap = con.execute("SELECT * FROM marts.customer_snapshot_current LIMIT 50").df()
    p = predict_proba(bundle, add_calendar_feature(snap, bundle["horizon_days"], bundle["peak_months"]))
    assert np.all((p >= 0) & (p <= 1))


def test_propensity_table_covers_current_snapshot(con):
    exists = con.execute("SELECT COUNT(*) FROM information_schema.tables "
                         "WHERE table_schema = 'marts' AND table_name = 'customer_propensity'").fetchone()[0]
    if not exists:
        pytest.skip("run `python -m src.propensity` first")
    n_snap, n_prop, bad = con.execute("""
        SELECT (SELECT COUNT(*) FROM marts.customer_snapshot_current),
               (SELECT COUNT(*) FROM marts.customer_propensity),
               (SELECT COUNT(*) FROM marts.customer_propensity
                 WHERE propensity_90d NOT BETWEEN 0 AND 1 OR propensity_decile NOT BETWEEN 1 AND 10)
    """).fetchone()
    assert n_snap == n_prop and bad == 0
