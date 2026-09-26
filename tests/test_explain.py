"""Step 7 checks: labels, driver logic, SHAP additivity, driver tables."""
import numpy as np
import pandas as pd
import pytest

from src.config import path
from src.explain import (BEHAVIOUR, CAL_FEATURE, FAMILY, FEATURE_LABELS, contributions,
                         family_contributions, top_drivers)
from src.propensity import FEATURES, add_calendar_feature


def test_every_feature_has_a_label_and_family():
    assert set(FEATURES) <= set(FEATURE_LABELS)
    assert set(FEATURES) <= set(FAMILY)
    assert CAL_FEATURE not in BEHAVIOUR


def test_top_drivers_signs_limits_and_calendar_excluded():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((5, len(FEATURES))), columns=FEATURES)
    C = rng.normal(0, 1, (5, len(FEATURES)))
    C[:, FEATURES.index(CAL_FEATURE)] = -9.0          # largest magnitude, must still be skipped
    d = top_drivers(range(5), X, C, k=3)
    assert CAL_FEATURE not in set(d["feature"])
    assert d.groupby(["customer_id", "direction"]).size().max() <= 3
    assert (d.loc[d.direction == "raises", "contribution_logodds"] > 0).all()
    assert (d.loc[d.direction == "lowers", "contribution_logodds"] < 0).all()
    fam = family_contributions(range(5), C)
    assert "Calendar" not in set(fam["family"])


@pytest.fixture
def bundle():
    f = path("models_dir") / "propensity.joblib"
    if not f.exists():
        pytest.skip("run `python -m src.propensity` first")
    import joblib
    return joblib.load(f)


def test_shap_values_add_up_to_model_output(con, bundle):
    snap = con.execute("SELECT * FROM marts.customer_snapshot_current LIMIT 200").df()
    snap = add_calendar_feature(snap, bundle["horizon_days"], bundle["peak_months"])
    C, base = contributions(bundle["lightgbm"], snap)
    raw = bundle["lightgbm"].predict(snap[FEATURES], raw_score=True)
    assert np.allclose(base + C.sum(axis=1), raw, atol=1e-6)


def test_driver_tables(con):
    exists = con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'marts' "
                         "AND table_name = 'customer_drivers'").fetchone()[0]
    if not exists:
        pytest.skip("run `python -m src.explain` first")
    max_per_side, bad_sign, missing = con.execute("""
        WITH per_side AS (SELECT customer_id, direction, COUNT(*) AS n
                          FROM marts.customer_drivers GROUP BY 1, 2)
        SELECT (SELECT MAX(n) FROM per_side),
               (SELECT COUNT(*) FROM marts.customer_drivers
                 WHERE (direction = 'raises' AND contribution_logodds <= 0)
                    OR (direction = 'lowers' AND contribution_logodds >= 0)),
               (SELECT COUNT(*) FROM marts.customer_snapshot_current s
                 LEFT JOIN (SELECT DISTINCT customer_id FROM marts.customer_drivers) d USING (customer_id)
                 WHERE d.customer_id IS NULL)
    """).fetchone()
    assert max_per_side <= 3 and bad_sign == 0 and missing == 0
