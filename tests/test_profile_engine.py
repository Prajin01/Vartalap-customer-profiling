"""Step 8 checks: action rules, formatting, profile assembly, warehouse view and get_profile."""
import json

import numpy as np
import pandas as pd
import pytest

from src.profile_engine import VIEW, build_profile, choose_action, fmt, get_profile


def test_action_rules_in_priority_order():
    assert choose_action("High-cancellation accounts", 10, 0.9, 0.5, 0.3)["action"].startswith("Review service")
    assert choose_action("Core high-value repeat accounts", 9, 0.9, 0.5, 0)["action"].startswith("No incentive")
    assert choose_action("Core high-value repeat accounts", 6, 0.9, 0.5, 0)["action"].startswith("Priority retention")
    assert choose_action("Lapsed buyers", 2, 0.3, 3.0, 0)["action"].startswith("Low-cost win-back")
    assert choose_action("Lapsed buyers", 4, 0.3, 2.4, 0)["action"].startswith("Low-cost win-back")
    assert choose_action("Low-spend occasional buyers", 2, 0.1, 2.5, 0)["action"].startswith("Low-cost win-back")
    assert choose_action("Recent seasonal buyers", 5, 0.3, 1.0, 0)["action"].startswith("Time contact")
    assert choose_action(None, None, None, None, None)["action"].startswith("Standard")


def test_trend_text_handles_empty_periods():
    import math
    from src.profile_engine import trend_text
    assert trend_text(math.log(271 + 1), 271) == "up (no spend in the previous 90 days)"
    assert trend_text(0.0, 0) == "flat (no spend in either period)"
    assert trend_text(math.log(1 / 501), 0) == "down (no spend in the last 90 days)"
    assert trend_text(math.log(201 / 101), 200) == "up (+100%)"   # 100 -> 200


def test_formatting():
    assert fmt(1234.5, "gbp") == "£1,234" or fmt(1234.5, "gbp") == "£1,235"
    assert fmt(0.456, "pct") == "46%"
    assert fmt(float("nan"), "days") == "n/a"
    assert fmt(None, "int") == "n/a"


def test_profile_from_minimal_row_is_json_and_flags_missing_rhythm():
    row = {"customer_id": 1.0, "cutoff_date": pd.Timestamp("2011-12-10"), "frequency_total": 1,
           "median_gap_days": np.nan, "propensity_decile": np.int64(2), "segment_name": "Lapsed buyers",
           "overdue_ratio": np.nan}
    empty = pd.DataFrame(columns=["direction", "rank", "label", "family", "feature", "feature_value",
                                  "contribution_logodds"])
    p = build_profile(row, empty, pd.DataFrame(columns=["family", "contribution_logodds"]))
    json.dumps(p, default=str)
    assert p["customer_id"] == 1
    assert {"L1_attributes", "L2_behaviour", "L3_temporal", "L4_model", "L5_explanations"} <= set(p)
    assert any("rhythm" in n for n in p["data_notes"])


def _view_exists(con) -> bool:
    return con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'marts' "
                       "AND table_name = 'customer_profile_base'").fetchone()[0] > 0


def test_view_has_one_row_per_active_customer(con):
    if not _view_exists(con):
        pytest.skip("run `python -m src.profile_engine` first")
    n_view, n_distinct, n_snap, no_seg, no_score = con.execute(f"""
        SELECT COUNT(*), COUNT(DISTINCT customer_id),
               (SELECT COUNT(*) FROM marts.customer_snapshot_current),
               SUM(CASE WHEN segment_name IS NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN propensity_90d IS NULL THEN 1 ELSE 0 END)
        FROM {VIEW}
    """).fetchone()
    assert n_view == n_distinct == n_snap
    assert no_seg == 0 and no_score == 0


def test_get_profile_for_real_and_unknown_customer(con):
    if not _view_exists(con):
        pytest.skip("run `python -m src.profile_engine` first")
    cid = con.execute(f"SELECT customer_id FROM {VIEW} ORDER BY propensity_90d DESC LIMIT 1").fetchone()[0]
    p = get_profile(cid, con)
    assert p["found"] and 0 <= p["L4_model"]["propensity_90d"] <= 1
    assert p["L4_model"]["segment"]
    assert all(d["contribution_logodds"] > 0 for d in p["L5_explanations"]["raising_score"])
    assert all(d["contribution_logodds"] < 0 for d in p["L5_explanations"]["lowering_score"])
    json.dumps(p, default=str)
    assert get_profile("not-a-customer", con)["found"] is False
