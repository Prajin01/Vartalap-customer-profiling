"""Checks on the saved segmentation (Step 5)."""
import json

import numpy as np
import pandas as pd
import pytest

from src.config import path
from src.preprocess import LogTransformer, Winsorizer

SEG_JSON = path("models_dir") / "segments.json"


def test_log_transformer_only_touches_listed_columns():
    df = pd.DataFrame({"a": [0.0, 9.0], "b": [1.0, 2.0]})
    out = LogTransformer(columns=["a"]).fit(df).transform(df)
    assert np.allclose(out["a"], np.log1p([0, 9])) and out["b"].tolist() == [1.0, 2.0]


def test_winsorizer_uses_fit_quantiles_only():
    train = pd.DataFrame({"x": np.arange(101, dtype=float)})
    w = Winsorizer(0.01, 0.99).fit(train)
    out = w.transform(pd.DataFrame({"x": [-1000.0, 1000.0]}))
    assert out["x"].tolist() == [1.0, 99.0]


@pytest.fixture(scope="module")
def seg():
    if not SEG_JSON.exists():
        pytest.skip("Run: python -m src.segmentation")
    return json.loads(SEG_JSON.read_text(encoding="utf-8"))


def test_every_active_customer_has_exactly_one_segment(con, seg):
    n_snapshot = con.execute("SELECT COUNT(*) FROM marts.customer_snapshot_current").fetchone()[0]
    n_rows, n_unique = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT customer_id) FROM marts.customer_segments").fetchone()
    assert n_rows == n_unique == n_snapshot


def test_segment_names_unique_and_match_k(seg):
    names = list(seg["names"].values())
    assert len(names) == seg["k"] == len(set(names))
    assert not any(n.lower().startswith("cluster") for n in names)


def test_chosen_solution_is_reasonably_stable(seg):
    chosen = [m for m in seg["metrics"] if m["method"] == seg["method"] and m["k"] == seg["k"]][0]
    assert chosen["stability_ari"] >= 0.6


def test_country_not_used_for_clustering(seg):
    assert "country" not in seg["features"]
