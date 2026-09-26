"""Reusable preprocessing: log transform for skewed features, winsorising, imputation, scaling.

All statistics (quantiles, medians, means, stds) are learned in .fit() on the data passed
to it only - for the model that is the TRAIN split - so nothing leaks from validation/test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Non-negative, heavily right-skewed features -> log1p
SKEWED_FEATURES = [
    "recency_days", "tenure_days", "frequency_total", "frequency_365d", "frequency_90d",
    "gross_spend_total", "gross_spend_365d", "gross_spend_90d", "aov_365d", "median_order_value",
    "avg_products_per_order", "median_units_per_order", "median_line_quantity", "avg_unit_price_paid",
    "distinct_products", "mean_gap_days", "median_gap_days", "overdue_ratio",
]


class LogTransformer(BaseEstimator, TransformerMixin):
    """log1p on the listed columns (negatives clipped to 0). Other columns pass through."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = [c for c in (self.columns or []) if c in X.columns]
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.columns_:
            X[c] = np.log1p(pd.to_numeric(X[c], errors="coerce").clip(lower=0))
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip every column to quantiles learned at fit time (NaN-aware)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        self.lo_ = X.quantile(self.lower)
        self.hi_ = X.quantile(self.upper)
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        return X.clip(lower=self.lo_, upper=self.hi_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def make_numeric_pipeline(features: list[str]) -> Pipeline:
    """log1p (skewed only) -> winsorise 1%/99% -> median impute -> standardise."""
    return Pipeline([
        ("log", LogTransformer(columns=[f for f in features if f in SKEWED_FEATURES])),
        ("winsor", Winsorizer(0.01, 0.99)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def skew_report(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Skewness before and after log1p for the skewed features (EDA evidence)."""
    rows = []
    for f in features:
        if f not in SKEWED_FEATURES:
            continue
        s = pd.to_numeric(df[f], errors="coerce").dropna()
        rows.append({"feature": f, "skew_raw": round(s.skew(), 2),
                     "skew_log1p": round(np.log1p(s.clip(lower=0)).skew(), 2)})
    return pd.DataFrame(rows)
