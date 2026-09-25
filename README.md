# Customer Behaviour & Profiling — ML Proof of Concept

A reproducible customer profiling system built on the **UCI Online Retail II** dataset:
a DuckDB SQL data model and point-in-time feature tables, behavioural segmentation,
a leakage-safe purchase-propensity model with calibration, SHAP explanations, and a
`get_profile(customer_id)` engine.

> Built as the Task 2 proof-of-concept for the Vartalap AI — AI/ML Junior Intern selection.

## Status

| Step | Component | Status |
|---|---|---|
| 1 | Ingest (xlsx → parquet) + data-quality report | ✅ |
| 2 | SQL staging + star schema + integrity tests | ✅ |
| 3 | SQL analytics showcase (10 business questions) | ⏳ |
| 4 | Point-in-time feature & label macros + leakage tests | ⏳ |
| 5 | EDA + feature dictionary | ⏳ |
| 6 | Segmentation (K-Means vs GMM, stability) | ⏳ |
| 7 | Purchase-propensity model (baselines → LightGBM, calibration) | ⏳ |
| 8 | Explainability (SHAP + permutation importance) | ⏳ |
| 9 | Customer profile engine | ⏳ |
| 10 | Technical report | ⏳ |

## Quickstart

Requires **Python 3.11 or 3.12**.

```bash
# 1. environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

# 2. data: download "Online Retail II" from the UCI ML Repository (dataset id 502),
#    unzip, and place online_retail_II.xlsx in data/raw/

# 3. build
python -m src.ingest              # xlsx -> data/interim/transactions.parquet
python -m src.build_warehouse     # DuckDB warehouse + reports/data_quality.md
pytest                            # integrity tests
```

## Architecture

```
xlsx ──ingest──▶ raw.transactions ──01_staging──▶ staging.lines (typed + flagged, no rows dropped)
                                                     │
                                   02_data_quality ──┴──▶ reports/data_quality.md
                                                     │
                                  03_core_model ─────▶ core: fact_line · fact_invoice ·
                                                             dim_customer · dim_product · dim_date
                                                     │
                     (next) marts: customer_features(cutoff) · labels(cutoff, horizon)
                                                     │
                     src/: preprocess → segment → train → explain → profile_engine
```

**Design principles**
- All transformation logic lives in `sql/`; Python orchestrates and models.
- Staging flags problems instead of deleting rows, so every exclusion is auditable
  (`exclusion_reason`).
- Full-history dimensions are descriptive only. Model features come from a
  point-in-time SQL macro, which prevents future data leaking into training.

## Project structure

```
config/      config.yaml — paths, seeds, cutoff dates
sql/         01_staging · 02_data_quality · 03_core_model · checks/ · analytics/
src/         config · db · ingest · build_warehouse (+ modelling modules to come)
notebooks/   demo notebooks (call src/, contain no core logic)
models/      feature_schema.json · segments.json · metrics.json (joblib files gitignored)
reports/     data_quality.md · feature_dictionary.md · technical_report.md
tests/       data-model integrity, SQL utils (leakage tests to come)
data/        gitignored
```

## Data and licence

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D — licensed under **CC BY 4.0**.

Customer IDs are anonymised; the dataset contains no personal data. No external
data is joined and no re-identification is attempted.

## Commit convention

`type(scope): summary` — types: `feat`, `fix`, `data`, `test`, `docs`, `refactor`, `chore`.
