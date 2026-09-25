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
| 3 | SQL analytics showcase (10 business questions) | ✅ |
| 4 | Point-in-time feature & label macros + leakage tests | ⏳ |
| 5 | EDA + feature dictionary + segmentation | ⏳ |
| 6 | Purchase-propensity model (baselines → LightGBM, calibration) | ⏳ |
| 7 | Explainability (SHAP + permutation importance) | ⏳ |
| 8 | Customer profile engine | ⏳ |
| 9 | Technical report | ⏳ |

## Quickstart

Requires **Python 3.11 or 3.12**.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# data: download "Online Retail II" from the UCI ML Repository (dataset id 502),
#       unzip, and place online_retail_II.xlsx in data/raw/

python -m src.ingest              # xlsx -> data/interim/transactions.parquet
python -m src.build_warehouse     # DuckDB warehouse + reports/data_quality.md
python -m src.analytics           # 10 SQL business queries -> reports/sql_analytics.md
pytest                            # integrity + analytics tests
```

## Key data facts (from `reports/data_quality.md`)

- 1,067,371 raw lines → 1,021,128 clean lines (95.7% kept); every exclusion has a recorded reason.
- 22,523 duplicate lines came from the two workbook sheets overlapping (1–9 Dec 2010) and were removed.
- 22.8% of lines have no customer ID (15.4% of sales value) — analysed for revenue, excluded from profiles.
- **72.4% of 5,852 identified customers purchased more than once** → repeat behaviour supports a
  "purchase in the next 90 days" prediction target.
- Net revenue reconciles exactly across staging → line fact → invoice fact (£18,926,266.18).

## Architecture

```
xlsx ──ingest──▶ raw.transactions ──01_staging──▶ staging.lines (typed + flagged, no rows dropped)
                                                     │
                                   02_data_quality ──┴──▶ reports/data_quality.md
                                                     │
                                  03_core_model ─────▶ core: fact_line · fact_invoice ·
                                                             dim_customer · dim_product · dim_date
                                                     │
                   sql/analytics/q01–q10 ────────────┤──▶ notebooks/01_sql_showcase.ipynb
                                                     │
                     (next) marts: customer_features(cutoff) · labels(cutoff, horizon)
                                                     │
                     src/: preprocess → segment → train → explain → profile_engine
```

**Design principles**
- All transformation logic lives in `sql/`; Python orchestrates and models.
- Staging flags problems instead of deleting rows, so every exclusion is auditable.
- Full-history dimensions are descriptive only. Model features come from a point-in-time
  SQL macro, which prevents future data leaking into training.

## SQL analytics showcase

Notebook: [`notebooks/01_sql_showcase.ipynb`](notebooks/01_sql_showcase.ipynb) ·
Results: [`reports/sql_analytics.md`](reports/sql_analytics.md)

| Query | Business question | Key techniques |
|---|---|---|
| Q01 | Top customers and revenue concentration | JOIN, RANK, SUM OVER, running share |
| Q02 | Monthly revenue, orders, active customers, growth | DATE_TRUNC, LAG (MoM/YoY), moving AVG |
| Q03 | Weekly revenue and best weeks per year | dim_date JOIN, LAG, DENSE_RANK PARTITION BY |
| Q04 | Order timing by weekday and time of day | CASE bands, share via SUM(COUNT) OVER |
| Q05 | RFM segments (rule-based baseline) | NTILE, CASE, HAVING, date math |
| Q06 | Repeat-purchase rate by market and quarter | ROW_NUMBER, LEFT JOIN, HAVING |
| Q07 | Spend trend per customer | LAG, LEAD, running SUM, rolling AVG |
| Q08 | Top products within each category | ROW_NUMBER vs RANK vs DENSE_RANK |
| Q09 | Monthly cohort retention | cohort CTEs, date_diff, FIRST_VALUE |
| Q10 | Lapsed / at-risk customers vs their own rhythm | LAG gaps, median, CASE rules |

## Project structure

```
config/      config.yaml — paths, seeds, cutoff dates
sql/         01_staging · 02_data_quality · 03_core_model · checks/ · analytics/q01–q10
src/         config · db · ingest · build_warehouse · analytics (+ modelling modules to come)
notebooks/   01_sql_showcase (calls src/, contains no core logic)
models/      feature_schema.json · segments.json · metrics.json (joblib files gitignored)
reports/     data_quality.md · sql_analytics.md · figures/
tests/       data-model integrity, analytics consistency, SQL utils
data/        gitignored
```

## Data and licence

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D — licensed under **CC BY 4.0**.

Customer IDs are anonymised; the dataset contains no personal data. No external
data is joined and no re-identification is attempted.

## Commit convention

`type(scope): summary` — types: `feat`, `fix`, `data`, `test`, `docs`, `refactor`, `chore`.
