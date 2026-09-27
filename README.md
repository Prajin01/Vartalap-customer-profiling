# Customer Behaviour & Profiling — ML Proof of Concept

A reproducible customer profiling system built on the public **UCI Online Retail II** dataset:
a DuckDB SQL data model and point-in-time feature tables, behavioural segmentation, a leakage-safe
90-day purchase-propensity model, SHAP explanations, and a `get_profile(customer_id)` engine that
returns a layered, explainable profile for any active customer.

> Built as the Task 2 proof-of-concept for the Vartalap AI — AI/ML Junior Intern selection.

## Read this repo in 5 minutes

| If you want… | Open |
| --- | --- |
| the whole story: decisions, results, what went wrong, what I learned | [`reports/technical_report.md`](reports/technical_report.md) |
| what the system produces for a customer | [`reports/example_profiles.md`](reports/example_profiles.md) |
| model metrics, calibration, fairness audit | [`reports/model_report.md`](reports/model_report.md) · honest interpretation: [`reports/model_findings.md`](reports/model_findings.md) |
| what drives the predictions | [`reports/explainability.md`](reports/explainability.md) |
| the SQL work | [`notebooks/01_sql_showcase.ipynb`](notebooks/01_sql_showcase.ipynb) · [`sql/`](sql/) |
| how this scales to a many-trait profile, responsibly | [`reports/trait_taxonomy.md`](reports/trait_taxonomy.md) |

## Status

| Step | Component | Output | Status |
| --- | --- | --- | --- |
| 1 | Ingest + data-quality report | `reports/data_quality.md` | ✅ |
| 2 | SQL staging + star schema + integrity tests | `sql/01–03`, DuckDB `core.*` | ✅ |
| 3 | SQL analytics showcase (10 business questions) | `notebooks/01_sql_showcase.ipynb` | ✅ |
| 4 | Point-in-time feature & label SQL macros + leakage test | `sql/04–05`, `reports/feature_dictionary.md` | ✅ |
| 5 | Segmentation (K-Means vs GMM, stability) | `reports/segmentation.md`, `notebooks/02` | ✅ |
| 6 | Purchase-propensity model (baselines → LightGBM) | `reports/model_report.md`, `reports/model_findings.md` | ✅ |
| 7 | Explainability (TreeSHAP + permutation importance) | `reports/explainability.md`, `notebooks/03` | ✅ |
| 8 | Customer profile engine (L1–L5 layers) + web app | `src/profile_engine.py`, `app/streamlit_app.py`, `notebooks/04` | ✅ |
| 9 | Technical report, trait taxonomy | `reports/` | ✅ |

64 automated tests (`pytest`) cover data integrity, SQL analytics, feature leakage, split windows,
SHAP additivity, driver tables, the profile engine and a smoke test of the web app.

## Results at a glance

| Question | Answer (held-out test unless stated) |
| --- | --- |
| Can we rank who will buy in the next 90 days? | ROC-AUC **0.767** (95% CI 0.751–0.779), PR-AUC 0.789 vs 0.495 base rate |
| Is it useful for targeting? | Top 20% of scores: **88%** bought vs 49.5% overall (lift 1.78) |
| Better than a simple rule? | Yes — "bought in last 170 days" rule: ROC-AUC 0.708, top-20% precision 0.757 |
| LightGBM vs logistic regression? | **Statistically tied** on test (0.767 vs 0.770) — reported honestly |
| What is purchasing associated with? | Consistent recent activity: active months in last 12, purchase rhythm, recent spend, recency |
| Customer segments | 5 named segments; *Core high-value repeat accounts* = 25% of customers, **72% of revenue** |
| Known weaknesses | Peak-season over-prediction (+10 pts); yes/no flag unusable in peak season → targeting uses deciles. See [`model_findings.md`](reports/model_findings.md) |

## Key decisions (full log in the technical report)

| Decision | Why |
| --- | --- |
| Online Retail II over Olist | 72% repeat buyers make a behavioural purchase target possible; Olist customers mostly buy once |
| SQL table macro for features | one leakage-tested definition for training and scoring |
| Time-based split, test scored once | honest estimate of future performance |
| K-Means k = 5 | beat GMM at every k; stable (ARI 0.88); every segment implies a different action |
| Baselines before LightGBM | the model must beat a recency rule; it does, and it ties logistic regression (reported) |
| No resampling or class weights | classes are 33–58% positive; protects calibrated probabilities |
| Deciles, not a yes/no flag, for targeting | the F1 threshold broke under a peak-season base-rate shift |
| Behaviour only, `country` for audit only | no demographic or psychological inference; fairness checked |

## Try the profile engine

```text
python -m src.profile_engine --examples       # example customer IDs per segment
python -m src.profile_engine 16983            # full layered profile
python -m src.profile_engine 16983 --json     # same, as JSON (API / CRM shape)
```

```text
CUSTOMER 16983  -  profile as of 2011-12-10
L1  Account            first purchase, last purchase, tenure, market (descriptive only)
L2  Purchasing         orders, spend, order value, basket, range, re-ordering, cancellations  [+ percentiles]
L3  Timing & trends    recency, rhythm, overdue ratio, active months, spend trend, seasonality
L4  Model outputs      segment: Core high-value repeat accounts | propensity 75% - decile 9 of 10
L5  Why (associations) + active months in the last 12 (+0.52)  + irregularity of order gaps (+0.23) ...
Suggested action       No incentive needed ... (rule, reason and caveat shown)
```

Full examples, one per segment: [`reports/example_profiles.md`](reports/example_profiles.md).

## Web app — Customer Profiling Studio

```text
streamlit run app/streamlit_app.py        # opens http://localhost:8501 in your browser
```

| Tab | What it does |
| --- | --- |
| Overview | headline test metrics, segment table (SQL), spend share and propensity deciles by segment |
| Customer profile | search / random / example customers → layered profile, driver chart, suggested action, JSON download |
| Target list | filter by segment, decile and spend percentile → contact list with actions, CSV download, hold-out reminder |
| Model & explainability | metrics for all models with CIs, calibration, seasonality ablation, fairness audit, SHAP charts |

The app is read-only: it reads the warehouse and saved outputs, trains nothing, and runs locally.

## Quickstart (reproduce everything)

Requires **Python 3.11** (tested on Windows 11, Python 3.11.2).

```text
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# data: download "Online Retail II" from the UCI ML Repository (dataset id 502),
#       unzip, and place online_retail_II.xlsx in data/raw/

python run_pipeline.py            # all 8 steps in order (about 5-10 minutes)
python -m pytest -q               # 64 tests
streamlit run app/streamlit_app.py   # web app
```

Or step by step:

```text
python -m src.ingest              # xlsx -> data/interim/transactions.parquet
python -m src.build_warehouse     # DuckDB warehouse + reports/data_quality.md
python -m src.analytics           # 10 SQL business queries -> reports/sql_analytics.md
python -m src.features            # point-in-time features + labels -> marts.model_frame
python -m src.segmentation        # K-Means vs GMM, stability, named segments
python -m src.propensity          # baselines, LightGBM, calibration check, audit -> reports/model_report.md
python -m src.explain             # TreeSHAP, permutation importance, drivers -> reports/explainability.md
python -m src.profile_engine      # profile view + reports/example_profiles.md
```

Notebooks (`notebooks/01`–`04`) read the warehouse and saved outputs; run them after the pipeline.
All parameters (paths, seed, cutoffs, model grid) live in `config/config.yaml`; random seed 42.

## Data and key facts

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5CG6D> — **CC BY 4.0**. A UK online gift-ware retailer, Dec 2009 – Dec 2011;
many customers are wholesalers. Customer IDs are anonymised; no personal data; nothing external is joined;
no re-identification is attempted.

- 1,067,371 raw lines → 1,021,128 clean lines (95.7% kept); every exclusion has a recorded reason.
- 22,523 duplicate lines from the two workbook sheets overlapping (1–9 Dec 2010) were removed.
- 22.8% of lines have no customer ID (15.4% of sales value) — kept for revenue analysis, excluded from profiles.
- 72.4% of 5,852 identified customers purchased more than once → a repeat-purchase target is supported.
- Net revenue reconciles exactly across staging → line fact → invoice fact (£18,926,266.18).

## Architecture

```text
xlsx ──ingest──▶ raw.transactions ──01_staging──▶ staging.lines (typed + flagged, no rows dropped)
                                                     │
                                   02_data_quality ──┴──▶ reports/data_quality.md
                                                     │
                                  03_core_model ─────▶ core: fact_line · fact_invoice ·
                                                             dim_customer · dim_product · dim_date
                   sql/analytics/q01–q10 ────────────┤──▶ notebooks/01_sql_showcase.ipynb
        04/05 macros ─▶ customer_features(cutoff) · purchase_labels(cutoff, 90)
                          → marts.model_frame (train/valid/test) · marts.customer_snapshot_current
                                                     │
      src/segmentation ─▶ marts.customer_segments    │
      src/propensity   ─▶ marts.customer_propensity  │
      src/explain      ─▶ marts.customer_drivers · marts.customer_driver_families
                                                     │
      06_profile_view  ─▶ marts.customer_profile_base ─▶ src/profile_engine.get_profile(customer_id)
```

**Design principles.** Transformation logic lives in SQL; Python orchestrates and models. Staging flags
problems instead of deleting rows, so every exclusion is auditable. Model features come only from a
point-in-time SQL macro, which prevents future data leaking into training.

## Leakage-safe prediction design

**Target:** will an active customer (≥1 purchase in the previous 365 days) place at least one order in the
next **90 days**?

| Macro | Reads | Returns |
| --- | --- | --- |
| `customer_features(cutoff)` | only rows with `invoice_date < cutoff` | 33 behavioural features |
| `purchase_labels(cutoff, 90)` | only rows with `cutoff ≤ invoice_date < cutoff + 90d` | the label |

```text
train: 10 monthly cutoffs Jun-2010 … Mar-2011  (36,446 rows; label windows end ≤ 30 May 2011)
valid: cutoff 1 Jun 2011   (4,324 customers, 36.6% positive) -> all tuning, model choice, threshold
test : cutoff 1 Sep 2011   (4,324 customers, 49.5% positive) -> scored once
```

`tests/test_features.py` deletes every row on/after a cutoff, rebuilds the macro and requires identical
features — proof that no future data is used. A calendar feature (share of the forecast window in the
Sep–Nov peak) is known at the cutoff and is not leakage; its effect is measured by an ablation.

## Segmentation

12 interpretable behavioural features → log1p → winsorise → standardise. K-Means and Gaussian Mixture
Models compared for k = 2–8 on silhouette, Davies-Bouldin, Calinski-Harabasz, BIC and bootstrap stability
(ARI); HDBSCAN as a density diagnostic (83.9% noise → no natural dense clusters). **K-Means k = 5**
(silhouette 0.20, Davies-Bouldin 1.47, stability ARI 0.88), named from measured profiles:

| Segment | Customers | Revenue | Defining behaviour |
| --- | --- | --- | --- |
| Core high-value repeat accounts | 25.1% | 72.0% | ~7 orders/year, re-order 41% of products |
| Recent seasonal buyers | 31.8% | 12.3% | 66% of spend in Oct–Dec |
| Lapsed buyers | 29.0% | 8.6% | no order for ~173 days |
| Low-spend occasional buyers | 11.4% | 2.5% | few, small orders |
| High-cancellation accounts | 2.7% | 4.6% | 24% of spend value cancelled |

## Responsible AI

- **Behaviour, not psychology.** Profiles contain observed purchasing behaviour and model-estimated
  propensities only — no demographics, no personality or emotional labels.
- **Association, not causation.** Drivers explain what the model associates with a score. Action suggestions
  carry a caveat: measure impact with a randomised hold-out group.
- **Fairness.** `country` is excluded from all models and used only for a UK vs International audit
  (similar ranking quality and error rates; see `reports/model_report.md`).
- **Honest evaluation.** Weaknesses found on the test set are documented in
  [`reports/model_findings.md`](reports/model_findings.md), not tuned away.

## SQL showcase

Notebook: [`notebooks/01_sql_showcase.ipynb`](notebooks/01_sql_showcase.ipynb) · results: [`reports/sql_analytics.md`](reports/sql_analytics.md)

| Query | Business question | Key techniques |
| --- | --- | --- |
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

SQL also builds the ML feature table (`sql/04_features_macro.sql`), the labels (`sql/05_labels_macro.sql`)
and the profile view (`sql/06_profile_view.sql`: PERCENT_RANK, STRING_AGG … FILTER, multi-table joins).

## Project structure

```text
config/      config.yaml — paths, seed, cutoffs, model settings
sql/         01_staging · 02_data_quality · 03_core_model · 04_features_macro · 05_labels_macro ·
             06_profile_view · checks/ · analytics/q01–q10
src/         config · db · ingest · build_warehouse · analytics · features · preprocess ·
             segmentation · propensity · explain · profile_engine
notebooks/   01_sql_showcase · 02_eda_segmentation · 03_model_explainability · 04_profile_engine
models/      feature_schema.json · segments.json · metrics.json · explainability.json (joblib files gitignored)
reports/     technical_report · trait_taxonomy · data_quality · sql_analytics · feature_dictionary ·
             segmentation · model_report · model_findings · explainability · example_profiles · figures/
tests/       integrity, analytics, leakage, split windows, SHAP additivity, profile engine
app/         streamlit_app.py — Customer Profiling Studio (web UI)
run_pipeline.py   runs all steps in order
data/        gitignored
```

## Commit convention

`type(scope): summary` — types: `feat`, `fix`, `data`, `test`, `docs`, `refactor`, `chore`.
