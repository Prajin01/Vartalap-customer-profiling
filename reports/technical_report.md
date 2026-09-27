# Technical Report — Customer Behaviour & Profiling POC

**Vartalap AI · AI/ML Junior Intern selection · Task 2**
Every number in this report is produced by the pipeline and can be traced to a file in `reports/` or
`models/*.json` (see Appendix A). Nothing is estimated by hand.

---

## Executive summary

- **What was built:** an end-to-end, reproducible customer profiling system on public data — SQL warehouse
  → 33 point-in-time behavioural features → 5 named segments → a 90-day purchase-propensity model →
  per-customer explanations → a `get_profile(customer_id)` engine with suggested, testable actions.
- **Headline result:** on a held-out peak-season window the model ranks buyers well (ROC-AUC 0.767,
  95% CI 0.751–0.779); the top 20% of scored customers bought at **88%** vs a 49.5% base rate, clearly above a
  recency-rule baseline (75.7%).
- **Segments:** *Core high-value repeat accounts* are 25% of customers but **72% of revenue**.
- **What purchasing is associated with:** consistent, recent activity — active months in the last year, a short
  and regular purchase rhythm, recent spend. Time of day and price level carry almost no signal.
- **Honest findings:** logistic regression ties LightGBM; the seasonal adjustment overshoots in peak season; an
  F1-chosen yes/no threshold is unusable in peak season. All three are documented, none were tuned away on test,
  and the product design (decile-based targeting) accounts for them.
- **Engineering:** SQL does the data work (staging, star schema, analytics, feature and label macros, profile
  view); one command rebuilds everything; 63 automated tests, including a leakage test.

---

## 1. Problem, interpretation and objectives

> *"Build an ML-based customer profiling and behavioural analysis system using relevant datasets, identify
> meaningful customer characteristics and behavioural patterns, segment customers based on those
> characteristics, and analyze the key factors associated with behaviours such as purchasing or selling decisions."*

| Assignment phrase | Interpretation in this POC | Section |
| --- | --- | --- |
| profiling system | a layered profile (L1–L5) for any customer, served by a function and a CLI | §9 |
| characteristics & behavioural patterns | 33 measured behavioural features in 11 families, plus SQL analytics | §4, §5 |
| segment customers | K-Means vs GMM with stability testing; 5 named segments | §6 |
| key factors associated with purchasing | propensity model + SHAP and permutation importance, worded as associations | §7, §8 |
| selling decisions | the dataset has one seller, so seller behaviour cannot be modelled; the POC supports the **retailer's** selling decisions (whom to prioritise, how) | §9 |
| core SQL fundamentals | all data transformation, analytics and feature tables in SQL, consumed by Python | §4 |

**Success criteria set before modelling:** (1) no leakage, proven by a test; (2) the model must beat a simple
rule on held-out data; (3) every profile field must be measurable from the data; (4) no protected attribute or
proxy used as a model input; (5) one-command reproducibility.

---

## 2. Dataset selection

| Criterion | UCI Online Retail II | Olist (Brazil) | dunnhumby Complete Journey | REES46 event logs |
| --- | --- | --- | --- | --- |
| Repeat behaviour (needed for a purchase target) | **High** (72.4% repeat) | Low (vast majority buy once) | High | Medium |
| Temporal depth | 2 years, timestamped | 2 years | 2 years | months, very granular |
| Relational richness for SQL | 1 table → modelled into a star schema | **9 tables** | several tables | 1 very large table |
| Engagement / campaign data | none | reviews | campaigns, coupons | views, carts |
| Licence / access | **CC BY 4.0**, direct download | CC BY-NC-SA 4.0 | registration, custom terms | terms to check |
| Feasible in one week | **Yes** (~1M lines) | Yes | Yes | needs heavy sampling |

**Choice: UCI Online Retail II.** A repeat-purchase target is the backbone of behavioural profiling, and 72.4%
of identified customers repeat. Olist's relational richness is attractive for SQL, but with most customers buying
once, a behavioural model would mostly learn "one-time buyer". The single-table weakness was turned into an
opportunity: the star schema (§4) is built in SQL. **Trade-off accepted:** no web engagement or campaign data, so
engagement and promotion-response traits are documented as future, consent-based additions (`trait_taxonomy.md`).

---

## 3. Data quality and cleaning

Source: Chen, D. (2012), *Online Retail II*, UCI ML Repository, CC BY 4.0. UK online gift-ware retailer,
1 Dec 2009 – 9 Dec 2011; many accounts are wholesalers.

| Fact | Value |
| --- | --- |
| Raw lines (two workbook sheets) | 1,067,371 |
| Clean lines kept | 1,021,128 (95.7%) |
| Duplicate lines from overlapping sheets (1–9 Dec 2010) | 22,523 removed |
| Lines without customer ID | 22.8% of lines, 15.4% of sales value |
| Identified customers / share who repeat | 5,852 / 72.4% |
| Net revenue, reconciled across staging → line fact → invoice fact | £18,926,266.18 |

**Approach.** Staging keeps every row and adds typed columns, quality flags and an `exclusion_reason`, so every
exclusion is auditable and counted in `reports/data_quality.md`. Anonymous lines stay in revenue analysis but are
excluded from profiles (a profile needs an identity). Cancellations (invoice numbers starting with `C`) are kept as
their own behaviour — cancellation rate and cancelled-value share — instead of being silently netted off.

---

## 4. Architecture and SQL

**Why DuckDB:** embedded (no server, so an evaluator reproduces with `pip install`), columnar and fast on ~1M rows,
full analytical SQL with window functions, CTEs and **table macros** (parameterised SQL functions), which made the
point-in-time feature table possible in pure SQL. Alternatives: PostgreSQL (needs a server), SQLite (weaker
analytical SQL), pandas only (would hide the SQL the evaluators asked to see).

```text
raw ─▶ staging.lines (typed, flagged) ─▶ core: fact_line · fact_invoice · dim_customer · dim_product · dim_date
     ─▶ marts: model_frame · customer_snapshot_current · customer_segments · customer_propensity ·
               customer_drivers · customer_driver_families · customer_profile_base (view)
```

| SQL file | Role |
| --- | --- |
| `01_staging.sql` | typing, flags, exclusion reasons, duplicate detection |
| `02_data_quality.sql` | counts and reconciliations for the data-quality report |
| `03_core_model.sql` | star schema; heuristic product groups |
| `analytics/q01–q10` | 10 business questions: top customers, monthly and weekly revenue, order timing, RFM, repeat rate, spend trend per customer, rank within category, cohort retention, lapsed customers |
| `04_features_macro.sql` | `customer_features(cutoff)` — 33 features from rows strictly before the cutoff |
| `05_labels_macro.sql` | `purchase_labels(cutoff, horizon)` — the future label window |
| `06_profile_view.sql` | the profile view: multi-table joins, `PERCENT_RANK`, `STRING_AGG … FILTER` |

SQL techniques used across these files: `SELECT / WHERE / GROUP BY / HAVING`, INNER and LEFT joins, `CASE WHEN`,
date arithmetic, monthly and weekly aggregation, CTEs, `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `NTILE`, `LAG`, `LEAD`,
`FIRST_VALUE`, `SUM / AVG … OVER (PARTITION BY … ORDER BY …)`, `PERCENT_RANK`, `MEDIAN`, ordered and filtered
aggregates, and table macros. Python consumes these tables directly: the model frame, the scoring snapshot and the
profile view are all SQL outputs.

---

## 5. Feature engineering — the customer representation

All model features come from **one** SQL macro, `customer_features(cutoff)`, which reads only rows with
`invoice_date < cutoff`. The same macro produces training rows (past cutoffs) and scoring rows (the current
snapshot), so training and serving cannot drift apart. The full dictionary (definition, SQL CTE, rationale) is in
`reports/feature_dictionary.md`.

| Family | Features | Examples |
| --- | --- | --- |
| RFM | 5 | days since last and first purchase; orders all-time / 12 months / 90 days |
| Monetary | 3 | spend all-time / 12 months / 90 days |
| Basket | 5 | average and median order value, products per order, units per order, quantity per line |
| Price | 2 | average unit price, price paid vs each product's typical price |
| Diversity | 4 | distinct products and product groups, spend entropy across groups, main-group share |
| Seasonality | 2 | Christmas-product share, Oct–Dec spend share |
| Loyalty | 1 | share of lines re-ordering a previously bought product |
| Cancellations | 2 | cancellation rate, cancelled value share |
| Rhythm | 4 | mean and median gap between orders, gap irregularity, overdue ratio |
| Trend / consistency | 4 | 90-day spend and order trends, 12-month slope, active months in the last 12 |
| Timing | 1 | share of orders placed before noon |
| **Total** | **33** | by layer: 1 L1 (raw attribute) · 19 L2 (aggregated behaviour) · 13 L3 (temporal) |

**Design choices.** Gross spend and cancellations are separate features, so cancelling is not hidden inside net
revenue. Trend ratios use a symmetric, zero-safe log ratio. Price paid is compared with each product's median price
**before the cutoff** (point-in-time, no leakage). Rhythm features are left `NULL` when a customer has too few order
days — "no rhythm yet" is information the tree model can use, and it is better than an invented value.
**The count was set by what the data supports, not by a target number.**

**Leakage proof.** A test deletes every transaction on or after a cutoff, rebuilds the macro and requires identical
features. It passes.

---

## 6. Segmentation

**Input:** 12 interpretable features (chosen for business readability), log1p on skewed features → winsorise at
1%/99% → standardise (`src/preprocess.py`; all statistics learned in `fit` only).

**Methods compared for k = 2…8:** K-Means and Gaussian Mixture Models, scored on silhouette, Davies-Bouldin,
Calinski-Harabasz, BIC (GMM) and bootstrap stability (adjusted Rand index between refits on resamples). HDBSCAN was
run as a density diagnostic.

| Finding | Evidence | Decision |
| --- | --- | --- |
| K-Means beats GMM | better at every k on the separation metrics | K-Means |
| No natural dense clusters | HDBSCAN labels 83.9% of customers as noise | segments are useful partitions of a continuum, not natural groups |
| k = 5 | silhouette 0.20, Davies-Bouldin 1.47 (best for k ≥ 3), stability ARI 0.88 | each segment implies a different action |

| Segment (named from its measured profile) | Customers | Revenue | Defining behaviour |
| --- | --- | --- | --- |
| Core high-value repeat accounts | 25.1% | 72.0% | ~7 orders/year, re-order 41% of products |
| Recent seasonal buyers | 31.8% | 12.3% | 66% of spend in Oct–Dec |
| Lapsed buyers | 29.0% | 8.6% | no order for ~173 days |
| Low-spend occasional buyers | 11.4% | 2.5% | few, small orders |
| High-cancellation accounts | 2.7% | 4.6% | 24% of spend value cancelled |

**Trade-offs made explicitly.** Silhouette 0.20 is modest, as expected for continuous behaviour; k = 5 was kept for
stability and actionability. The High-cancellation segment is below my own 3% minimum-size rule but was kept on
purpose: it is stable and behaviourally distinct. **A hypothesis that failed:** I expected high-value accounts to
split into wholesalers and retailers; the data did not support it.

---

## 7. Predictive model — 90-day purchase propensity

### 7.1 Target and windows

**Target:** `purchased_in_horizon` = at least one purchase in `[cutoff, cutoff + 90 days)`, for customers with at
least one purchase in the 365 days before the cutoff. The data supports it: positive rates of 33–58% per cutoff.

| Split | Cutoffs | Rows | Positive rate | Used for |
| --- | --- | --- | --- | --- |
| Train | 10 monthly, 1 Jun 2010 – 1 Mar 2011 | 36,446 | 33.1–58.3% by cutoff | fitting |
| Valid | 1 Jun 2011 (window Jun–Aug) | 4,324 | 36.6% | hyper-parameters, model choice, calibration decision, threshold |
| Test | 1 Sep 2011 (window Sep–Nov) | 4,324 | 49.5% | **scored once** |

Every training label window ends (30 May 2011) before the validation cutoff, and the validation window ends before
the test cutoff — checked in code and in a test. Stacking monthly cutoffs gives more training examples and covers a
full seasonal cycle; the cost (the same customer appears at several cutoffs) is acknowledged and does not leak into
later splits.

**Seasonality found in the labels:** training windows covering Sep–Nov had the highest positive rate (58.3% for the
Sep 2010 cutoff); windows containing December were lower because the retailer shuts down over Christmas. The test
window is peak season and validation is not. A calendar feature — the share of the forecast window falling in
Sep–Nov — was added. It is known at the cutoff (it is only the calendar), so it is not leakage, and the months were
chosen from training windows only.

### 7.2 Models and selection

| Step | Choice | Why |
| --- | --- | --- |
| Baseline 1 | recency rule "bought in the last *t* days", *t* tuned on valid → 170 days | the rule a business would use without ML |
| Baseline 2 | logistic regression on log / winsorised / scaled features, C ∈ {0.01, 0.1, 1} → C = 1 | strong, interpretable reference |
| Main model | LightGBM, 6 configurations (leaves × minimum child samples), early stopping on valid | non-linearities; native missing values; exact TreeSHAP |
| Selection metric | valid ROC-AUC | independent of prevalence, which differs between valid and test |
| Selection rule (fixed in advance) | keep LightGBM only if it beats logistic regression by ≥ 0.005 | no complexity without a gain |
| Class imbalance | no weights, no resampling | positive rate 33–58%; weights or resampling would distort probabilities |
| Calibration | Platt scaling, applied only if it improves 5-fold CV Brier on valid | 0.1614 (Platt) vs 0.1619 (raw) → **not applied** |
| Threshold | maximises F1 on valid → 0.340 | standard choice; see finding 3 |

Validation ROC-AUC: LightGBM 0.814 (all six configurations 0.812–0.814, so the choice is robust), logistic
regression 0.806, recency rule 0.746. LightGBM passed the rule. Validation calibration was good: ECE 0.024, mean
predicted 0.355 vs 0.366 observed.

### 7.3 Results (held-out test, n = 4,324, prevalence 49.5%)

| Metric | Recency rule | Logistic regression | LightGBM (chosen) |
| --- | --- | --- | --- |
| ROC-AUC | 0.708 | 0.770 | **0.767** (95% CI 0.751–0.779) |
| PR-AUC | 0.680 | 0.791 | **0.789** (95% CI 0.772–0.803) |
| Precision in top 20% | 0.757 | 0.874 | **0.880** |
| Lift in top 20% | 1.53 | 1.76 | **1.78** (maximum possible ≈ 2.0 at this prevalence) |
| Brier / ECE | – | 0.200 / 0.082 | 0.207 / 0.106 |
| Mean predicted vs actual | – | 0.577 vs 0.495 | 0.599 vs 0.495 |
| Precision / recall / F1 at threshold | 0.613 / 0.775 / 0.685 | 0.510 / 0.989 / 0.673 | 0.525 / 0.967 / 0.681 |

Confusion matrix, LightGBM at 0.340: TN 310 · FP 1,872 · FN 70 · TP 2,072.

**Why these metrics.** ROC-AUC measures ranking independent of prevalence. PR-AUC focuses on buyers and is compared
with its random baseline (the prevalence). Precision and lift in the top 20% match how a campaign uses a score.
Precision, recall, F1 and the confusion matrix show what a yes/no flag would do. Brier, ECE and reliability curves
show whether a displayed probability can be trusted — needed because profiles show probabilities to people.
Bootstrap confidence intervals show how much of a difference is noise.

### 7.4 Findings — reported, not tuned away

1. **LightGBM and logistic regression are statistically tied on test.** 0.770 lies inside LightGBM's CI. The pre-set
   margin (0.005) was smaller than test uncertainty (CI width ≈ 0.03); a margin tied to bootstrap uncertainty would
   have chosen the simpler model. LightGBM stays because the rule was fixed in advance and it enables exact
   per-customer SHAP.
2. **Seasonality: right direction, wrong size.** Ablation on test: with the calendar feature the model over-predicts
   (mean 0.599 vs 0.495, ECE 0.106); without it, it under-predicts (0.410, ECE 0.085); ranking barely changes (0.767
   vs 0.761). One peak season in training cannot estimate the next season's strength. **Impact on the product:**
   current scores (cutoff 10 Dec 2011) fall in the off-season setting that validation confirmed as well calibrated.
3. **The F1 threshold over-selects in peak season.** On validation it flagged 38% of customers (precision 0.66,
   recall 0.69); on test it flagged 91% (false-positive rate 0.85). F1 favours recall, and the seasonal
   over-prediction pushed scores above 0.34. **Decision:** targeting uses deciles and a top-20% list, which are
   robust to base-rate shifts; in production the threshold should come from contact cost and customer value.

### 7.5 Fairness audit

`country` is excluded from all models and used only here.

| Test | n | Prevalence | Mean predicted | ROC-AUC | Recall | False-positive rate |
| --- | --- | --- | --- | --- | --- | --- |
| UK | 3,937 | 0.493 | 0.597 | 0.765 | 0.965 | 0.854 |
| International | 387 | 0.525 | 0.621 | 0.777 | 0.985 | 0.902 |

Ranking quality and over-prediction are similar across groups; there is no evidence that the score serves one group
worse. The International estimate is uncertain because the group is small.

---

## 8. Explainability

**Methods.** Exact TreeSHAP computed by LightGBM itself (`pred_contrib=True`, the same trees as prediction;
additivity error 5.8 × 10⁻¹⁵) on the 4,261 profiled customers; permutation importance on validation, per feature
and per family (correlated features shuffled together); logistic-regression coefficients as a cross-check. The
calendar feature is identical for every customer at a cutoff, so it is reported as one seasonal shift and excluded
from the customer driver lists.

| Top features by mean abs SHAP | Mean abs SHAP | Direction (association) | Permutation AUC drop |
| --- | --- | --- | --- |
| active months in the last 12 | 0.307 | more → higher score | 0.0139 |
| average days between orders | 0.142 | longer → lower score | 0.0042 |
| spend in the last 90 days | 0.137 | more → higher score | 0.0047 |
| irregularity of order gaps | 0.121 | more irregular → lower score | 0.0016 |
| days since last purchase | 0.121 | longer → lower score | 0.0099 |

| Family | SHAP share | Permutation AUC drop (whole family shuffled) |
| --- | --- | --- |
| Trend / consistency | 21.3% | 0.0137 |
| Rhythm | 18.4% | 0.0266 |
| RFM | 16.7% | 0.0265 |
| Monetary | 14.9% | 0.0123 |
| Seasonality | 6.1% | 0.0029 |
| Loyalty | 6.0% | 0.0018 |
| Diversity | 4.6% | 0.0007 |
| Basket | 4.5% | 0.0012 |
| Price | 3.2% | 0.0001 |
| Cancellations | 2.9% | 0.0007 |
| Timing | 1.4% | −0.0002 (no signal) |

**Interpretation.** The two methods agree (Spearman rank correlation 0.82). Purchasing is associated with
**consistent, recent, regular activity**. No single family is indispensable (largest drop 0.027) because this signal
is shared across Rhythm, RFM and Trend. Logistic-regression signs agree with SHAP directions for 17 of 27 features;
the disagreements come from correlated features, whose individual coefficients are unstable, so conclusions are drawn
at family level. All of this explains the **model**; none of it shows what would make a customer buy.

---

## 9. Profile engine

`get_profile(customer_id)` reads one SQL view, `marts.customer_profile_base`, and returns a JSON-ready profile:

| Layer | Content |
| --- | --- |
| L1 | first and last purchase, tenure, market (descriptive only, never a model input) |
| L2 | 13 purchasing measures, with percentile context from `PERCENT_RANK` |
| L3 | 10 timing and trend measures |
| L4 | segment, 90-day propensity, decile (the yes/no flag carries a peak-season warning) |
| L5 | top 3 features raising and lowering the score, in plain language, plus family totals |
| Action | a rule-based suggestion, its reason, and a caveat |
| Data notes | how many invoices the profile rests on, whether rhythm is measurable, accounts may be businesses |

**Suggested-action rules**, in priority order — transparent decision support, not an estimate of impact:

| # | Condition | Suggestion |
| --- | --- | --- |
| 1 | High-cancellation segment | review service and cancellation reasons before any marketing |
| 2 | decile 9–10 | no incentive needed (discounts would mostly subsidise likely purchases — an assumption to test) |
| 3 | decile 4–8 and spend in the top 25% | priority retention contact |
| 4 | Lapsed segment, or decile ≤ 3 and silent ≥ 2× the usual gap | low-cost win-back only |
| 5 | Seasonal segment | time contact before the Sep–Nov buying window |
| 6 | otherwise | standard communication |

| Example (typical member of its segment) | Propensity | Decile | Suggested action |
| --- | --- | --- | --- |
| 16983 — Core high-value repeat accounts | 0.747 | 9 | no incentive needed |
| 17708 — Lapsed buyers (silent 2.4× their usual gap) | 0.224 | 4 | low-cost win-back |
| 14034 — High-cancellation accounts | 0.217 | 4 | review service first |

Full profiles, one per segment: `reports/example_profiles.md`. Unknown or inactive IDs return an explicit message.

**Improvements made after reviewing real output.** The first real run exposed three presentation issues, each fixed
and covered by a test: a spend trend of "+27,090%" when the previous period had no spend (now "up — no spend in the
previous 90 days"); "higher than 0% of customers" for tied lowest values (now "in the lowest group"); and a rule gap
where a lapsed customer in decile 4 received no action (rule 4 now covers the Lapsed segment).

---

## 10. Testing and reproducibility

63 tests (`python -m pytest -q`), including:

| Area | Examples of what is checked |
| --- | --- |
| Data and SQL (Steps 1–5) | integrity and reconciliation of the warehouse; analytics queries; segmentation outputs |
| Leakage | features are identical after deleting all post-cutoff data |
| Splits | label windows never overlap the next split; calendar feature values on known dates |
| Model artefacts | the saved model returns probabilities; the score table covers every active customer |
| Explainability | SHAP values add up to the model output; driver signs, limits and calendar exclusion |
| Profile engine | action rules, trend and percentile wording, one row per customer, unknown IDs |

Pinned dependencies (`requirements.txt`), random seed 42, all parameters in `config/config.yaml`, and
`python run_pipeline.py` rebuilds everything in order.

---

## 11. Responsible AI

| Principle | Implementation |
| --- | --- |
| Lawful data | public, CC BY 4.0, anonymised IDs; no scraping, no joins to external data, no re-identification |
| Behaviour, not psychology | profiles contain observed behaviour and model-estimated propensities only; no demographics, personality or emotion labels |
| Fairness | `country` never a model input; used only for the subgroup audit (§7.5) |
| Association vs causation | drivers are "associated with" or "raise the score"; every action recommends a randomised hold-out |
| Transparency | plain-language drivers; profiles designed to be showable to the customer |
| Honest evaluation | test scored once; weaknesses documented in `reports/model_findings.md` |

**Biases to keep in mind.** *Sampling:* one UK retailer, 2009–2011, identified customers only (15% of value is
anonymous and invisible). *Label:* "no purchase in 90 days" can reflect seasonality or a business closing, not
dissatisfaction. *Measurement:* tenure is left-censored at December 2009; product groups are heuristic; accounts may
be businesses, not individuals.

---

## 12. Decision log

| Decision | Alternatives considered | Why this one |
| --- | --- | --- |
| Online Retail II | Olist, dunnhumby, REES46 | repeat behaviour, clear licence, feasible in a week |
| DuckDB | PostgreSQL, SQLite, pandas only | embedded, analytical SQL, table macros |
| Flag instead of delete in staging | drop bad rows | auditable exclusions |
| SQL table macro for features | pandas feature code | one definition for training and scoring; leakage is testable |
| 90-day horizon | 30 or 60 days | typical gap between orders ≈ 50 days, so shorter windows would mostly miss regular buyers; gives a balanced target |
| Stacked monthly training cutoffs | a single cutoff | more data; covers a seasonal cycle |
| K-Means, k = 5 | GMM, HDBSCAN, other k | better metrics, stable, actionable |
| Keep a 2.7% segment | enforce the 3% minimum | stable and distinct |
| LightGBM (pre-set rule) | logistic regression | passed the rule; exact TreeSHAP (the tie is reported) |
| No class weights, no resampling | SMOTE, class weights | classes balanced enough; protects calibration |
| Calendar feature Sep–Nov | Oct–Dec, month dummies | best matches training labels; Christmas shutdown |
| Conditional calibration | always calibrate on valid | valid is off-season; calibrating there could harm peak season |
| Deciles for targeting | the F1 threshold flag | robust to base-rate shifts (finding 3) |
| Calendar excluded from drivers | show it per customer | it is context, not customer behaviour |
| Rule-based actions with caveats | uplift model | no experiment data; honest about causality |

---

## 13. What went wrong, and what I learned

- **A hypothesis failed.** High-value accounts did not split into wholesalers and retailers. Reporting it is part of
  the result.
- **Seasonality was bigger than expected.** It showed up in the positive rate per cutoff before any modelling; I
  handled it with a calendar feature, then measured it with an ablation instead of assuming it worked.
- **Evaluation corrected my expectations twice.** The simpler model tied the complex one, and a standard threshold
  rule broke under a base-rate shift. I learned to set selection margins relative to uncertainty and to choose
  thresholds from costs, not F1.
- **Reading real outputs matters.** The metrics were fine, but reading actual profiles exposed an absurd percentage,
  an awkward "0%" phrase and a rule gap. Each became a fix and a test.
- **Honesty is a design constraint.** Several of the most useful decisions — deciles over flags, conditional
  calibration, excluding the calendar from drivers — came directly from reporting problems instead of hiding them.

---

## 14. Limitations and roadmap

| Limitation | Next step |
| --- | --- |
| One peak season in training | several years of history, or recalibrate the base rate at the start of each season |
| F1-based threshold | cost-based threshold or a fixed contact budget agreed with the business |
| Associations only | randomised hold-out tests of the actions; uplift modelling once experiment data exists |
| No engagement or campaign data | consented first-party events, campaign logs and support conversations (`trait_taxonomy.md`) |
| Batch, local | scheduled pipeline; `get_profile` behind an API; monthly drift and calibration monitoring |
| Heuristic product groups | a proper product taxonomy from the catalogue |

---

## Appendix A — where each claim comes from

| Claim | File |
| --- | --- |
| Data volumes, duplicates, reconciliation | `reports/data_quality.md` |
| Feature definitions | `reports/feature_dictionary.md`, `models/feature_schema.json` |
| Cutoffs and positive rates | `reports/model_frame_summary.md` |
| Segmentation metrics and profiles | `reports/segmentation.md`, `models/segments.json` |
| Model metrics, CIs, ablation, audit, confusion matrix | `reports/model_report.md`, `models/metrics.json` |
| Interpretation of model weaknesses | `reports/model_findings.md` |
| SHAP and permutation results | `reports/explainability.md`, `models/explainability.json` |
| Example profiles | `reports/example_profiles.md` |
