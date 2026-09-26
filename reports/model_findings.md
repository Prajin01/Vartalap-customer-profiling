# Step 6 - Findings and Honest Interpretation

Companion to the auto-generated `model_report.md` (numbers below are copied from it).
All choices were fixed on validation **before** test was scored; nothing below was changed after
seeing test. Where test exposed a weakness, it is reported here rather than tuned away.

## 1. The score ranks customers well

| Test (cutoff 2011-09-01, n = 4,324, prevalence 49.5%) | Recency rule | Logistic regression | LightGBM (chosen) |
| --- | --- | --- | --- |
| ROC-AUC | 0.708 | 0.770 | 0.767 (95% CI 0.751-0.779) |
| PR-AUC | 0.680 | 0.791 | 0.789 (95% CI 0.772-0.803) |
| Precision in top 20% | 0.757 | 0.874 | 0.880 |
| Lift in top 20% | 1.53 | 1.76 | 1.78 |

- Both learned models clearly beat the "bought in the last 170 days" rule.
- The top-scored 20% of active customers bought at 88% vs 49.5% overall. At 49.5% prevalence
  the maximum possible lift is about 2.0, so 1.78 uses most of the available headroom.
- Validation lift was higher (2.19) partly because valid prevalence was lower (36.6%), which
  raises the lift ceiling - the two lifts are not directly comparable.

## 2. LightGBM and logistic regression are statistically equivalent

LightGBM was chosen by a pre-set rule (beat logistic regression on valid ROC-AUC by >= 0.005;
valid 0.814 vs 0.806). On test the order flipped (0.767 vs 0.770), and 0.770 lies inside
LightGBM's 95% interval. **Conclusion: the tree model adds no measurable ranking gain on this
data.** Most signal is captured by monotone relationships (recency, frequency, rhythm) that a
linear model on log-scaled features already represents.

Lesson: the 0.005 margin was smaller than the test uncertainty (interval width ~0.03). A
margin tied to bootstrap uncertainty would have selected the simpler model. LightGBM is kept
because the rule was fixed in advance and it enables per-customer SHAP explanations in Step 7;
logistic-regression coefficients are reported alongside as a cross-check.

## 3. Seasonality: the calendar feature has the right direction, not the right size

| Test | mean predicted | prevalence | ECE | ROC-AUC |
| --- | --- | --- | --- | --- |
| LightGBM with calendar feature | 0.599 | 0.495 | 0.106 | 0.767 |
| LightGBM without calendar feature | 0.410 | 0.495 | 0.085 | 0.761 |

- Without the feature the model under-predicts peak season by ~8.5 points; with it, it
  over-predicts by ~10.4 points.
- Cause: the feature learns the size of the peak from a single season (Sep-Nov 2010, 58%
  positive). The 2011 peak was weaker (49.5%). One season cannot estimate year-to-year variation.
- Ranking is essentially unaffected (0.767 vs 0.761); the feature mainly shifts the probability
  level.
- **Impact on the product:** current profile scores use cutoff 2011-12-10, where the peak share
  is 0 - the off-season setting validation confirmed as well calibrated (valid ECE 0.024, mean
  predicted 0.355 vs 0.366). Peak-season miscalibration does not reach the current profiles.
- With more history: estimate seasonality across several years, or recalibrate the base rate
  at the start of each season using the first weeks of data.

## 4. The yes/no flag is not usable in peak season

The threshold (0.340) was chosen to maximise F1 on validation. On test it flags 91% of customers
(3,944 of 4,324), with 1,872 false positives against 310 true negatives (false-positive rate
0.85). Two causes: F1 favours recall at moderate prevalence, and the seasonal over-prediction
pushed most scores above the threshold.

**Decision:** the profile engine and campaign guidance use **rank-based targeting**
(propensity decile, top-20% list), which is robust to base-rate shifts. The binary flag is kept
in `marts.customer_propensity` for transparency but is not recommended for targeting.
A better threshold rule would come from business costs (cost of a contact vs value of a
reactivated customer) or a fixed contact budget, not from F1.

## 5. Subgroup audit (country used for audit only)

| Test | n | prevalence | mean predicted | ROC-AUC | recall | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| UK | 3,937 | 0.493 | 0.597 | 0.765 | 0.965 | 0.854 |
| International | 387 | 0.525 | 0.621 | 0.777 | 0.985 | 0.902 |

Ranking quality and over-prediction are similar for both groups (over-prediction +10.4 vs +9.6
points). No evidence that the score serves one group worse. The International group is small, so
its estimates are uncertain.

## What this means for "purchasing decisions"

The model shows that future purchasing is strongly **associated** with how recently and how
regularly a customer has bought, and that it can **prioritise** whom to contact. It does not
show that contacting anyone changes their behaviour - that would need a controlled experiment
(e.g. a randomised hold-out group in a campaign).
