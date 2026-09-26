"""Step 6 - 90-day purchase-propensity model.

Target (built in SQL, Step 4): purchased_in_horizon = >=1 purchase in [cutoff, cutoff + 90d)
for customers with >=1 purchase in the 365 days before the cutoff.

Protocol
  train : stacked monthly cutoffs 2010-06 .. 2011-03   -> fit models
  valid : cutoff 2011-06-01                           -> hyper-parameters, model choice,
                                                         calibration decision, threshold
  test  : cutoff 2011-09-01                           -> evaluated ONCE, nothing tuned on it

Models
  1. Recency rule      "bought in the last t days"        (t tuned on valid)
  2. Logistic regression on log/winsorised/scaled features (C tuned on valid)
  3. LightGBM          native NaN handling, small grid, early stopping on valid

Outputs
  models/propensity.joblib   chosen model + threshold + feature list (gitignored)
  models/metrics.json        every number in the report
  reports/model_report.md    human-readable report
  reports/figures/step6_*.png
  marts.customer_propensity  current-snapshot scores (used by the profile engine)

Run:  python -m src.propensity
"""
from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # silences a joblib/loky warning on Windows

import json
from itertools import product

import joblib
import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.build_warehouse import df_to_markdown
from src.config import load_config, path
from src.db import connect
from src.features import LABEL, MODEL_FEATURES, horizon_days
from src.preprocess import make_numeric_pipeline

CAL_FEATURE = "label_window_peak_share"
FEATURES = MODEL_FEATURES + [CAL_FEATURE]
AUDIT_COLUMN = "country"   # audit only - never a model input


# ----------------------------------------------------------------------------- config
def pcfg() -> dict:
    return load_config()["propensity"]


def seed() -> int:
    return int(load_config()["project"]["random_seed"])


# ----------------------------------------------------------------------------- calendar feature
def peak_share(cutoff, horizon: int, months) -> float:
    """Share of the label window [cutoff, cutoff + horizon) whose days fall in `months`."""
    days = pd.date_range(pd.Timestamp(cutoff), periods=int(horizon), freq="D")
    return float(np.isin(days.month, list(months)).mean())


def add_calendar_feature(df: pd.DataFrame, horizon: int | None = None, months=None) -> pd.DataFrame:
    horizon = horizon or horizon_days()
    months = months or pcfg()["peak_months"]
    cache: dict = {}

    def share(c):
        k = pd.Timestamp(c)
        if k not in cache:
            cache[k] = peak_share(k, horizon, months)
        return cache[k]

    out = df.copy()
    out[CAL_FEATURE] = pd.to_datetime(out["cutoff_date"]).map(share).astype(float)
    return out


# ----------------------------------------------------------------------------- metrics
def precision_at_top(y, s, frac: float) -> float:
    y, s = np.asarray(y), np.asarray(s)
    n = max(1, int(round(frac * len(s))))
    top = np.argsort(-s, kind="mergesort")[:n]
    return float(y[top].mean())


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    ids = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum((ids == b).mean() * abs(y[ids == b].mean() - p[ids == b].mean())
                     for b in range(bins) if (ids == b).any()))


def best_f1_threshold(y, s) -> float:
    prec, rec, thr = precision_recall_curve(y, s)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def threshold_metrics(y, s, t: float) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(s) >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(t),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "selection_rate": float(pred.mean()),
            "false_positive_rate": float(fp / max(fp + tn, 1)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def evaluate(y, s, threshold: float | None = None, is_probability: bool = True,
             top_fraction: float = 0.2) -> dict:
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    prev = float(y.mean())
    p_top = precision_at_top(y, s, top_fraction)
    out = {"n": int(len(y)), "prevalence": prev,
           "roc_auc": float(roc_auc_score(y, s)),
           "pr_auc": float(average_precision_score(y, s)),
           f"precision_at_top{int(top_fraction * 100)}": p_top,
           f"lift_at_top{int(top_fraction * 100)}": p_top / prev if prev else float("nan")}
    if is_probability:
        pc = np.clip(s, 1e-6, 1 - 1e-6)
        out.update({"brier": float(brier_score_loss(y, pc)),
                    "log_loss": float(log_loss(y, pc, labels=[0, 1])),
                    "ece": expected_calibration_error(y, pc),
                    "mean_predicted": float(pc.mean())})
    if threshold is not None:
        out.update(threshold_metrics(y, s, threshold))
    return out


def bootstrap_ci(y, s, reps: int, rng: np.random.Generator) -> dict:
    """95% percentile intervals for ROC-AUC and PR-AUC, resampling customers."""
    y, s = np.asarray(y), np.asarray(s)
    roc, pr = [], []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        roc.append(roc_auc_score(y[i], s[i]))
        pr.append(average_precision_score(y[i], s[i]))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"roc_auc_95ci": q(roc), "pr_auc_95ci": q(pr), "reps": len(roc)}


# ----------------------------------------------------------------------------- models
def fit_logreg(X: pd.DataFrame, y, C: float) -> Pipeline:
    pipe = Pipeline([("prep", make_numeric_pipeline(FEATURES)),
                     ("clf", LogisticRegression(C=C, max_iter=5000))])
    return pipe.fit(X[FEATURES], y)


def fit_lgbm(Xtr, ytr, Xva, yva, num_leaves: int, min_child_samples: int, cfg: dict, rs: int):
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=5000, learning_rate=cfg["learning_rate"],
        num_leaves=num_leaves, min_child_samples=min_child_samples,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=rs, n_jobs=4, deterministic=True, force_row_wise=True, verbose=-1)
    model.fit(Xtr[FEATURES], ytr, eval_set=[(Xva[FEATURES], yva)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
    return model


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibration_decision(p_valid, y_valid, min_gain: float, rs: int):
    """Platt scaling evaluated by 5-fold CV on VALID; returns (calibrator or None, diagnostics)."""
    y_valid = np.asarray(y_valid)
    z = _logit(p_valid)
    cv_pred = np.zeros(len(y_valid))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rs).split(z, y_valid):
        cv_pred[te] = LogisticRegression(C=1e6).fit(z[tr], y_valid[tr]).predict_proba(z[te])[:, 1]
    raw, cal = brier_score_loss(y_valid, p_valid), brier_score_loss(y_valid, cv_pred)
    use = (raw - cal) >= min_gain
    diag = {"valid_brier_raw": float(raw), "valid_brier_platt_cv": float(cal),
            "applied": bool(use)}
    return (LogisticRegression(C=1e6).fit(z, y_valid) if use else None), diag


def predict_proba(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Calibrated purchase probability for rows that already carry FEATURES."""
    p = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    if bundle.get("calibrator") is not None:
        p = bundle["calibrator"].predict_proba(_logit(p))[:, 1]
    return p


# ----------------------------------------------------------------------------- data
def load_model_frame(con) -> pd.DataFrame:
    df = con.execute("SELECT * FROM marts.model_frame").df()
    df = add_calendar_feature(df)
    df[LABEL] = df[LABEL].astype(int)
    return df


def check_time_order(df: pd.DataFrame) -> dict:
    """Label windows must end (exclusive) on or before the next split's cutoff."""
    g = df.groupby("split").agg(first_cutoff=("cutoff_date", "min"),
                                last_label_end=("label_end_date", "max"))
    g = g.reindex(["train", "valid", "test"]).apply(pd.to_datetime)
    ok = (g.loc["train", "last_label_end"] <= g.loc["valid", "first_cutoff"]
          and g.loc["valid", "last_label_end"] <= g.loc["test", "first_cutoff"])
    if not ok:
        raise RuntimeError(f"Split windows overlap:\n{g}")
    return {k: {c: str(v.date()) for c, v in row.items()} for k, row in g.iterrows()}


# ----------------------------------------------------------------------------- figures
def _save_figures(y_va, y_te, scores_va, scores_te, chosen: str, fig_dir) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    files = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, name, curve in ((axes[0], "ROC (test)", roc_curve), (axes[1], "Precision-recall (test)", precision_recall_curve)):
        for model, s in scores_te.items():
            a, b, _ = curve(y_te, s)
            x, yv = (a, b) if curve is roc_curve else (b, a)
            ax.plot(x, yv, label=model, lw=2 if model == chosen else 1)
        if curve is roc_curve:
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        else:
            ax.axhline(np.mean(y_te), color="k", ls=":", lw=0.8, label="prevalence")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(name); ax.legend(fontsize=8)
    fig.tight_layout(); f = fig_dir / "step6_roc_pr_test.png"; fig.savefig(f, dpi=130); plt.close(fig)
    files.append(f.name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (split, y, sc) in zip(axes, (("valid", y_va, scores_va), ("test", y_te, scores_te))):
        for model in ("logistic_regression", "lightgbm"):
            frac_pos, mean_pred = calibration_curve(y, sc[model], n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, "o-", ms=3, label=model, lw=2 if model == chosen else 1)
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_title(f"Reliability - {split} (prevalence {np.mean(y):.1%})")
        ax.set_xlabel("Mean predicted probability"); ax.legend(fontsize=8)
    axes[0].set_ylabel("Observed purchase rate")
    fig.tight_layout(); f = fig_dir / "step6_calibration.png"; fig.savefig(f, dpi=130); plt.close(fig)
    files.append(f.name)
    return files


# ----------------------------------------------------------------------------- report
def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else v


def write_report(m: dict, reports_dir) -> None:
    top = f"precision_at_top{int(m['config']['top_fraction'] * 100)}"
    lift = top.replace("precision", "lift")
    cols = ["roc_auc", "pr_auc", top, lift, "brier", "ece", "mean_predicted", "prevalence",
            "precision", "recall", "f1", "selection_rate"]
    rows = []
    for split in ("valid", "test"):
        for model, r in m["results"][split].items():
            rows.append({"split": split, "model": model, **{c: _fmt(r.get(c, "-")) for c in cols}})
    results = pd.DataFrame(rows)

    t = m["results"]["test"][m["chosen_model"]]
    cm = t["confusion"]
    confusion = pd.DataFrame({"": ["actual: no purchase", "actual: purchase"],
                              "predicted: no purchase": [cm["tn"], cm["fn"]],
                              "predicted: purchase": [cm["fp"], cm["tp"]]})
    audit = pd.DataFrame(m["audit_test"]).T.reset_index().rename(columns={"index": "group"})
    audit["n"] = audit["n"].astype(float).astype(int)
    audit = audit.map(_fmt)
    abl = pd.DataFrame(m["ablation_test"]).T.reset_index().rename(columns={"index": "variant"})
    abl = abl.map(_fmt)
    grid = pd.DataFrame(m["lgbm_grid"]).map(_fmt)
    ci = m["test_bootstrap"]
    splits = pd.DataFrame(m["splits"]).T.reset_index().rename(columns={"index": "split"})

    text = f"""# Step 6 - Purchase-Propensity Model

## Question
For a customer who bought in the last 365 days, how likely is at least one purchase in the next
{m['horizon_days']} days? The score is a **model-estimated propensity** based on observed purchase
history. It describes association, not cause: a high score does not mean any feature *makes* the
customer buy.

## Design
| Item | Choice |
| --- | --- |
| Population | customers with >=1 purchase in the 365 days before the cutoff |
| Features | {len(m['features'])} = {len(m['features']) - 1} SQL features (`customer_features(cutoff)`, rows strictly before the cutoff) + `{CAL_FEATURE}` |
| Calendar feature | share of the {m['horizon_days']}-day label window falling in months {m['config']['peak_months']}; known at the cutoff, so not leakage |
| Label | `{LABEL}` from `purchase_labels(cutoff, {m['horizon_days']})` |
| Split | time-based; train = stacked monthly cutoffs, valid and test = later single cutoffs |
| Excluded | `country` (audit only), `customer_id`, `cutoff_date` |
| Imbalance | positive rate 33-58% per cutoff -> no resampling, no class weights (they would distort probabilities); threshold tuned on valid instead |
| Tuned on | validation only. Test was scored once, after every choice was fixed |

### Split windows (label end is exclusive)
{df_to_markdown(splits)}

## Model selection (validation)
- Logistic regression C = {m['logreg_best_C']} (grid {m['config']['logreg_C']}, chosen by valid ROC-AUC).
- LightGBM grid (valid ROC-AUC, early stopping on valid log loss):

{df_to_markdown(grid)}

- Rule: keep LightGBM only if it beats logistic regression on valid ROC-AUC by >= {m['config']['min_auc_gain_for_trees']}.
  Valid ROC-AUC: LightGBM {m['results']['valid']['lightgbm']['roc_auc']:.3f} vs logistic {m['results']['valid']['logistic_regression']['roc_auc']:.3f}
  -> **chosen model: `{m['chosen_model']}`**.
- Calibration: Platt scaling, 5-fold CV Brier on valid {m['calibration']['valid_brier_platt_cv']:.4f} vs raw {m['calibration']['valid_brier_raw']:.4f}
  -> applied: **{m['calibration']['applied']}**.
- Threshold: maximises F1 on valid -> **{m['threshold']:.3f}**. Recency rule: "bought in the last {m['recency_rule_days']:.0f} days".

## Results (valid and test)
Selection metrics were computed on valid; the test rows are the one-time held-out evaluation.
For the recency rule, calibration metrics do not apply ("-").

{df_to_markdown(results)}

Test uncertainty for `{m['chosen_model']}` (bootstrap over customers, {ci['reps']} reps):
ROC-AUC 95% CI [{ci['roc_auc_95ci'][0]:.3f}, {ci['roc_auc_95ci'][1]:.3f}],
PR-AUC 95% CI [{ci['pr_auc_95ci'][0]:.3f}, {ci['pr_auc_95ci'][1]:.3f}].

### Confusion matrix - test, threshold {m['threshold']:.3f}
{df_to_markdown(confusion)}

![ROC and PR](figures/step6_roc_pr_test.png)

![Calibration](figures/step6_calibration.png)

## Seasonality check (ablation on test)
Valid (Jun-Aug window) is off-season; test (Sep-Nov window) is peak season. The same model is
refitted without the calendar feature to show what the feature does. `mean_predicted` close to
`prevalence` means the overall level of the probabilities is right.

{df_to_markdown(abl)}

## Fairness / subgroup audit (test, country used for audit only)
Differences here are **descriptive**. The groups differ in size and behaviour; a gap in
selection rate is not by itself evidence of unfair treatment, but a gap in recall or
calibration at the same threshold would mean the score serves one group worse.

{df_to_markdown(audit)}

## Why these metrics
| Metric | Why it is reported |
| --- | --- |
| ROC-AUC | Ranking quality independent of the base rate - important because valid ({m['results']['valid']['lightgbm']['prevalence']:.1%}) and test ({m['results']['test']['lightgbm']['prevalence']:.1%}) prevalence differ. Used for model selection. |
| PR-AUC | Ranking quality focused on the buyers; compare against prevalence (its random baseline). |
| Precision@top-{int(m['config']['top_fraction'] * 100)}% / lift | Matches how a campaign uses a score: contact the top slice. Robust to threshold choice. |
| Precision / recall / F1 at threshold | What happens if the score is turned into a yes/no flag. |
| Confusion matrix | Makes the error types concrete (wasted contacts vs missed buyers). |
| Brier, ECE, reliability curve, mean predicted | Whether "0.7" really means about 70% - needed because the profile engine shows probabilities to people. |

## Leakage checklist
- Features use `invoice_date < cutoff`; labels use `cutoff <= invoice_date < cutoff + {m['horizon_days']}` (SQL macros, tested for future-data invariance).
- Every train label window ends before the valid cutoff; the valid window ends before the test cutoff (checked in code, see table above).
- Preprocessing statistics (log/winsor/impute/scale) are learned on train only.
- The calendar feature uses only the calendar dates of the future window, not anything that happens in it.
- Residual risk: the same customer appears at several train cutoffs (correlated rows). This does not leak into valid/test, which are later in time, but it makes the train sample look larger than it is.

## Limitations
- One retailer, UK-centred, 2009-2011. Seasonality is learned from a single year (one peak season in train).
- Early train cutoffs have short history (data starts Dec 2009), so `tenure_days` is left-censored and the population at early cutoffs is younger.
- ~23% of transaction lines have no customer ID; those purchases are invisible to the model.
- Valid is off-season, so the calibration decision and threshold were chosen outside peak season; the test rows show how they carry over.
- Scores are associations useful for prioritising contact, not estimates of the effect of any action.
"""
    (reports_dir / "model_report.md").write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------- scoring
def score_current_snapshot(con, bundle: dict) -> pd.DataFrame:
    snap = con.execute("SELECT * FROM marts.customer_snapshot_current").df()
    if "cutoff_date" not in snap.columns:
        snap["cutoff_date"] = con.execute(
            "SELECT CAST(MAX(invoice_date) + INTERVAL 1 DAY AS DATE) FROM core.fact_line").fetchone()[0]
    snap = add_calendar_feature(snap, bundle["horizon_days"], bundle["peak_months"])
    p = predict_proba(bundle, snap)
    out = pd.DataFrame({
        "customer_id": snap["customer_id"].values,
        "scoring_cutoff": pd.to_datetime(snap["cutoff_date"]).dt.date.values,
        "propensity_90d": np.round(p, 4),
        "propensity_decile": pd.qcut(pd.Series(p).rank(method="first"), 10, labels=range(1, 11)).astype(int).values,
        "predicted_buyer": (p >= bundle["threshold"]).astype(int),
    })
    con.register("propensity_tmp", out)
    con.execute("CREATE OR REPLACE TABLE marts.customer_propensity AS SELECT * FROM propensity_tmp")
    con.unregister("propensity_tmp")
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    cfg, rs = pcfg(), seed()
    np.random.seed(rs)
    top_frac = float(cfg["top_fraction"])
    reports_dir, models_dir = path("reports_dir"), path("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    con = connect()
    try:
        print("1/7 loading marts.model_frame ...")
        df = load_model_frame(con)
    finally:
        con.close()
    splits = check_time_order(df)
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    ytr, yva, yte = tr[LABEL].values, va[LABEL].values, te[LABEL].values
    print(f"    train {len(tr):,} rows | valid {len(va):,} | test {len(te):,}")

    print("2/7 baselines: recency rule + logistic regression ...")
    rec_va, rec_te = -va["recency_days"].values, -te["recency_days"].values
    rec_t = best_f1_threshold(yva, rec_va)
    lr_scores = {}
    for C in cfg["logreg_C"]:
        pipe = fit_logreg(tr, ytr, C)
        lr_scores[C] = (roc_auc_score(yva, pipe.predict_proba(va[FEATURES])[:, 1]), pipe)
    best_C = max(lr_scores, key=lambda c: lr_scores[c][0])
    logreg = lr_scores[best_C][1]

    print("3/7 LightGBM grid (early stopping on valid) ...")
    grid, best = [], None
    for nl, mcs in product(cfg["lgbm_grid"]["num_leaves"], cfg["lgbm_grid"]["min_child_samples"]):
        mdl = fit_lgbm(tr, ytr, va, yva, nl, mcs, cfg, rs)
        auc = roc_auc_score(yva, mdl.predict_proba(va[FEATURES])[:, 1])
        grid.append({"num_leaves": nl, "min_child_samples": mcs,
                     "best_iteration": int(mdl.best_iteration_ or mdl.n_estimators), "valid_roc_auc": float(auc)})
        if best is None or auc > best[0]:
            best = (auc, mdl, nl, mcs)
    lgbm = best[1]

    print("4/7 model choice, calibration and threshold (valid only) ...")
    raw_va = {"logistic_regression": logreg.predict_proba(va[FEATURES])[:, 1],
              "lightgbm": lgbm.predict_proba(va[FEATURES])[:, 1]}
    gain = roc_auc_score(yva, raw_va["lightgbm"]) - roc_auc_score(yva, raw_va["logistic_regression"])
    chosen = "lightgbm" if gain >= cfg["min_auc_gain_for_trees"] else "logistic_regression"
    calibrator, cal_diag = calibration_decision(raw_va[chosen], yva, cfg["min_brier_gain_for_calibration"], rs)

    bundle = {"model_type": chosen, "model": lgbm if chosen == "lightgbm" else logreg,
              "calibrator": calibrator, "features": FEATURES, "label": LABEL,
              "horizon_days": horizon_days(), "peak_months": list(cfg["peak_months"]),
              "threshold": None, "lightgbm": lgbm, "logistic_regression": logreg,
              "trained_on": "train split only (valid used for tuning/calibration/threshold)"}
    p_va = predict_proba(bundle, va)
    bundle["threshold"] = best_f1_threshold(yva, p_va)

    print("5/7 scoring valid and test (test used once) ...")
    p_te = predict_proba(bundle, te)
    scores_va = {"recency_rule": rec_va, **raw_va}
    scores_te = {"recency_rule": rec_te,
                 "logistic_regression": logreg.predict_proba(te[FEATURES])[:, 1],
                 "lightgbm": lgbm.predict_proba(te[FEATURES])[:, 1]}
    scores_va[chosen], scores_te[chosen] = p_va, p_te   # chosen model shown after calibration

    results = {"valid": {}, "test": {}}
    for split, y, sc in (("valid", yva, scores_va), ("test", yte, scores_te)):
        for name, s in sc.items():
            if name == "recency_rule":
                results[split][name] = evaluate(y, s, rec_t, is_probability=False, top_fraction=top_frac)
            else:
                t = bundle["threshold"] if name == chosen else best_f1_threshold(yva, scores_va[name])
                results[split][name] = evaluate(y, s, t, top_fraction=top_frac)

    print("6/7 ablation, subgroup audit, bootstrap ...")
    abl_model = lgb.LGBMClassifier(**{**lgbm.get_params(), "n_estimators": int(lgbm.best_iteration_ or lgbm.n_estimators)})
    no_cal = [f for f in FEATURES if f != CAL_FEATURE]
    abl_model.fit(tr[no_cal], ytr)
    p_abl = abl_model.predict_proba(te[no_cal])[:, 1]
    p_lgbm_te = lgbm.predict_proba(te[FEATURES])[:, 1]
    keys = ("roc_auc", "pr_auc", "brier", "ece", "mean_predicted", "prevalence")
    ablation = {"lightgbm_with_calendar": {k: evaluate(yte, p_lgbm_te)[k] for k in keys},
                "lightgbm_without_calendar": {k: evaluate(yte, p_abl)[k] for k in keys}}

    audit = {}
    group = np.where(te[AUDIT_COLUMN].astype(str).str.strip() == "United Kingdom", "UK", "International")
    for g in ("UK", "International"):
        m = group == g
        r = {"n": int(m.sum()), "prevalence": float(yte[m].mean()), "mean_predicted": float(p_te[m].mean())}
        if len(np.unique(yte[m])) == 2:
            r.update({"roc_auc": float(roc_auc_score(yte[m], p_te[m])),
                      "pr_auc": float(average_precision_score(yte[m], p_te[m]))})
        tm = threshold_metrics(yte[m], p_te[m], bundle["threshold"])
        r.update({k: tm[k] for k in ("selection_rate", "recall", "precision", "false_positive_rate")})
        audit[g] = r

    ci = bootstrap_ci(yte, p_te, int(cfg["bootstrap_reps"]), np.random.default_rng(rs))
    figs = _save_figures(yva, yte, {**raw_va, chosen: p_va}, scores_te, chosen, reports_dir / "figures")

    metrics = {
        "target": LABEL, "horizon_days": horizon_days(), "features": FEATURES,
        "excluded_from_model": [AUDIT_COLUMN, "customer_id", "cutoff_date"],
        "splits": splits, "chosen_model": chosen, "valid_auc_gain_lgbm_vs_logreg": float(gain),
        "logreg_best_C": best_C, "lgbm_best": {"num_leaves": best[2], "min_child_samples": best[3],
                                               "best_iteration": int(lgbm.best_iteration_ or lgbm.n_estimators)},
        "lgbm_grid": grid, "calibration": cal_diag, "threshold": bundle["threshold"],
        "recency_rule_days": float(-rec_t), "results": results, "ablation_test": ablation,
        "audit_test": audit, "test_bootstrap": ci, "figures": figs,
        "config": {k: cfg[k] for k in ("peak_months", "logreg_C", "min_auc_gain_for_trees",
                                        "min_brier_gain_for_calibration", "top_fraction")},
        "random_seed": rs,
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    joblib.dump(bundle, models_dir / "propensity.joblib")
    write_report(metrics, reports_dir)

    print("7/7 scoring current snapshot -> marts.customer_propensity ...")
    con = connect()
    try:
        cur = score_current_snapshot(con, bundle)
    finally:
        con.close()

    t = results["test"][chosen]
    print(f"\nChosen: {chosen} | threshold {bundle['threshold']:.3f} | calibration applied: {cal_diag['applied']}")
    print(f"TEST  ROC-AUC {t['roc_auc']:.3f}  PR-AUC {t['pr_auc']:.3f}  "
          f"P@top{int(top_frac * 100)} {t[f'precision_at_top{int(top_frac * 100)}']:.3f}  "
          f"F1 {t['f1']:.3f}  Brier {t['brier']:.3f}  mean pred {t['mean_predicted']:.3f} vs prevalence {t['prevalence']:.3f}")
    print(f"Scored {len(cur):,} current customers.")
    print("Done -> models/propensity.joblib, models/metrics.json, reports/model_report.md")


if __name__ == "__main__":
    main()
