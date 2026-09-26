"""Step 7 - Explainability for the purchase-propensity model.

Global drivers
  * SHAP (TreeSHAP, computed by LightGBM itself via pred_contrib=True - exact, and it uses the
    same best iteration as predict_proba) on the current snapshot, i.e. the customers we profile.
  * Permutation importance on VALID (drop in ROC-AUC when a feature, or a whole feature family,
    is shuffled). Family-level shuffling is shown because several features are correlated
    (e.g. frequency_total / frequency_365d / frequency_90d) and share credit.
  * Logistic-regression coefficients (standardised log features) as a cross-check.
Per-customer drivers
  * Top 3 features raising and top 3 lowering each customer's score -> marts.customer_drivers
  * Family-level contribution per customer -> marts.customer_driver_families

All outputs describe what the MODEL associates with a higher or lower score. They are not
causes: shuffling or changing a feature in the model does not tell us what would happen if a
customer's behaviour changed.

Run:  python -m src.explain
"""
from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.build_warehouse import df_to_markdown
from src.config import load_config, path
from src.db import connect
from src.features import FEATURES as SQL_FEATURES
from src.features import LABEL
from src.propensity import CAL_FEATURE, FEATURES, add_calendar_feature, load_model_frame

TOP_K = 3
PERM_REPEATS = 5
# The calendar feature is context (identical for every customer at one cutoff), not customer
# behaviour, so it is excluded from rankings and driver lists and reported as a single shift.
BEHAVIOUR = [f for f in FEATURES if f != CAL_FEATURE]

FAMILY = {f[0]: f[2] for f in SQL_FEATURES}
FAMILY[CAL_FEATURE] = "Calendar"

# Plain-language labels used in reports and in the profile engine (Step 8).
FEATURE_LABELS = {
    "recency_days": "days since last purchase",
    "tenure_days": "days since first purchase",
    "frequency_total": "orders (all history)",
    "frequency_365d": "orders in the last 12 months",
    "frequency_90d": "orders in the last 90 days",
    "gross_spend_total": "total spend",
    "gross_spend_365d": "spend in the last 12 months",
    "gross_spend_90d": "spend in the last 90 days",
    "aov_365d": "average order value (12 months)",
    "median_order_value": "median order value",
    "avg_products_per_order": "products per order",
    "median_units_per_order": "units per order (median)",
    "median_line_quantity": "quantity per order line (median)",
    "avg_unit_price_paid": "average unit price paid",
    "price_index": "price paid vs typical price",
    "distinct_products": "distinct products bought",
    "distinct_product_groups": "product groups bought",
    "product_group_entropy": "spread of spend across product groups",
    "top_group_share": "share of spend in main product group",
    "christmas_share": "share of spend on Christmas products",
    "repeat_sku_share": "share of lines re-ordering a product",
    "cancel_rate": "cancellations per order",
    "cancel_value_share": "cancelled value vs spend",
    "mean_gap_days": "average days between orders",
    "median_gap_days": "typical days between orders",
    "gap_cv": "irregularity of order gaps",
    "overdue_ratio": "days since last order vs usual gap",
    "spend_trend_90d_log_ratio": "spend trend (last 90d vs previous 90d)",
    "order_trend_90d": "order trend (last 90d vs previous 90d)",
    "active_windows_12m": "active months in the last 12",
    "spend_slope_rel_12m": "12-month spend trend",
    "q4_spend_share": "share of spend in Oct-Dec",
    "morning_order_share": "share of orders placed before noon",
    CAL_FEATURE: "share of the next 90 days in peak season",
}


# ----------------------------------------------------------------------------- core
def contributions(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP values (log-odds) per row and feature, plus the base value."""
    out = np.asarray(model.predict(X[FEATURES], pred_contrib=True))
    return out[:, :-1], float(out[0, -1])


def direction_of(values: pd.Series, contrib: np.ndarray) -> tuple[str, float]:
    """Spearman correlation between a feature and its SHAP value -> wording of the association."""
    s = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce").values, "c": contrib}).dropna()
    if len(s) < 30 or s["v"].nunique() < 3:
        return "too few distinct values", float("nan")
    rho = float(s["v"].rank().corr(s["c"].rank()))
    if rho >= 0.3:
        return "higher value -> higher score", rho
    if rho <= -0.3:
        return "higher value -> lower score", rho
    return "mixed / non-linear", rho


def top_drivers(ids, X: pd.DataFrame, C: np.ndarray, k: int = TOP_K) -> pd.DataFrame:
    """Long table: for each customer the k largest positive and k largest negative contributions."""
    feats = np.array(FEATURES)
    rows = []
    order_up = np.argsort(-C, axis=1)[:, :k + 1]   # +1: the calendar column may be skipped
    order_dn = np.argsort(C, axis=1)[:, :k + 1]
    for i, cid in enumerate(ids):
        for direction, order, keep in (("raises", order_up[i], lambda v: v > 0),
                                       ("lowers", order_dn[i], lambda v: v < 0)):
            rank = 0
            for j in order:
                if feats[j] == CAL_FEATURE or not keep(C[i, j]) or rank == k:
                    continue
                rank += 1
                f = feats[j]
                v = X.iloc[i][f]
                rows.append({"customer_id": cid, "direction": direction, "rank": rank,
                             "feature": f, "label": FEATURE_LABELS[f], "family": FAMILY[f],
                             "feature_value": None if pd.isna(v) else float(v),
                             "contribution_logodds": round(float(C[i, j]), 4)})
    return pd.DataFrame(rows)


def family_contributions(ids, C: np.ndarray) -> pd.DataFrame:
    fam = pd.DataFrame(C, columns=FEATURES)[BEHAVIOUR].T.groupby(lambda f: FAMILY[f]).sum().T
    fam.insert(0, "customer_id", list(ids))
    return fam.melt(id_vars="customer_id", var_name="family", value_name="contribution_logodds")


def grouped_permutation(model, X: pd.DataFrame, y, groups: dict[str, list[str]],
                        repeats: int, rng: np.random.Generator) -> pd.DataFrame:
    """Drop in ROC-AUC when all columns of a group are shuffled together (same row order)."""
    base = roc_auc_score(y, model.predict_proba(X[FEATURES])[:, 1])
    rows = []
    for g, cols in groups.items():
        drops = []
        for _ in range(repeats):
            Xp = X[FEATURES].copy()
            idx = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].values[idx]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        rows.append({"family": g, "n_features": len(cols),
                     "perm_auc_drop": float(np.mean(drops)), "perm_auc_drop_std": float(np.std(drops))})
    return pd.DataFrame(rows)


def logreg_coefficients(logreg) -> pd.Series | None:
    coef = logreg.named_steps["clf"].coef_[0]
    if len(coef) != len(FEATURES):
        return None
    return pd.Series(coef, index=FEATURES)


# ----------------------------------------------------------------------------- figures
def _figures(X: pd.DataFrame, C: np.ndarray, glob: pd.DataFrame, fam: pd.DataFrame, fig_dir) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    files = []

    top = glob.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["label"], top["mean_abs_shap"], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP| (log-odds) - current snapshot")
    ax.set_title("Top 15 features by average contribution to the score")
    fig.tight_layout(); f = fig_dir / "step7_shap_global.png"; fig.savefig(f, dpi=130); plt.close(fig)
    files.append(f.name)

    fam_s = fam.sort_values("shap_share", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    axes[0].barh(fam_s["family"], fam_s["shap_share"] * 100, color="#4C72B0")
    axes[0].set_xlabel("Share of total |SHAP| (%)"); axes[0].set_title("SHAP (current snapshot)")
    axes[1].barh(fam_s["family"], fam_s["perm_auc_drop"], xerr=fam_s["perm_auc_drop_std"], color="#DD8452")
    axes[1].set_xlabel("Drop in ROC-AUC when shuffled"); axes[1].set_title("Permutation (validation)")
    fig.suptitle("Feature families: two independent views of importance")
    fig.tight_layout(); f = fig_dir / "step7_family_importance.png"; fig.savefig(f, dpi=130); plt.close(fig)
    files.append(f.name)

    feats = list(glob["feature"].head(3))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, feat in zip(axes, feats):
        j = FEATURES.index(feat)
        v = pd.to_numeric(X[feat], errors="coerce").values
        ax.scatter(v, C[:, j], s=4, alpha=0.3)
        if np.nanmin(v) >= 0 and np.nanmax(v) > 50 * max(np.nanmedian(v), 1e-9):
            ax.set_xscale("symlog")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(FEATURE_LABELS[feat]); ax.set_title(feat, fontsize=9)
    axes[0].set_ylabel("SHAP contribution (log-odds)")
    fig.suptitle("How the top 3 features relate to the score (each dot = one customer)")
    fig.tight_layout(); f = fig_dir / "step7_shap_dependence.png"; fig.savefig(f, dpi=130); plt.close(fig)
    files.append(f.name)

    try:  # optional beeswarm from the shap library; the pipeline does not depend on it
        import shap
        shap.summary_plot(C, X[FEATURES].astype(float).values, feature_names=[FEATURE_LABELS[f] for f in FEATURES],
                          max_display=15, show=False)
        f = fig_dir / "step7_shap_beeswarm.png"
        plt.gcf().set_size_inches(9, 7); plt.tight_layout(); plt.savefig(f, dpi=130); plt.close("all")
        files.append(f.name)
    except Exception as exc:  # noqa: BLE001
        print(f"    (beeswarm skipped: {exc})")
    return files


# ----------------------------------------------------------------------------- report
def write_report(m: dict, glob: pd.DataFrame, fam: pd.DataFrame, reports_dir) -> None:
    g = glob.head(15).copy()
    g.insert(0, "rank", range(1, len(g) + 1))
    g = g[["rank", "feature", "label", "family", "mean_abs_shap", "direction", "spearman_rho",
           "perm_auc_drop", "logreg_coef"]]
    g = g.map(lambda v: f"{v:.3f}" if isinstance(v, float) and not pd.isna(v) else ("-" if pd.isna(v) else v))
    f = fam.sort_values("shap_share", ascending=False)[["family", "n_features", "shap_share", "perm_auc_drop", "perm_auc_drop_std"]].copy()
    f["shap_share"] = (f["shap_share"] * 100).map(lambda v: f"{v:.1f}%")
    f = f.map(lambda v: f"{v:.4f}" if isinstance(v, float) else v)
    cal_note = ("" if not m["calibrator_applied"] else
                "\nNote: a Platt calibrator is applied after the model; SHAP values explain the model's "
                "uncalibrated log-odds, so they explain the ranking exactly and the probability approximately.\n")

    text = f"""# Step 7 - Explainability

Model explained: `{m['model_explained']}` (chosen in Step 6). SHAP values are exact TreeSHAP values
computed by LightGBM (`pred_contrib=True`), in **log-odds**; for every customer
`base value ({m['base_value_logodds']:.3f}) + sum of contributions = model log-odds`
(max additivity error on the snapshot: {m['max_additivity_error']:.2e}).{cal_note}

**How to read this.** A feature "raising the score" means the model associates that customer's
value with a higher probability of buying in the next 90 days, compared with the average
customer. These are associations learned from history, **not causes**: the model does not tell
us that changing a customer's behaviour, or contacting them, would change the outcome.
Correlated features (for example the three order-count windows) share credit, so the
family-level view is more reliable than any single feature.

**Seasonal context.** The calendar feature (`{CAL_FEATURE}`) is the same for every customer at a
given cutoff, so it is not a customer characteristic. At the scoring cutoff the next 90 days are
off-season and it shifts every customer's log-odds by **{m['calendar_shift_logodds']:+.3f}**. It is
excluded from the rankings and driver lists below.

## Global drivers - top 15 customer-behaviour features
SHAP on the current snapshot ({m['n_snapshot']:,} customers, cutoff {m['scoring_cutoff']});
permutation importance on validation ({m['n_valid']:,} customers, ROC-AUC {m['valid_auc']:.3f});
`logreg_coef` = logistic-regression coefficient on the standardised (log) feature, as a cross-check
(sign should usually agree with `direction`; magnitudes are unstable under correlation).

{df_to_markdown(g)}

Agreement between the two importance methods (Spearman correlation of feature ranks, SHAP vs
permutation): **{m['rank_agreement_shap_vs_perm']:.2f}**.
Direction agreement between SHAP and logistic-regression signs (features with a clear
direction): **{m['sign_agreement_shap_vs_logreg']}**.

![Global SHAP](figures/step7_shap_global.png)

![Dependence](figures/step7_shap_dependence.png)

## Feature families
{df_to_markdown(f)}

![Families](figures/step7_family_importance.png)

## Per-customer drivers
For every customer in the snapshot, the {TOP_K} features raising and the {TOP_K} features lowering
their score are stored in `marts.customer_drivers` (with plain-language labels), and
family-level totals in `marts.customer_driver_families`. The profile engine (Step 8) reads
these tables.

## Limitations
- SHAP explains the model, not the world. If the model is wrong for a customer, so is the explanation.
- Correlated features split credit arbitrarily between them; use families for conclusions.
- Missing rhythm features (customers with too few orders) are handled natively by LightGBM;
  their SHAP value reflects "missing" as information (few repeat orders), not a measured gap.
- Permutation importance is measured off-season (validation). The calendar feature is constant
  there, so it cannot be measured this way; its effect is shown by the Step 6 ablation.
"""
    (reports_dir / "explainability.md").write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------- main
def _write_table(con, name: str, df: pd.DataFrame) -> None:
    con.register("tmp_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_df")
    con.unregister("tmp_df")


def main() -> None:
    rs = int(load_config()["project"]["random_seed"])
    reports_dir, models_dir = path("reports_dir"), path("models_dir")
    bundle = joblib.load(models_dir / "propensity.joblib")
    lgbm = bundle["lightgbm"]

    con = connect()
    try:
        print("1/5 loading validation frame and current snapshot ...")
        va = load_model_frame(con).query("split == 'valid'").reset_index(drop=True)
        snap = con.execute("SELECT * FROM marts.customer_snapshot_current").df()
        if "cutoff_date" not in snap.columns:
            snap["cutoff_date"] = con.execute(
                "SELECT CAST(MAX(invoice_date) + INTERVAL 1 DAY AS DATE) FROM core.fact_line").fetchone()[0]
        snap = add_calendar_feature(snap, bundle["horizon_days"], bundle["peak_months"])

        print("2/5 SHAP contributions (exact TreeSHAP via LightGBM) ...")
        C, base = contributions(lgbm, snap)
        raw = np.asarray(lgbm.predict(snap[FEATURES], raw_score=True))
        add_err = float(np.max(np.abs(base + C.sum(axis=1) - raw)))

        jb = [FEATURES.index(f) for f in BEHAVIOUR]
        cal_shift = float(C[:, FEATURES.index(CAL_FEATURE)].mean())
        glob = pd.DataFrame({"feature": BEHAVIOUR, "label": [FEATURE_LABELS[f] for f in BEHAVIOUR],
                             "family": [FAMILY[f] for f in BEHAVIOUR],
                             "mean_abs_shap": np.abs(C[:, jb]).mean(axis=0)})
        dirs = [direction_of(snap[f], C[:, FEATURES.index(f)]) for f in BEHAVIOUR]
        glob["direction"] = [d[0] for d in dirs]
        glob["spearman_rho"] = [d[1] for d in dirs]

        print("3/5 permutation importance on validation (features and families) ...")
        yva = va[LABEL].values
        single = grouped_permutation(lgbm, va, yva, {f: [f] for f in BEHAVIOUR}, PERM_REPEATS,
                                     np.random.default_rng(rs))
        glob["perm_auc_drop"] = single["perm_auc_drop"].values   # same order as BEHAVIOUR
        coef = logreg_coefficients(bundle["logistic_regression"])
        glob["logreg_coef"] = coef.reindex(BEHAVIOUR).values if coef is not None else np.nan
        glob = glob.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        groups: dict[str, list[str]] = {}
        for f in BEHAVIOUR:
            groups.setdefault(FAMILY[f], []).append(f)
        fam = grouped_permutation(lgbm, va, yva, groups, PERM_REPEATS, np.random.default_rng(rs))
        fam_shap = pd.DataFrame(C, columns=FEATURES)[BEHAVIOUR].T.groupby(lambda f: FAMILY[f]).sum().T.abs().mean()
        fam["shap_share"] = fam["family"].map(fam_shap / fam_shap.sum())

        print("4/5 per-customer drivers -> marts.customer_drivers, marts.customer_driver_families ...")
        ids = snap["customer_id"].values
        drivers = top_drivers(ids, snap, C)
        fam_long = family_contributions(ids, C)
        _write_table(con, "marts.customer_drivers", drivers)
        _write_table(con, "marts.customer_driver_families", fam_long)
    finally:
        con.close()

    print("5/5 figures, report and metrics ...")
    figs = _figures(snap, C, glob, fam, reports_dir / "figures")
    clear = glob[glob["direction"].str.startswith("higher")].dropna(subset=["logreg_coef"])
    agree = int(((clear["direction"] == "higher value -> higher score") == (clear["logreg_coef"] > 0)).sum())
    m = {
        "model_explained": bundle["model_type"], "calibrator_applied": bundle.get("calibrator") is not None,
        "base_value_logodds": base, "max_additivity_error": add_err,
        "calendar_shift_logodds": cal_shift,
        "n_snapshot": int(len(snap)), "scoring_cutoff": str(pd.to_datetime(snap["cutoff_date"]).iloc[0].date()),
        "n_valid": int(len(va)), "valid_auc": float(roc_auc_score(yva, lgbm.predict_proba(va[FEATURES])[:, 1])),
        "rank_agreement_shap_vs_perm": float(glob["mean_abs_shap"].rank().corr(glob["perm_auc_drop"].rank())),
        "sign_agreement_shap_vs_logreg": f"{agree} of {len(clear)}",
        "global_top15": glob.head(15).to_dict(orient="records"),
        "families": fam.to_dict(orient="records"), "figures": figs, "feature_labels": FEATURE_LABELS,
    }
    (models_dir / "explainability.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    write_report(m, glob, fam, reports_dir)

    print("\nTop 5 features by mean |SHAP|:")
    print(glob.head(5)[["feature", "mean_abs_shap", "direction", "perm_auc_drop"]].to_string(index=False))
    print(f"\nFamilies (SHAP share | permutation AUC drop):")
    print(fam.sort_values("shap_share", ascending=False)[["family", "shap_share", "perm_auc_drop"]].to_string(index=False))
    print(f"\nRank agreement SHAP vs permutation: {m['rank_agreement_shap_vs_perm']:.2f} | "
          f"sign agreement with logistic regression: {m['sign_agreement_shap_vs_logreg']} | "
          f"additivity error {add_err:.1e}")
    print(f"Drivers stored for {drivers['customer_id'].nunique():,} customers.")
    print("Done -> reports/explainability.md, models/explainability.json")


if __name__ == "__main__":
    main()
