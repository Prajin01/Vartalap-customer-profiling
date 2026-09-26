"""Step 5 - Behavioural segmentation of active customers.

Compares K-Means and Gaussian Mixture Models for k = 2..8 on the current snapshot using
silhouette, Davies-Bouldin, Calinski-Harabasz, BIC (GMM) and bootstrap stability (ARI),
runs HDBSCAN as an outlier/structure diagnostic, selects (method, k), names segments from
their measured profiles and saves everything.

Run:  python -m src.segmentation
Override the automatic choice in config.yaml -> segmentation: {method, k, names}.
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.cluster import HDBSCAN, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,  # noqa: E402
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture  # noqa: E402

from src.build_warehouse import df_to_markdown  # noqa: E402
from src.config import load_config, path  # noqa: E402
from src.preprocess import make_numeric_pipeline  # noqa: E402

# Interpretable behavioural dimensions (the model in Step 6 uses all 33 features;
# clustering on 33 correlated features would be noisy and hard to name).
SEGMENT_FEATURES = [
    "recency_days",               # how recently
    "frequency_365d",             # how often
    "gross_spend_365d",           # how much
    "aov_365d",                   # order size
    "median_line_quantity",       # bulk buying
    "distinct_products",          # range
    "product_group_entropy",      # specialist vs generalist
    "repeat_sku_share",           # re-ordering
    "cancel_value_share",         # cancellations
    "spend_trend_90d_log_ratio",  # growing / declining
    "active_windows_12m",         # consistency
    "q4_spend_share",             # seasonality
]
PROFILE_EXTRA = ["tenure_days", "frequency_total", "gross_spend_total", "median_gap_days"]
K_RANGE = range(2, 9)


def _cfg() -> dict:
    return load_config().get("segmentation") or {}


def _seed() -> int:
    return int(load_config()["project"]["random_seed"])


def load_snapshot(con) -> pd.DataFrame:
    return con.execute("SELECT * FROM marts.customer_snapshot_current ORDER BY customer_id").df()


def make_model(method: str, k: int, seed: int):
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=20, random_state=seed)
    return GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=seed)


def bootstrap_stability(method: str, k: int, X: np.ndarray, ref_labels: np.ndarray,
                        seed: int, n_boot: int) -> float:
    """Mean ARI between the full-data solution and solutions fitted on bootstrap resamples."""
    rng = np.random.default_rng(seed)
    scores = []
    for b in range(n_boot):
        idx = rng.choice(len(X), size=len(X), replace=True)
        m = make_model(method, k, seed + b + 1).fit(X[idx])
        scores.append(adjusted_rand_score(ref_labels, m.predict(X)))
    return float(np.mean(scores))


def evaluate(X: np.ndarray, seed: int, n_boot: int = 20) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        for method in ("kmeans", "gmm"):
            model = make_model(method, k, seed)
            labels = model.fit_predict(X)
            n_used = len(np.unique(labels))
            rows.append({
                "method": method, "k": k,
                "silhouette": silhouette_score(X, labels, sample_size=min(3000, len(X)), random_state=seed)
                              if n_used > 1 else np.nan,
                "davies_bouldin": davies_bouldin_score(X, labels) if n_used > 1 else np.nan,
                "calinski_harabasz": calinski_harabasz_score(X, labels) if n_used > 1 else np.nan,
                "bic": model.bic(X) if method == "gmm" else np.nan,
                "min_cluster_share": np.bincount(labels, minlength=k).min() / len(labels),
                "stability_ari": bootstrap_stability(method, k, X, labels, seed, n_boot),
            })
            print(f"  {method:6s} k={k}: silhouette={rows[-1]['silhouette']:.3f} "
                  f"stability={rows[-1]['stability_ari']:.3f}")
    return pd.DataFrame(rows)


def choose(metrics: pd.DataFrame) -> tuple[str, int, str]:
    """Rank-based choice among k>=3 candidates that are stable and have no tiny segment.
    Ties go to the smaller k (simpler, more actionable)."""
    cfg = _cfg()
    if cfg.get("method") and cfg.get("k"):
        return cfg["method"], int(cfg["k"]), "set manually in config.yaml"
    cand = metrics[(metrics["k"] >= 3) & (metrics["stability_ari"] >= 0.75) & (metrics["min_cluster_share"] >= 0.03)]
    rule = "k>=3, bootstrap ARI>=0.75, smallest segment>=3%"
    if cand.empty:
        cand, rule = metrics[metrics["k"] >= 3], "k>=3 (no candidate met the stability/size filters)"
    score = (cand["silhouette"].rank(ascending=False) + cand["davies_bouldin"].rank(ascending=True)
             + cand["calinski_harabasz"].rank(ascending=False) + cand["stability_ari"].rank(ascending=False))
    best = cand.assign(score=score).sort_values(["score", "k"]).iloc[0]
    return best["method"], int(best["k"]), f"best combined rank of 4 metrics among {rule}"


def hdbscan_diagnostic(X: np.ndarray) -> dict:
    labels = HDBSCAN(min_cluster_size=50).fit_predict(X)
    return {"n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
            "noise_share": round(float((labels == -1).mean()), 3)}


def profile(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    cols = SEGMENT_FEATURES + PROFILE_EXTRA
    d = df.assign(segment_id=labels)
    prof = d.groupby("segment_id")[cols].median()
    prof.insert(0, "pct_revenue_365d", (100 * d.groupby("segment_id")["gross_spend_365d"].sum()
                                        / d["gross_spend_365d"].sum()).round(1))
    prof.insert(0, "pct_customers", (100 * d["segment_id"].value_counts(normalize=True)).round(1))
    prof.insert(0, "n_customers", d["segment_id"].value_counts())
    return prof.sort_index()


def draft_names(prof: pd.DataFrame, pop: pd.Series) -> dict[int, str]:
    """Rule-based draft names from medians relative to the population median."""
    names = {}
    for seg, r in prof.iterrows():
        rec = r["recency_days"] / max(pop["recency_days"], 1)
        freq = r["frequency_365d"] / max(pop["frequency_365d"], 1)
        spend = r["gross_spend_365d"] / max(pop["gross_spend_365d"], 1)
        bulk = r["median_line_quantity"] / max(pop["median_line_quantity"], 1)
        engagement = "Lapsing" if rec >= 2.5 else "Frequent" if freq >= 2 else "Recent" if rec <= 0.5 else "Occasional"
        value = "high-value" if spend >= 2 else "low-value" if spend <= 0.5 else "mid-value"
        kind = "bulk buyers" if bulk >= 2 else "buyers"
        names[int(seg)] = f"{engagement} {value} {kind}"
    seen: dict[str, int] = {}
    for seg in sorted(names):                      # make duplicates unique
        base = names[seg]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            names[seg] = f"{base} ({seen[base]})"
    overrides = {int(k): v for k, v in (_cfg().get("names") or {}).items()}
    return {**names, **overrides}


def _figures(metrics, X, labels, prof_z, names, fig_dir):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, col, title in zip(axes, ["silhouette", "davies_bouldin", "calinski_harabasz", "stability_ari"],
                              ["Silhouette (higher better)", "Davies-Bouldin (lower better)",
                               "Calinski-Harabasz (higher better)", "Bootstrap ARI (higher better)"]):
        for method, g in metrics.groupby("method"):
            ax.plot(g["k"], g[col], marker="o", label=method)
        ax.set_title(title, fontsize=10); ax.set_xlabel("k")
    axes[0].legend()
    plt.tight_layout(); plt.savefig(fig_dir / "seg_metrics.png", dpi=120); plt.close()

    xy = PCA(n_components=2, random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    for seg in np.unique(labels):
        m = labels == seg
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.5, label=names[int(seg)])
    ax.set_title("Segments projected on the first two principal components")
    ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout(); plt.savefig(fig_dir / "seg_pca.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(prof_z) + 2))
    sns.heatmap(prof_z.rename(index=names), annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "segment mean (standardised units)"})
    ax.set_title("Segment profiles (0 = population average)")
    plt.tight_layout(); plt.savefig(fig_dir / "seg_profile_heatmap.png", dpi=120); plt.close()


def run(df: pd.DataFrame, n_boot: int = 20) -> dict:
    seed = _seed()
    pipe = make_numeric_pipeline(SEGMENT_FEATURES)
    X = pipe.fit_transform(df[SEGMENT_FEATURES])

    print("Evaluating K-Means and GMM for k = 2..8 (with bootstrap stability) ...")
    metrics = evaluate(X, seed, n_boot)
    method, k, reason = choose(metrics)
    print(f"Chosen: {method}, k={k} ({reason})")

    model = make_model(method, k, seed).fit(X)
    labels = model.predict(X)
    prob = model.predict_proba(X).max(axis=1) if method == "gmm" else np.full(len(X), np.nan)

    prof = profile(df, labels)
    names = draft_names(prof, df[SEGMENT_FEATURES + PROFILE_EXTRA].median())
    prof_z = pd.DataFrame(X, columns=SEGMENT_FEATURES).assign(segment_id=labels).groupby("segment_id").mean()
    hdb = hdbscan_diagnostic(X)

    assignments = pd.DataFrame({
        "customer_id": df["customer_id"].values,
        "segment_id": labels.astype(int),
        "segment_name": [names[int(s)] for s in labels],
        "membership_prob": prob,
    })
    return {"pipe": pipe, "model": model, "method": method, "k": k, "reason": reason,
            "metrics": metrics, "profile": prof, "profile_z": prof_z, "names": names,
            "hdbscan": hdb, "assignments": assignments, "X": X, "labels": labels}


def assign_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Assign saved segments to new rows with the same feature columns (used by the profile engine)."""
    art = joblib.load(path("models_dir") / "segmentation.joblib")
    X = art["pipe"].transform(df[art["features"]])
    labels = art["model"].predict(X)
    return pd.DataFrame({"customer_id": df["customer_id"].values, "segment_id": labels.astype(int),
                         "segment_name": [art["names"][int(s)] for s in labels]})


def save(res: dict, df: pd.DataFrame, con) -> None:
    models, reports = path("models_dir"), path("reports_dir")
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({"pipe": res["pipe"], "model": res["model"], "method": res["method"], "k": res["k"],
                 "features": SEGMENT_FEATURES, "names": res["names"]}, models / "segmentation.joblib")

    con.register("seg_df", res["assignments"])
    con.execute("CREATE OR REPLACE TABLE marts.customer_segments AS SELECT * FROM seg_df")
    con.unregister("seg_df")

    prof = res["profile"].copy()
    prof.insert(0, "segment_name", prof.index.map(res["names"]))
    audit = (df.assign(segment_id=res["labels"],
                       market=np.where(df["country"] == "United Kingdom", "UK", "International"))
               .pipe(lambda d: pd.crosstab(d["segment_id"], d["market"], normalize="columns") * 100)
               .round(1))
    audit.index = audit.index.map(res["names"])

    segments_json = {
        "method": res["method"], "k": res["k"], "selection_reason": res["reason"],
        "features": SEGMENT_FEATURES, "preprocessing": "log1p (skewed) -> winsorise 1/99% -> median impute -> standardise",
        "snapshot": "marts.customer_snapshot_current (active customers at the last data date)",
        "names": {str(k): v for k, v in res["names"].items()},
        "profiles": json.loads(prof.round(3).to_json(orient="index")),
        "hdbscan_diagnostic": res["hdbscan"],
        "metrics": json.loads(res["metrics"].round(4).to_json(orient="records")),
    }
    (models / "segments.json").write_text(json.dumps(segments_json, indent=2), encoding="utf-8")

    _figures(res["metrics"], res["X"], res["labels"], res["profile_z"], res["names"], fig_dir)

    (reports / "segmentation.md").write_text(
        "# Customer Segmentation\n\n"
        f"**Population:** {len(df):,} active customers (>=1 purchase in the 365 days before the last data date).  \n"
        f"**Features ({len(SEGMENT_FEATURES)}):** {', '.join(SEGMENT_FEATURES)}.  \n"
        "**Preprocessing:** log1p on skewed features -> winsorise at 1%/99% -> median imputation -> standardisation.\n\n"
        f"## Model selection\n\nChosen: **{res['method']} with k = {res['k']}** ({res['reason']}).\n\n"
        + df_to_markdown(res["metrics"].round(3)) +
        f"\n\n**HDBSCAN diagnostic** (min cluster size 50): {res['hdbscan']['n_clusters']} dense clusters, "
        f"{100 * res['hdbscan']['noise_share']:.1f}% of customers labelled noise. A large noise share means the "
        "data has no sharp density-separated groups, so partitioning methods (K-Means/GMM) are appropriate.\n\n"
        "![metrics](figures/seg_metrics.png)\n\n"
        "## Segment profiles (medians)\n\n" + df_to_markdown(prof.reset_index().round(2)) +
        "\n\n![profiles](figures/seg_profile_heatmap.png)\n\n![pca](figures/seg_pca.png)\n\n"
        "## Audit: segment mix by market (country is NOT a clustering feature)\n\n"
        "% of each market's customers in each segment.\n\n" + df_to_markdown(audit.reset_index()) +
        "\n\nSegment names describe observed purchasing behaviour only; they make no claims about "
        "customers' motives or personality.\n",
        encoding="utf-8")


def main() -> None:
    from src.db import connect
    con = connect()
    try:
        df = load_snapshot(con)
        res = run(df)
        save(res, df, con)
    finally:
        con.close()
    print(res["profile"].assign(name=lambda p: p.index.map(res["names"]))
          [["name", "n_customers", "pct_customers", "pct_revenue_365d"]].to_string())
    print("Done -> models/segmentation.joblib, models/segments.json, reports/segmentation.md, marts.customer_segments")


if __name__ == "__main__":
    main()
