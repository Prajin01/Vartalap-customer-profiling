"""Customer Profiling Studio - Streamlit front end for the POC.

Run from the project root (after the pipeline has been run once):
    streamlit run app/streamlit_app.py

Reads only what the pipeline produced: the DuckDB warehouse (read-only), models/*.json and
reports/figures. No model is trained here.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402  (ships with streamlit)
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import path  # noqa: E402
from src.db import connect  # noqa: E402
from src.profile_engine import VIEW, choose_action, get_profile  # noqa: E402

st.set_page_config(page_title="Customer Profiling Studio", page_icon="🧭", layout="wide")

SEG_COLORS = {"high-value": "#4F46E5", "seasonal": "#F59E0B", "lapsed": "#6B7280",
              "cancel": "#DC2626", "low-spend": "#10B981"}


def seg_color(name) -> str:
    n = str(name or "").lower()
    return next((c for k, c in SEG_COLORS.items() if k in n), "#4F46E5")


st.markdown("""
<style>
.hero {padding: 1.1rem 1.4rem; border-radius: 14px; color: white;
       background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%); margin-bottom: 0.8rem;}
.hero h1 {font-size: 1.7rem; margin: 0; color: white;}
.hero p {margin: 0.2rem 0 0 0; opacity: 0.92;}
.card {border: 1px solid #E5E7EB; border-radius: 12px; padding: 0.9rem 1.1rem; background: #FFFFFF;}
.badge {display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; color: white;
        font-weight: 600; font-size: 0.85rem;}
.small {color: #6B7280; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data (cached)
@st.cache_data(show_spinner=False)
def load_base() -> pd.DataFrame:
    con = connect(read_only=True)
    try:
        df = con.execute(f"SELECT * FROM {VIEW}").df()
    finally:
        con.close()
    df["customer_id"] = df["customer_id"].map(_clean_id)
    df["suggested_action"] = [
        choose_action(r.segment_name, r.propensity_decile, r.pct_spend_365d, r.overdue_ratio,
                      r.cancel_value_share)["action"]
        for r in df.itertuples()]
    return df


@st.cache_data(show_spinner=False)
def load_segment_summary() -> pd.DataFrame:
    con = connect(read_only=True)
    try:
        return con.execute(f"""
            SELECT segment_name                                   AS segment,
                   COUNT(*)                                       AS customers,
                   ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)                         AS pct_customers,
                   ROUND(100.0 * SUM(gross_spend_365d) / SUM(SUM(gross_spend_365d)) OVER (), 1) AS pct_spend_12m,
                   ROUND(AVG(propensity_90d), 3)                  AS avg_propensity,
                   ROUND(MEDIAN(frequency_365d), 1)               AS median_orders_12m,
                   ROUND(MEDIAN(recency_days), 0)                 AS median_days_since_purchase
            FROM {VIEW}
            GROUP BY segment_name
            ORDER BY pct_spend_12m DESC
        """).df()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    f = path("models_dir") / name
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


@st.cache_data(show_spinner=False)
def profile_of(cid: str) -> dict:
    return get_profile(cid)


def _clean_id(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() else v
    except (TypeError, ValueError):
        return v


def fig(name: str):
    f = path("reports_dir") / "figures" / name
    if f.exists():
        st.image(str(f))
    else:
        st.caption(f"(figure {name} not found - run the pipeline)")


# ----------------------------------------------------------------------------- guard
try:
    base = load_base()
except Exception as exc:  # noqa: BLE001
    st.error("Could not read the warehouse. Run the pipeline first (`python run_pipeline.py`), "
             "and close other programs that have `data/warehouse.duckdb` open for writing.")
    st.exception(exc)
    st.stop()

metrics = load_json("metrics.json")
expl = load_json("explainability.json")
chosen = metrics.get("chosen_model", "lightgbm")
test = metrics.get("results", {}).get("test", {})
cutoff = str(pd.to_datetime(base["cutoff_date"]).max().date())

st.markdown(f"""
<div class="hero">
  <h1>🧭 Customer Profiling Studio</h1>
  <p>UCI Online Retail II · {len(base):,} active customers profiled as of {cutoff} ·
     segments, 90-day purchase propensity and explanations - observed behaviour only</p>
</div>
""", unsafe_allow_html=True)

tab_over, tab_prof, tab_target, tab_model = st.tabs(
    ["🏠 Overview", "👤 Customer profile", "🎯 Target list", "📊 Model & explainability"])

# ============================================================================= OVERVIEW
with tab_over:
    t = test.get(chosen, {})
    top_key = next((k for k in t if k.startswith("precision_at_top")), None)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers profiled", f"{len(base):,}")
    c2.metric("ROC-AUC (test)", f"{t.get('roc_auc', float('nan')):.3f}",
              help="Ranking quality on the held-out Sep-Nov 2011 window (scored once).")
    if top_key:
        c3.metric("Top-20% hit rate (test)", f"{t[top_key]:.0%}", help="Share of the top-scored 20% who actually bought.")
        c4.metric("Base rate (test)", f"{t.get('prevalence', 0):.0%}",
                  delta=f"lift x{t[top_key] / t['prevalence']:.2f}", delta_color="off")

    seg = load_segment_summary()
    st.subheader("Segments")
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(seg.rename(columns={"pct_customers": "% customers", "pct_spend_12m": "% spend",
                                         "avg_propensity": "avg propensity", "median_orders_12m": "median orders",
                                         "median_days_since_purchase": "median days since purchase"}),
                     use_container_width=True, hide_index=True)
        st.caption("Spend share uses each customer's last-12-month spend at the scoring date.")
    with right:
        long = seg.melt(id_vars="segment", value_vars=["pct_customers", "pct_spend_12m"],
                        var_name="measure", value_name="percent")
        long["measure"] = long["measure"].map({"pct_customers": "% of customers", "pct_spend_12m": "% of spend"})
        st.altair_chart(
            alt.Chart(long).mark_bar().encode(
                y=alt.Y("segment:N", title=None, sort="-x"),
                x=alt.X("percent:Q", title="%"),
                color=alt.Color("measure:N", title=None, scale=alt.Scale(range=["#A5B4FC", "#4F46E5"])),
                yOffset="measure:N",
                tooltip=["segment", "measure", "percent"]).properties(height=280),
            use_container_width=True)

    st.subheader("Who is likely to buy in the next 90 days?")
    dist = base.groupby(["segment_name", "propensity_decile"]).size().reset_index(name="customers")
    st.altair_chart(
        alt.Chart(dist).mark_bar().encode(
            x=alt.X("propensity_decile:O", title="propensity decile (10 = most likely)"),
            y=alt.Y("customers:Q", title="customers"),
            color=alt.Color("segment_name:N", title="segment",
                            scale=alt.Scale(domain=sorted(dist["segment_name"].dropna().unique()),
                                            range=[seg_color(s) for s in sorted(dist["segment_name"].dropna().unique())])),
            tooltip=["segment_name", "propensity_decile", "customers"]).properties(height=300),
        use_container_width=True)
    st.info("Scores are **associations** learned from past behaviour, not causes. Deciles are used for targeting; "
            "the probability level is calibrated for this off-season scoring date (see Model tab).")

# ============================================================================= PROFILE
with tab_prof:
    if "cid" not in st.session_state:
        st.session_state["cid"] = str(base.sort_values("propensity_90d", ascending=False)["customer_id"].iloc[0])

    ids = base["customer_id"].astype(str).tolist()
    examples = (base.assign(d=(base["propensity_90d"] - base.groupby("segment_name")["propensity_90d"].transform("median")).abs())
                .sort_values("d").groupby("segment_name").head(1))

    a, b = st.columns([2, 3])
    with a:
        st.text_input("Customer ID", key="cid", help="Any ID from the dataset, e.g. 16983")
        st.button("🎲 Random customer", on_click=lambda: st.session_state.update(cid=random.choice(ids)))
    with b:
        st.write("Typical customer per segment:")
        cols = st.columns(len(examples))
        for col, (_, r) in zip(cols, examples.iterrows()):
            col.button(f"{r['segment_name'].split()[0]} · {r['customer_id']}", key=f"ex_{r['customer_id']}",
                       on_click=lambda c=str(r["customer_id"]): st.session_state.update(cid=c),
                       use_container_width=True)

    cid = str(st.session_state["cid"]).strip()
    p = profile_of(cid) if cid else {"found": False, "message": "Enter a customer ID."}
    st.divider()

    if not p["found"]:
        st.warning(p["message"])
    else:
        L1, L4, L5, act = p["L1_attributes"], p["L4_model"], p["L5_explanations"], p["suggested_action"]
        prob = L4["propensity_90d"] or 0.0
        h1, h2, h3 = st.columns([2, 2, 3])
        with h1:
            st.markdown(f"### Customer {p['customer_id']}")
            st.markdown(f"<span class='badge' style='background:{seg_color(L4['segment'])}'>{L4['segment']}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"<p class='small'>profile as of {p['scoring_cutoff']} · market: {L1['market']} "
                        f"(descriptive only, not a model input)</p>", unsafe_allow_html=True)
        with h2:
            st.metric("Purchase propensity (next 90 days)", f"{prob:.0%}",
                      help=L4["meaning"])
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f"Decile **{L4['propensity_decile']}** of 10")
        with h3:
            st.markdown("**Suggested action**")
            st.success(act["action"])
            st.caption(f"Why: {act['reason']}")

        st.markdown("#### L1 · Account")
        k1, k2, k3 = st.columns(3)
        k1.metric("First purchase", L1["first_purchase_date"])
        k2.metric("Last purchase", L1["last_purchase_date"])
        k3.metric("Customer for", L1["customer_for"])

        c_left, c_right = st.columns(2)
        for col, title, key in ((c_left, "L2 · Purchasing behaviour", "L2_behaviour"),
                                (c_right, "L3 · Timing and trends", "L3_temporal")):
            with col:
                st.markdown(f"#### {title}")
                df = pd.DataFrame(p[key])[["label", "value"] + (["context"] if any("context" in f for f in p[key]) else [])]
                df = df.rename(columns={"label": "measure", "context": "compared with other active customers"})
                st.dataframe(df.fillna(""), use_container_width=True, hide_index=True, height=35 * (len(df) + 1) + 3)

        st.markdown("#### L5 · Why the score is what it is")
        drv = pd.DataFrame(L5["raising_score"] + L5["lowering_score"])
        if len(drv):
            drv["effect"] = drv["contribution_logodds"].map(lambda v: "raises score" if v > 0 else "lowers score")
            drv["name"] = drv["label"] + "  (" + drv["value"].astype(str) + ")"
            st.altair_chart(
                alt.Chart(drv).mark_bar(cornerRadius=4).encode(
                    x=alt.X("contribution_logodds:Q", title="contribution to the score (log-odds, vs average customer)"),
                    y=alt.Y("name:N", sort="-x", title=None),
                    color=alt.Color("effect:N", title=None,
                                    scale=alt.Scale(domain=["raises score", "lowers score"], range=["#16A34A", "#DC2626"])),
                    tooltip=["label", "value", "family", "contribution_logodds"]).properties(height=260),
                use_container_width=True)
        st.caption(L5["note"])

        with st.expander("Data notes, caveat and raw JSON"):
            for n in p["data_notes"]:
                st.write("• " + n)
            st.write("• " + act["caveat"])
            st.write("• " + L4["flag_note"])
            st.json(p, expanded=False)
        st.download_button("⬇️ Download profile (JSON)", json.dumps(p, indent=2, default=str),
                           file_name=f"profile_{p['customer_id']}.json", mime="application/json")

# ============================================================================= TARGET LIST
with tab_target:
    st.markdown("Build a contact list from the scores. Filters run on the current snapshot.")
    f1, f2, f3 = st.columns(3)
    segs = sorted(base["segment_name"].dropna().unique())
    pick = f1.multiselect("Segments", segs, default=segs)
    lo, hi = f2.slider("Propensity decile", 1, 10, (4, 8))
    min_pct = f3.slider("Minimum spend percentile", 0, 100, 0, step=5)

    sel = base[base["segment_name"].isin(pick)
               & base["propensity_decile"].between(lo, hi)
               & (base["pct_spend_365d"] * 100 >= min_pct)].copy()
    sel = sel.sort_values(["propensity_decile", "gross_spend_365d"], ascending=[False, False])

    m1, m2, m3 = st.columns(3)
    m1.metric("Customers selected", f"{len(sel):,}")
    m2.metric("Expected buyers (sum of propensities)", f"{sel['propensity_90d'].sum():,.0f}")
    m3.metric("Their last-12-month spend", f"£{sel['gross_spend_365d'].sum():,.0f}")

    sel["customer_id"] = sel["customer_id"].astype(str)
    show = sel[["customer_id", "segment_name", "propensity_90d", "propensity_decile", "gross_spend_365d",
                "recency_days", "frequency_365d", "suggested_action"]].rename(columns={
        "segment_name": "segment", "propensity_90d": "propensity", "propensity_decile": "decile",
        "gross_spend_365d": "spend_12m_gbp", "recency_days": "days_since_purchase", "frequency_365d": "orders_12m"})
    st.dataframe(show, use_container_width=True, hide_index=True, height=420,
                 column_config={"propensity": st.column_config.ProgressColumn("propensity", min_value=0.0,
                                                                              max_value=1.0, format="%.2f"),
                                "spend_12m_gbp": st.column_config.NumberColumn("spend 12m (£)", format="%.0f")})
    st.download_button("⬇️ Download list (CSV)", show.to_csv(index=False), file_name="target_list.csv", mime="text/csv")
    st.warning("Keep a **random hold-out group** (e.g. 10-20% of this list, not contacted). Comparing it with the "
               "contacted group is the only way to learn whether the action changes behaviour - the score alone "
               "cannot tell you that.")

# ============================================================================= MODEL
with tab_model:
    st.markdown(f"Chosen model: **{chosen}** · threshold {metrics.get('threshold', float('nan')):.3f} · "
                f"calibration applied: **{metrics.get('calibration', {}).get('applied')}** · "
                f"all choices made on validation; test scored once.")
    rows = []
    for split in ("valid", "test"):
        for name, r in metrics.get("results", {}).get(split, {}).items():
            k = next((x for x in r if x.startswith("precision_at_top")), None)
            rows.append({"split": split, "model": name, "ROC-AUC": r.get("roc_auc"), "PR-AUC": r.get("pr_auc"),
                         "top-20% precision": r.get(k) if k else None, "Brier": r.get("brier"),
                         "mean predicted": r.get("mean_predicted"), "prevalence": r.get("prevalence")})
    if rows:
        st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)
    ci = metrics.get("test_bootstrap", {})
    if ci:
        st.caption(f"Test ROC-AUC 95% CI {ci['roc_auc_95ci'][0]:.3f}-{ci['roc_auc_95ci'][1]:.3f} · "
                   f"PR-AUC 95% CI {ci['pr_auc_95ci'][0]:.3f}-{ci['pr_auc_95ci'][1]:.3f} (bootstrap, {ci['reps']} reps)")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Ranking - test window**")
        fig("step6_roc_pr_test.png")
    with g2:
        st.markdown("**Calibration - valid vs test**")
        fig("step6_calibration.png")

    st.markdown("#### What the evaluation showed (reported, not tuned away)")
    st.markdown(
        "1. **LightGBM and logistic regression are statistically tied** on test.\n"
        "2. **Seasonality:** test was peak season; the calendar feature moved predictions the right way but "
        "overshot. One peak season cannot estimate next year's peak strength.\n"
        "3. **The F1 threshold over-selects in peak season**, so targeting uses deciles.\n\n"
        "Details: `reports/model_findings.md`.")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Seasonality ablation (test)**")
        if metrics.get("ablation_test"):
            st.dataframe(pd.DataFrame(metrics["ablation_test"]).T.round(3), use_container_width=True)
    with a2:
        st.markdown("**Fairness audit (country used only here)**")
        if metrics.get("audit_test"):
            st.dataframe(pd.DataFrame(metrics["audit_test"]).T.round(3), use_container_width=True)

    st.markdown("#### Explainability")
    if expl:
        st.caption(f"Exact TreeSHAP (additivity error {expl['max_additivity_error']:.1e}); SHAP vs permutation "
                   f"rank agreement {expl['rank_agreement_shap_vs_perm']:.2f}. Associations, not causes.")
    e1, e2 = st.columns(2)
    with e1:
        fig("step7_shap_global.png")
    with e2:
        fig("step7_family_importance.png")
    fig("step7_shap_dependence.png")

st.divider()
st.caption("Public, anonymised data (UCI Online Retail II, CC BY 4.0). Profiles describe observed account behaviour "
           "and model-estimated propensities - no demographics, no psychological labels. Suggested actions are "
           "rules, not estimates of impact; test them with a randomised hold-out.")
