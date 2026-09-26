"""Step 8 - Customer profile engine.

    get_profile(customer_id)  -> dict with layers L1-L5, a suggested action and data notes
    render_text(profile)      -> readable text for the CLI / notebook

Every field comes from the warehouse view marts.customer_profile_base (sql/06_profile_view.sql),
which joins point-in-time features, the Step 5 segment, the Step 6 propensity and the Step 7
drivers. Nothing is inferred beyond what the data supports: no demographics, no psychological
labels - only observed behaviour and model-estimated propensities.

CLI (project root, .venv active):
    python -m src.profile_engine                 build the view + write reports/example_profiles.md
    python -m src.profile_engine 12346           print one customer's profile
    python -m src.profile_engine 12346 --json    same, as JSON
    python -m src.profile_engine --examples      list example customer IDs per segment
"""
from __future__ import annotations

import argparse
import json
import math

import pandas as pd

from src.config import PROJECT_ROOT, path
from src.db import connect
from src.explain import FEATURE_LABELS

VIEW = "marts.customer_profile_base"
SQL_FILE = "sql/06_profile_view.sql"

CAVEAT = ("Rule-based suggestion from observed behaviour and a predictive score. It is not an "
          "estimate of what the action would achieve; measure impact with a randomised hold-out group.")
FLAG_NOTE = ("The yes/no flag over-selects in peak season (see reports/model_findings.md); "
             "use the decile for targeting.")


# ----------------------------------------------------------------------------- formatting
def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def fmt(value, kind: str) -> str:
    v = _num(value)
    if v is None:
        return "n/a"
    if kind == "gbp":
        return f"£{v:,.0f}" if abs(v) >= 100 else f"£{v:,.2f}"
    if kind == "pct":
        return f"{v * 100:.0f}%"
    if kind == "days":
        return f"{v:,.0f} days"
    if kind == "int":
        return f"{v:,.0f}"
    if kind == "ratio":
        return f"{v:.2f}"
    if kind == "trend":
        change = math.exp(v) - 1          # log ratio -> % change (with the +1 smoothing)
        word = "up" if v > 0.1 else "down" if v < -0.1 else "flat"
        return f"{word} ({change * 100:+.0f}%)"
    return f"{v:,.1f}"


def pct_text(p) -> str | None:
    v = _num(p)
    if v is None:
        return None
    if v < 0.005:
        return "in the lowest group of active customers (tied)"
    return f"higher than {v * 100:.0f}% of active customers"


def trend_text(log_ratio, spend_last_90d) -> str:
    """Spend change, last 90 days vs the 90 days before, without absurd % when a period is ~zero."""
    lr, last = _num(log_ratio), _num(spend_last_90d)
    if lr is None:
        return "n/a"
    if last is None:
        return fmt(lr, "trend")
    prev = (last + 1) / math.exp(lr) - 1      # invert ln((last + 1) / (prev + 1))
    if last < 1 and prev < 1:
        return "flat (no spend in either period)"
    if prev < 1:
        return "up (no spend in the previous 90 days)"
    if last < 1:
        return "down (no spend in the last 90 days)"
    change = last / prev - 1
    word = "up" if change > 0.1 else "down" if change < -0.1 else "flat"
    return f"{word} ({change * 100:+.0f}%)"


def _date(v) -> str:
    try:
        return str(pd.Timestamp(v).date())
    except (TypeError, ValueError):
        return "n/a"


# (feature, format, percentile column or None)
L2_FIELDS = [
    ("frequency_365d", "int", "pct_frequency_365d"),
    ("gross_spend_365d", "gbp", "pct_spend_365d"),
    ("aov_365d", "gbp", "pct_aov_365d"),
    ("avg_products_per_order", "float", None),
    ("median_units_per_order", "int", None),
    ("distinct_products", "int", "pct_distinct_products"),
    ("distinct_product_groups", "int", None),
    ("top_group_share", "pct", None),
    ("christmas_share", "pct", None),
    ("repeat_sku_share", "pct", None),
    ("price_index", "ratio", None),
    ("cancel_rate", "ratio", None),
    ("cancel_value_share", "pct", None),
]
L3_FIELDS = [
    ("recency_days", "days", "pct_recency"),
    ("median_gap_days", "days", None),
    ("gap_cv", "ratio", None),
    ("overdue_ratio", "ratio", None),
    ("active_windows_12m", "int", "pct_active_windows_12m"),
    ("spend_trend_90d_log_ratio", "trend", None),
    ("order_trend_90d", "int", None),
    ("spend_slope_rel_12m", "ratio", None),
    ("q4_spend_share", "pct", None),
    ("morning_order_share", "pct", None),
]


KIND = {f: k for f, k, _ in L2_FIELDS + L3_FIELDS}
KIND.update({"frequency_total": "int", "frequency_90d": "int", "gross_spend_total": "gbp",
             "gross_spend_90d": "gbp", "median_order_value": "gbp", "avg_unit_price_paid": "gbp",
             "median_line_quantity": "int", "tenure_days": "days", "mean_gap_days": "days",
             "product_group_entropy": "ratio"})


def _fields(row: dict, spec) -> list[dict]:
    out = []
    for feat, kind, pcol in spec:
        value = (trend_text(row.get(feat), row.get("gross_spend_90d")) if kind == "trend"
                 else fmt(row.get(feat), kind))
        item = {"feature": feat, "label": FEATURE_LABELS[feat], "value": value, "raw": _num(row.get(feat))}
        if pcol:
            if feat == "recency_days":
                p = _num(row.get(pcol))
                item["context"] = (None if p is None else "among the least recent active customers (tied)"
                                   if p < 0.005 else f"more recent than {p * 100:.0f}% of active customers")
            else:
                item["context"] = pct_text(row.get(pcol))
        out.append(item)
    return out


# ----------------------------------------------------------------------------- action rules
def choose_action(segment: str | None, decile: int | None, pct_spend: float | None,
                  overdue_ratio: float | None, cancel_value_share: float | None) -> dict:
    """Transparent rules, evaluated in order. Returns action + the reason it was chosen."""
    seg = (segment or "").lower()
    d = int(decile) if decile is not None else None
    ps = _num(pct_spend)
    od = _num(overdue_ratio)
    cvs = _num(cancel_value_share)

    if "cancel" in seg:
        return {"action": "Review service first: look at why orders were cancelled before any marketing contact.",
                "reason": f"segment '{segment}'; {fmt(cvs, 'pct')} of spend value cancelled."}
    if d is not None and d >= 9:
        return {"action": "No incentive needed: keep stock and service reliable; standard newsletter only.",
                "reason": f"propensity decile {d} of 10 - already among the most likely to buy. "
                          "Discounts here would mostly go to purchases that were likely anyway (assumption to test)."}
    if d is not None and 4 <= d <= 8 and ps is not None and ps >= 0.75:
        return {"action": "Priority retention contact (personal, account-manager style).",
                "reason": f"high value (spend {pct_text(ps)}) but uncertain next purchase (decile {d})."}
    if "lapsed" in seg or (d is not None and d <= 3 and od is not None and od >= 2):
        why = f"segment '{segment}'" if "lapsed" in seg else f"low propensity (decile {d})"
        return {"action": "Low-cost win-back only (e.g. one email); no expensive outreach.",
                "reason": f"{why}; decile {d}; silent for longer than usual"
                          + (f" ({od:.1f}x their typical gap)." if od is not None else ".")}
    if "seasonal" in seg:
        return {"action": "Time contact ahead of the next seasonal buying window (Sep-Nov).",
                "reason": f"segment '{segment}': spend concentrated in the pre-Christmas period."}
    return {"action": "Standard communication; no special treatment.",
            "reason": f"no rule triggered (segment '{segment}', decile {d})."}


# ----------------------------------------------------------------------------- profile assembly
def build_profile(row: dict, drivers: pd.DataFrame, families: pd.DataFrame) -> dict:
    """Pure function: warehouse row + driver rows -> profile dict (JSON-serialisable)."""
    row = {**row, "customer_id": _clean_id(row.get("customer_id"))}
    freq_total = _num(row.get("frequency_total")) or 0
    notes = [f"Based on {freq_total:.0f} purchase invoice(s) before {_date(row.get('cutoff_date'))}."]
    if _num(row.get("median_gap_days")) is None:
        notes.append("Too few distinct order days to measure a purchase rhythm (gap features n/a).")
    elif _num(row.get("gap_cv")) is None:
        notes.append("Gap irregularity needs at least 3 order days (n/a).")
    notes.append("Behaviour is observed at account level; accounts may be businesses (wholesale), not individuals.")

    decile = _num(row.get("propensity_decile"))
    drv = drivers.sort_values(["direction", "rank"]) if len(drivers) else drivers

    def driver_list(direction):
        if not len(drv):
            return []
        sub = drv[drv["direction"] == direction]
        return [{"label": r["label"], "family": r["family"], "feature": r["feature"],
                 "value": fmt(r["feature_value"], KIND.get(r["feature"], "ratio")), "contribution_logodds": round(float(r["contribution_logodds"]), 3)}
                for _, r in sub.iterrows()]

    fam = ({r["family"]: round(float(r["contribution_logodds"]), 3) for _, r in
            families.sort_values("contribution_logodds", ascending=False).iterrows()} if len(families) else {})

    return {
        "found": True,
        "customer_id": row.get("customer_id"),
        "scoring_cutoff": _date(row.get("cutoff_date")),
        "L1_attributes": {
            "first_purchase_date": _date(row.get("first_purchase_date")),
            "last_purchase_date": _date(row.get("last_purchase_date")),
            "customer_for": fmt(row.get("tenure_days"), "days"),
            "market": row.get("market"),
            "market_note": "descriptive only; not used by any model, only for the fairness audit",
        },
        "L2_behaviour": _fields(row, L2_FIELDS),
        "L3_temporal": _fields(row, L3_FIELDS),
        "L4_model": {
            "segment": row.get("segment_name"),
            "segment_id": None if _num(row.get("segment_id")) is None else int(_num(row.get("segment_id"))),
            "propensity_90d": _num(row.get("propensity_90d")),
            "propensity_decile": None if decile is None else int(decile),
            "predicted_buyer_flag": None if _num(row.get("predicted_buyer")) is None else bool(row.get("predicted_buyer")),
            "flag_note": FLAG_NOTE,
            "meaning": "Model-estimated probability of at least one purchase in the next 90 days, "
                       "based on past behaviour (association, not cause).",
        },
        "L5_explanations": {
            "raising_score": driver_list("raises"),
            "lowering_score": driver_list("lowers"),
            "by_family_logodds": fam,
            "note": "SHAP contributions relative to the average customer; what the model associates "
                    "with this score, not causes.",
        },
        "suggested_action": {**choose_action(row.get("segment_name"), decile, row.get("pct_spend_365d"),
                                             row.get("overdue_ratio"), row.get("cancel_value_share")),
                             "caveat": CAVEAT},
        "data_notes": notes,
    }


def not_found(customer_id, n_active: int | None = None, cutoff=None) -> dict:
    scope = f"the {n_active:,} customers" if n_active else "customers"
    return {"found": False, "customer_id": customer_id,
            "message": (f"Customer {customer_id} is not among {scope} with a purchase in the 365 days before "
                        f"the scoring date{f' ({_date(cutoff)})' if cutoff else ''}. Profiles cover active, identified "
                        "customers only (anonymous transactions have no customer ID).")}


# ----------------------------------------------------------------------------- warehouse access
def _segment_columns(con) -> tuple[str | None, str]:
    cols = [r[0] for r in con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'marts' AND table_name = 'customer_segments' ORDER BY ordinal_position").fetchall()]
    if not cols:
        raise RuntimeError("marts.customer_segments not found - run `python -m src.segmentation` first")
    name_col = next((c for c in cols if "name" in c.lower()), None)
    if name_col is None:
        raise RuntimeError(f"no segment-name column in marts.customer_segments (columns: {cols})")
    id_col = next((c for c in cols if c not in (name_col, "customer_id")
                   and any(k in c.lower() for k in ("segment", "cluster"))), None)
    return id_col, name_col


def build_view(con) -> None:
    id_col, name_col = _segment_columns(con)
    sql = (PROJECT_ROOT / SQL_FILE).read_text(encoding="utf-8")
    sql = sql.replace("{segment_id_expr}", f's."{id_col}"' if id_col else "CAST(NULL AS INTEGER)")
    sql = sql.replace("{segment_name_col}", name_col)
    body = "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))
    con.execute(body)


_WHERE = "WHERE CAST(customer_id AS VARCHAR) = ? OR TRY_CAST(customer_id AS DOUBLE) = TRY_CAST(? AS DOUBLE)"


def get_profile(customer_id, con=None) -> dict:
    own = con is None
    con = con or connect(read_only=True)
    try:
        key = str(customer_id).strip()
        row = con.execute(f"SELECT * FROM {VIEW} {_WHERE}", [key, key]).df()
        if row.empty:
            n, cutoff = con.execute(f"SELECT COUNT(*), MAX(cutoff_date) FROM {VIEW}").fetchone()
            return not_found(customer_id, n, cutoff)
        drivers = con.execute(f"SELECT * FROM marts.customer_drivers {_WHERE}", [key, key]).df()
        families = con.execute(f"SELECT * FROM marts.customer_driver_families {_WHERE}", [key, key]).df()
    finally:
        if own:
            con.close()
    return build_profile(row.iloc[0].to_dict(), drivers, families)


def _clean_id(v):
    f = _num(v)
    return int(f) if f is not None and f.is_integer() else v


def example_ids(con, per_segment: int = 1) -> pd.DataFrame:
    """Customers closest to their segment's median propensity - typical members, not outliers."""
    return con.execute(f"""
        WITH seg AS (
            SELECT customer_id, segment_name, propensity_90d, propensity_decile,
                   MEDIAN(propensity_90d) OVER (PARTITION BY segment_name) AS seg_median
            FROM {VIEW}
        ), ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY segment_name
                                         ORDER BY ABS(propensity_90d - seg_median), customer_id) AS rn
            FROM seg
        )
        SELECT segment_name, customer_id, ROUND(propensity_90d, 3) AS propensity_90d, propensity_decile
        FROM ranked
        WHERE rn <= {int(per_segment)}
        ORDER BY propensity_90d DESC
    """).df()


# ----------------------------------------------------------------------------- rendering
def render_text(p: dict) -> str:
    if not p["found"]:
        return p["message"]
    L1, L4, L5, act = p["L1_attributes"], p["L4_model"], p["L5_explanations"], p["suggested_action"]
    lines = [f"CUSTOMER {p['customer_id']}  -  profile as of {p['scoring_cutoff']}", "=" * 64,
             "", "L1  Account",
             f"    first purchase {L1['first_purchase_date']} | last purchase {L1['last_purchase_date']} "
             f"| customer for {L1['customer_for']}",
             f"    market: {L1['market']}  ({L1['market_note']})"]
    for title, key in (("L2  Purchasing behaviour", "L2_behaviour"), ("L3  Timing and trends", "L3_temporal")):
        lines += ["", title]
        for f in p[key]:
            ctx = f"   [{f['context']}]" if f.get("context") else ""
            lines.append(f"    {f['label']:<42} {f['value']}{ctx}")
    prob = L4["propensity_90d"]
    lines += ["", "L4  Model outputs",
              f"    segment: {L4['segment']}",
              f"    purchase propensity (next 90 days): "
              f"{'n/a' if prob is None else f'{prob:.0%}'}  -  decile {L4['propensity_decile']} of 10",
              f"    ({L4['meaning']})",
              "", "L5  Why the score is what it is (associations)"]
    for label, key in (("raising", "raising_score"), ("lowering", "lowering_score")):
        for d in L5[key]:
            lines.append(f"    {'+' if label == 'raising' else '-'} {d['label']:<40} value {d['value']:<10} "
                         f"({d['contribution_logodds']:+.2f})")
    lines += ["", "Suggested action", f"    {act['action']}", f"    why: {act['reason']}",
              f"    note: {act['caveat']}", "", "Data notes"] + [f"    - {n}" for n in p["data_notes"]]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def write_examples(con) -> pd.DataFrame:
    ex = example_ids(con)
    parts = ["# Example Customer Profiles\n",
             "One typical customer per segment (closest to the segment's median propensity), generated by "
             "`python -m src.profile_engine`. Profiles describe observed behaviour and model-estimated "
             "propensities only.\n"]
    for _, r in ex.iterrows():
        parts.append(f"## {r['segment_name']}\n\n```text\n{render_text(get_profile(r['customer_id'], con))}\n```\n")
    (path("reports_dir") / "example_profiles.md").write_text("\n".join(parts), encoding="utf-8")
    return ex


def main() -> None:
    ap = argparse.ArgumentParser(description="Customer profile engine")
    ap.add_argument("customer_id", nargs="?", help="customer ID to profile")
    ap.add_argument("--json", action="store_true", help="print the profile as JSON")
    ap.add_argument("--examples", action="store_true", help="list example customer IDs per segment")
    a = ap.parse_args()

    if a.customer_id is None and not a.examples:
        con = connect()
        try:
            print("1/2 building view marts.customer_profile_base ...")
            build_view(con)
            n = con.execute(f"SELECT COUNT(*) FROM {VIEW}").fetchone()[0]
            print(f"    {n:,} customer profiles available")
            print("2/2 writing reports/example_profiles.md ...")
            ex = write_examples(con)
            print(ex.to_string(index=False))
            print("\n" + render_text(get_profile(ex.iloc[0]["customer_id"], con)))
        finally:
            con.close()
        return

    con = connect(read_only=True)
    try:
        if a.examples:
            print(example_ids(con, per_segment=3).to_string(index=False))
            return
        prof = get_profile(a.customer_id, con)
    finally:
        con.close()
    print(json.dumps(prof, indent=2, default=str) if a.json else render_text(prof))


if __name__ == "__main__":
    main()
