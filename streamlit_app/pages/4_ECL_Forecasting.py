"""Page 4: ECL Forecasting Engine — Per-grade dual-mode projector.

V6 Roadmap: Institutional-grade receivables forecast engine (mirrors HSBC PyCraft).
- Dual-mode: Operational (6-mo avg flat) vs CECL (3-phase macro-adjusted)
- Per-grade originations and liquidation sliders
- Grade-specific flow rates from flow_rates_by_scenario.csv
- Projected receivables tracker matching input receivables_tracker.csv format exactly
- Output: 13 columns × (8 buckets × 7 grades × N months) rows
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.data_loader import (
    load_flow_rates_extend, load_flow_rates_cecl,
    load_monthly_ecl_simple, load_alll_tracker,
    load_flow_through_rate, load_ecl_by_scenario,
    load_sensitivity_results, load_receivables_tracker,
    load_prepayment_rates,
)
from utils.styles import (
    inject_custom_css, sidebar_disclaimer, kpi_card, format_currency,
    GRADE_COLORS, GRADE_ORDER, SCENARIO_COLORS, DPD_COLORS,
)

PLOTLY_CONFIG = {"displayModeBar": True, "scrollZoom": True}
BUCKET_ORDER = ["Current", "30+", "60+", "90+", "120+", "150+", "180+", "GCO"]
# Map DPD bucket labels between input format and internal format
INPUT_BUCKET_MAP = {
    "Current": "Current", "30 DPD": "30+", "60 DPD": "60+", "90 DPD": "90+",
    "120 DPD": "120+", "150 DPD": "150+", "180+ DPD": "180+", "GCO": "GCO",
}
OUTPUT_BUCKET_MAP = {v: k for k, v in INPUT_BUCKET_MAP.items()}  # reverse
RATE_COLS = [
    "flow_rate_30", "flow_rate_60", "flow_rate_90",
    "flow_rate_120", "flow_rate_150", "flow_rate_180", "flow_rate_gco",
]
TRANSITIONS = [
    "Current→30+", "30+→60+", "60+→90+", "90+→120+",
    "120+→150+", "150+→180+", "180+→GCO",
]

# Historical grade mix (% of total balance) for default origination allocation
GRADE_MIX_DEFAULT = {
    "A": 16.8, "B": 26.8, "C": 27.9, "D": 15.8, "E": 8.5, "F": 3.2, "G": 1.0,
}
# Default liquidation factors by grade — derived from CPR (prepayment model)
# Uses latest vintage (2016-2018) average CPR across terms, converted to monthly SMM
def _load_cpr_defaults():
    """Load CPR rates from prepayment model and convert to monthly SMM for liquidation."""
    try:
        pr = load_prepayment_rates()
        if len(pr) > 0:
            # Use latest vintage group, average across terms
            latest = pr[pr["vintage_group"] == "2016-2018"]
            if len(latest) == 0:
                latest = pr
            grade_smm = latest.groupby("grade")["smm"].mean()
            return {g: round(grade_smm.get(g, 0.02) * 100, 1) for g in GRADE_ORDER}
    except Exception:
        pass
    # Fallback hardcoded values (based on typical LendingClub CPR by grade)
    return {"A": 2.5, "B": 2.2, "C": 2.0, "D": 1.8, "E": 1.6, "F": 1.4, "G": 1.2}

GRADE_LIQ_DEFAULT = _load_cpr_defaults()


# ═══════════════════════════════════════════════════════════════════════════════
# PROJECTION ENGINE — PER-GRADE
# ═══════════════════════════════════════════════════════════════════════════════


def get_grade_flow_rates(scenario_fr: pd.DataFrame, grade: str, scenario: str) -> dict:
    """Get flow rates for a specific grade and scenario."""
    row = scenario_fr[(scenario_fr["grade"] == grade) & (scenario_fr["scenario"] == scenario)]
    if len(row) == 0:
        return {c: 0.0 for c in RATE_COLS}
    return {c: row[c].values[0] for c in RATE_COLS if c in row.columns}


def get_grade_snapshot(recv: pd.DataFrame, grade: str, snapshot_date) -> dict:
    """Get receivables snapshot for a specific grade at a specific date."""
    df = recv[(recv["grade"] == grade) & (recv["month_date"] == snapshot_date)]
    snap = df.set_index("dpd_bucket")["balance"].to_dict()
    # Map input DPD labels to internal labels
    result = {}
    for input_label, internal_label in INPUT_BUCKET_MAP.items():
        result[internal_label] = snap.get(input_label, snap.get(internal_label, 0.0))
    return result


def project_grade(
    snapshot: dict,
    flow_rates: dict,
    n_months: int,
    new_origination: float,
    liquidation_factor: float,
    recovery_rate: float,
    mode: str,
    stressed_rates: dict | None,
    historical_rates: dict | None,
    grade: str,
    snapshot_date,
) -> pd.DataFrame:
    """Project receivables forward for a SINGLE grade.

    Returns DataFrame with one row per month per DPD bucket,
    matching the input receivables_tracker.csv format exactly.
    """
    buckets = ["Current", "30+", "60+", "90+", "120+", "150+", "180+"]
    bal = {b: snapshot.get(b, 0.0) for b in buckets}
    rows = []
    base_date = snapshot_date

    cumulative_ecl_simple = 0.0
    cumulative_ecl_dcf = 0.0

    for month in range(1, n_months + 1):
        # CECL 3-phase rate selection
        if mode == "cecl" and stressed_rates and historical_rates:
            if month <= 24:
                rates = stressed_rates
            elif month <= 36:
                w = (36 - month) / 12.0
                rates = {
                    k: w * stressed_rates.get(k, 0) + (1 - w) * historical_rates.get(k, 0)
                    for k in RATE_COLS
                }
            else:
                rates = historical_rates
        else:
            rates = flow_rates

        # Forward transitions
        new_bal = {}
        gco_amount = bal.get("180+", 0) * rates.get("flow_rate_gco", 0)
        new_bal["30+"] = bal["Current"] * rates.get("flow_rate_30", 0)
        new_bal["60+"] = bal["30+"] * rates.get("flow_rate_60", 0)
        new_bal["90+"] = bal["60+"] * rates.get("flow_rate_90", 0)
        new_bal["120+"] = bal["90+"] * rates.get("flow_rate_120", 0)
        new_bal["150+"] = bal["120+"] * rates.get("flow_rate_150", 0)
        new_bal["180+"] = bal["150+"] * rates.get("flow_rate_180", 0)

        # Current: minus outflow, minus liquidation, plus originations
        current_outflow = new_bal["30+"]
        current_runoff = bal["Current"] * liquidation_factor
        new_bal["Current"] = max(0, bal["Current"] - current_outflow - current_runoff + new_origination)

        recovery_amount = gco_amount * recovery_rate
        nco_amount = gco_amount - recovery_amount
        total_balance = sum(new_bal.values())

        # ECL calculations
        # Simple: NCO is the realized monthly loss
        ecl_simple_monthly = nco_amount
        # DCF: discount at monthly rate
        # FIX (Issue #5): use the user-selected discount_rate from the sidebar slider,
        # not a hardcoded 0.05 that silently ignores every interactive scenario.
        monthly_discount = (1 + discount_rate) ** (1 / 12) - 1
        ecl_dcf_monthly = nco_amount / ((1 + monthly_discount) ** month)

        cumulative_ecl_simple += ecl_simple_monthly
        cumulative_ecl_dcf += ecl_dcf_monthly

        loss_rate = nco_amount / total_balance if total_balance > 0 else 0
        alll_simple = cumulative_ecl_simple / total_balance if total_balance > 0 else 0
        alll_dcf = cumulative_ecl_dcf / total_balance if total_balance > 0 else 0

        proj_date = base_date + pd.DateOffset(months=month)
        date_str = proj_date.strftime("%Y-%m-%d")

        # One row per DPD bucket (matching input format exactly)
        for bucket in BUCKET_ORDER:
            if bucket == "GCO":
                bucket_bal = gco_amount
                bucket_gco = gco_amount
                bucket_recovery = recovery_amount
                bucket_nco = nco_amount
            else:
                bucket_bal = new_bal.get(bucket, 0)
                bucket_gco = 0.0
                bucket_recovery = 0.0
                bucket_nco = 0.0

            rows.append({
                "month_date": date_str,
                "dpd_bucket": OUTPUT_BUCKET_MAP.get(bucket, bucket),
                "grade": grade,
                "balance": bucket_bal,
                "count": 0,  # Not available from projection
                "gco_amount": bucket_gco,
                "recovery_amount": bucket_recovery,
                "nco_amount": bucket_nco,
                # FIX (Issue #12): loss_rate is a portfolio-level monthly NCO rate
                # (nco_amount / total_balance), not a GCO-bucket metric. Storing
                # it only on GCO rows means downstream aggregations across DPD
                # buckets miss 7 of 8 rows per grade per month. Belongs on "Current"
                # (the performing book row, which is the natural denominator).
                "loss_rate": loss_rate if bucket == "Current" else 0.0,
                "total_ecl_simple": ecl_simple_monthly if bucket == "Current" else 0.0,
                "total_ecl_dcf": ecl_dcf_monthly if bucket == "Current" else 0.0,
                "alll_ratio_simple": alll_simple if bucket == "Current" else 0.0,
                "alll_ratio_dcf": alll_dcf if bucket == "Current" else 0.0,
            })

        bal = new_bal

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

try:
    st.set_page_config(page_title="ECL Forecasting Engine", layout="wide")
except Exception:
    pass
inject_custom_css()
sidebar_disclaimer()

st.title("ECL Forecasting Engine")
st.markdown(
    "Dual-mode ECL projector: **Operational Forecast** (6-month rolling extend) "
    "vs **CECL Reserve** (3-phase macro-adjusted). Per-grade projection with "
    "grade-specific flow rates, originations, and liquidation factors."
)

st.info(
    "**ECL Data Note:** ECL projections use flow rates derived from synthetic monthly panel "
    "reconstruction. Dollar amounts are approximate. The framework is identical to production "
    "implementation with observed payment data."
)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: Mode, Scenario, Assumptions
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("### Forecasting Mode")
mode = st.sidebar.radio(
    "Select Mode",
    ["Operational Forecast (Extend)", "CECL Reserve Estimation"],
    index=1,
    help="**Operational**: 6-month rolling average extended flat. **CECL**: 3-phase macro-adjusted.",
)
is_cecl = "CECL" in mode
mode_label = "CECL Reserve" if is_cecl else "Operational Forecast"

st.sidebar.markdown("---")
st.sidebar.markdown("### FEG Scenario")
scenario = st.sidebar.selectbox(
    "Select FEG View",
    ["Pre-FEG (Pure Model)", "Central (Baseline Macro)", "Post-FEG (Weighted Scenarios)"],
    index=1,
)
feg_to_key = {
    "Pre-FEG (Pure Model)": "baseline",
    "Central (Baseline Macro)": "central",
    "Post-FEG (Weighted Scenarios)": "stress",
}
scenario_key = feg_to_key.get(scenario, "central")

st.sidebar.markdown("---")
st.sidebar.markdown("### Global Assumptions")

recovery_rate = st.sidebar.slider(
    "Recovery Rate (%)", 5, 30, 17, 1,
    help="% of charged-off balance recovered. LendingClub avg ~17%.",
) / 100

if is_cecl:
    discount_rate = st.sidebar.slider(
        "Discount Rate (% annual)", 0, 15, 5, 1,
        help="Discount rate for present-value ECL calculation.",
    ) / 100
else:
    discount_rate = 0.05

horizon_years = st.sidebar.selectbox(
    "Projection Horizon", [5, 10, 15, 20], index=1,
    help="Number of years to project forward.",
)
n_months = horizon_years * 12

# --- Per-Grade Originations ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Per-Grade Originations ($M/mo)")

total_orig = st.sidebar.number_input(
    "Total Monthly Originations ($M)",
    min_value=0, max_value=1000, value=0 if is_cecl else 100, step=10,
    help="Total monthly originations, distributed by grade mix below. $0 for CECL (closed portfolio).",
)

grade_originations = {}
if total_orig > 0:
    with st.sidebar.expander("Grade Allocation (% of total)", expanded=False):
        remaining = 100.0
        for i, g in enumerate(GRADE_ORDER):
            default_pct = GRADE_MIX_DEFAULT.get(g, 10.0)
            if i == len(GRADE_ORDER) - 1:
                # Last grade gets remainder
                pct = remaining
                st.sidebar.text(f"Grade {g}: {pct:.1f}% (remainder)")
            else:
                pct = st.sidebar.number_input(
                    f"Grade {g} (%)", min_value=0.0, max_value=100.0,
                    value=min(default_pct, remaining), step=1.0, key=f"orig_{g}",
                )
            grade_originations[g] = total_orig * pct / 100.0 * 1_000_000
            remaining = max(0, remaining - pct)
else:
    for g in GRADE_ORDER:
        grade_originations[g] = 0.0

# --- Per-Grade Liquidation ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Per-Grade Liquidation (% monthly)")
st.sidebar.caption("Defaults from CPR model (Notebook 055 — Kaplan-Meier survival analysis)")

with st.sidebar.expander("Liquidation Factors by Grade", expanded=False):
    grade_liquidation = {}
    for g in GRADE_ORDER:
        default_liq = GRADE_LIQ_DEFAULT.get(g, 2.0)
        liq = st.sidebar.slider(
            f"Grade {g}", 0.0, 10.0, default_liq, 0.1, key=f"liq_{g}",
        )
        grade_liquidation[g] = liq / 100.0

# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD / EXPORT ASSUMPTIONS (V6 Requirement)
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown("### Assumptions I/O")

uploaded_assumptions = st.sidebar.file_uploader(
    "Upload Assumptions (.xlsx)",
    type=["xlsx"],
    help="Upload an Excel file with grade-level assumptions (originations, liquidation, flow rates).",
)

if uploaded_assumptions is not None:
    try:
        xl = pd.read_excel(uploaded_assumptions, sheet_name=None)
        if "Grade Assumptions" in xl:
            ga = xl["Grade Assumptions"]
            for _, row in ga.iterrows():
                g = row.get("grade", "")
                if g in GRADE_ORDER:
                    if "origination_monthly" in row and pd.notna(row["origination_monthly"]):
                        grade_originations[g] = float(row["origination_monthly"])
                    if "liquidation_factor" in row and pd.notna(row["liquidation_factor"]):
                        grade_liquidation[g] = float(row["liquidation_factor"])
            st.sidebar.success("Assumptions loaded from file.")
        if "Global Assumptions" in xl:
            glob = xl["Global Assumptions"]
            if len(glob) > 0:
                gr = glob.iloc[0]
                if "recovery_rate" in gr and pd.notna(gr["recovery_rate"]):
                    recovery_rate = float(gr["recovery_rate"])
                if "discount_rate" in gr and pd.notna(gr["discount_rate"]):
                    discount_rate = float(gr["discount_rate"])
    except Exception as e:
        st.sidebar.error(f"Failed to parse assumptions: {e}")

# Export current assumptions button
assumptions_export_rows = []
for g in GRADE_ORDER:
    assumptions_export_rows.append({
        "grade": g,
        "origination_monthly": grade_originations.get(g, 0),
        "liquidation_factor": grade_liquidation.get(g, 0.02),
    })
assumptions_export_df = pd.DataFrame(assumptions_export_rows)

global_export_df = pd.DataFrame([{
    "mode": mode_label,
    "scenario": scenario_key,
    "recovery_rate": recovery_rate,
    "discount_rate": discount_rate,
    "horizon_years": horizon_years,
    "total_monthly_origination": sum(grade_originations.values()),
}])

exp_buf = io.BytesIO()
with pd.ExcelWriter(exp_buf, engine="openpyxl") as writer:
    assumptions_export_df.to_excel(writer, sheet_name="Grade Assumptions", index=False)
    global_export_df.to_excel(writer, sheet_name="Global Assumptions", index=False)

st.sidebar.download_button(
    label="Export Current Assumptions (.xlsx)",
    data=exp_buf.getvalue(),
    file_name="ecl_assumptions.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

recv = load_receivables_tracker()
recv["month_date"] = pd.to_datetime(recv["month_date"])
ecl_scen = load_ecl_by_scenario()
sens = load_sensitivity_results()

RESULTS = Path(__file__).resolve().parent.parent.parent / "data" / "results"
scenario_fr = pd.read_csv(RESULTS / "flow_rates_by_scenario.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINE SNAPSHOT DATE (peak portfolio balance)
# ═══════════════════════════════════════════════════════════════════════════════

monthly_total = recv.groupby("month_date")["balance"].sum()
snapshot_date = monthly_total.idxmax()

# ═══════════════════════════════════════════════════════════════════════════════
# RUN PER-GRADE PROJECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

all_grade_projections = []
grade_summaries = {}

for grade in GRADE_ORDER:
    # Grade-specific flow rates
    hist_rates = get_grade_flow_rates(scenario_fr, grade, "baseline")
    scen_rates = get_grade_flow_rates(scenario_fr, grade, scenario_key)

    if is_cecl:
        primary = scen_rates
    else:
        primary = scen_rates if scenario_key != "baseline" else hist_rates

    # Grade-specific snapshot
    snap = get_grade_snapshot(recv, grade, snapshot_date)

    # Run projection
    proj = project_grade(
        snapshot=snap,
        flow_rates=primary,
        n_months=n_months,
        new_origination=grade_originations.get(grade, 0.0),
        liquidation_factor=grade_liquidation.get(grade, 0.02),
        recovery_rate=recovery_rate,
        mode="cecl" if is_cecl else "operational",
        stressed_rates=scen_rates if is_cecl else None,
        historical_rates=hist_rates if is_cecl else None,
        grade=grade,
        snapshot_date=snapshot_date,
    )

    all_grade_projections.append(proj)

    # Summary metrics for this grade
    grade_nco = proj[proj["dpd_bucket"] == "GCO"]["nco_amount"].sum()
    grade_start = sum(snap.values())
    _ftr_rates = [v for v in primary.values() if v > 0]
    grade_ftr = np.prod(_ftr_rates) if len(_ftr_rates) > 0 else 0.0
    grade_summaries[grade] = {
        "starting_balance": grade_start,
        "total_nco": grade_nco,
        "ftr": grade_ftr,
        "origination": grade_originations.get(grade, 0.0),
        "liquidation": grade_liquidation.get(grade, 0.02),
        "rates": primary,
        "hist_rates": hist_rates,
        "scen_rates": scen_rates,
    }

# Combine all grades
projection_full = pd.concat(all_grade_projections, ignore_index=True)

# Aggregate portfolio-level for charts
proj_agg_rows = []
for month_date in projection_full["month_date"].unique():
    month_data = projection_full[projection_full["month_date"] == month_date]
    month_num = len(proj_agg_rows) + 1
    row = {"month": month_num, "month_date": month_date}
    for bucket in BUCKET_ORDER:
        bucket_label = OUTPUT_BUCKET_MAP.get(bucket, bucket)
        bd = month_data[month_data["dpd_bucket"] == bucket_label]
        row[bucket] = bd["balance"].sum()
    row["GCO_amount"] = month_data[month_data["dpd_bucket"].isin(["GCO"])]["gco_amount"].sum()
    row["Recovery"] = month_data[month_data["dpd_bucket"].isin(["GCO"])]["recovery_amount"].sum()
    row["NCO"] = month_data[month_data["dpd_bucket"].isin(["GCO"])]["nco_amount"].sum()
    row["Total_Balance"] = sum(row.get(b, 0) for b in ["Current", "30+", "60+", "90+", "120+", "150+", "180+"])
    row["ECL_Monthly"] = row["NCO"]
    proj_agg_rows.append(row)

projection = pd.DataFrame(proj_agg_rows)
projection["Cumulative_ECL"] = projection["ECL_Monthly"].cumsum()
projection["Cumulative_NCO"] = projection["NCO"].cumsum()
raw_alll = projection["Cumulative_ECL"] / projection["Total_Balance"].replace(0, np.nan)
projection["ALLL_Ratio"] = raw_alll.clip(upper=1.0)
starting_balance = sum(
    grade_summaries[g]["starting_balance"] for g in GRADE_ORDER if g in grade_summaries
)
projection["Loss_Rate_vs_Start"] = projection["Cumulative_ECL"] / max(starting_balance, 1)

# ═══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"### {mode_label} — Key Metrics ({scenario})")

total_cum_ecl = projection["Cumulative_ECL"].iloc[-1]
year1_ecl = projection.loc[projection["month"] <= 12, "ECL_Monthly"].sum()
total_origination = sum(grade_originations.values())
# Portfolio-weighted FTR
total_start = sum(grade_summaries[g]["starting_balance"] for g in GRADE_ORDER)
if total_start > 0:
    wtd_ftr = sum(
        grade_summaries[g]["ftr"] * grade_summaries[g]["starting_balance"]
        for g in GRADE_ORDER
    ) / total_start
else:
    wtd_ftr = 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    kpi_card("Mode", mode_label)
with c2:
    kpi_card("Starting Balance", format_currency(starting_balance))
with c3:
    kpi_card("Year 1 NCO", format_currency(year1_ecl))
with c4:
    kpi_card(f"{horizon_years}Y Cumulative NCO", format_currency(total_cum_ecl))
with c5:
    kpi_card("Monthly Origination", format_currency(total_origination))
with c6:
    kpi_card("Wtd FTR", f"{wtd_ftr:.4%}")

# Milestones
st.markdown("#### ECL Milestones")
milestone_cols = st.columns(4)
milestones = {"Year 1": 12, "Year 3": 36, "Year 5": 60, f"Year {horizon_years}": n_months}
for col, (label, months) in zip(milestone_cols, milestones.items()):
    with col:
        if len(projection) >= months:
            val = projection.loc[projection["month"] <= months, "ECL_Monthly"].sum()
            kpi_card(label + " ECL", format_currency(val))
        else:
            kpi_card(label + " ECL", "N/A")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# FLOW RATES TABLE — PER GRADE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### Flow Rates Used for Projection (by Grade)")
st.markdown("Grade-specific flow rates from `flow_rates_by_scenario.csv`. These drive the per-grade forward projection.")

# Build flow rates table: rows = grades, columns = transitions
fr_table_rows = []
for g in GRADE_ORDER:
    gs = grade_summaries[g]
    row = {"Grade": g}
    for rc, trans in zip(RATE_COLS, TRANSITIONS):
        row[trans] = gs["rates"].get(rc, 0)
    row["FTR"] = gs["ftr"]
    fr_table_rows.append(row)

fr_table = pd.DataFrame(fr_table_rows)
fr_display = fr_table.copy()
for col in fr_display.columns[1:]:
    fr_display[col] = fr_display[col].apply(lambda x: f"{x:.4%}")

st.dataframe(fr_display, use_container_width=True, hide_index=True)

if is_cecl:
    st.caption(
        "**CECL 3-Phase:** Scenario rates used for months 1-24 (R&S period), "
        "linear reversion months 25-36, baseline rates from month 37 onward."
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECTED RECEIVABLES TRACKER (Institutional Format)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### Projected Receivables Tracker (Institutional Format)")
st.markdown(
    "Forward projection in the same format as the historical receivables tracker. "
    f"Starting from **{snapshot_date.strftime('%Y-%m')}** snapshot, "
    f"projected **{horizon_years} years** forward."
)

# Show milestone months
milestone_months = [1, 3, 6, 12, 24, 36, 60, 84, 120]
milestone_months = [m for m in milestone_months if m <= n_months]

proj_milestones = projection[projection["month"].isin(milestone_months)].set_index("month")

dollar_rows = {}
for bucket in BUCKET_ORDER:
    if bucket in proj_milestones.columns:
        dollar_rows[bucket] = proj_milestones[bucket]
dollar_rows["Total Balance"] = proj_milestones["Total_Balance"]
dollar_rows["GCO (Monthly)"] = proj_milestones["GCO_amount"] if "GCO_amount" in proj_milestones.columns else proj_milestones.get("GCO", 0)
dollar_rows["Recovery"] = proj_milestones["Recovery"]
dollar_rows["NCO"] = proj_milestones["NCO"]
dollar_rows["Cumulative NCO"] = proj_milestones["Cumulative_NCO"]
dollar_rows["ALLL Ratio"] = proj_milestones["ALLL_Ratio"]

tracker_df = pd.DataFrame(dollar_rows).T
tracker_df.columns = [f"Month {m}" for m in milestone_months]

tracker_display = tracker_df.copy()
for col in tracker_display.columns:
    tracker_display[col] = tracker_display.apply(
        lambda row: f"{row[col]:.2%}" if row.name == "ALLL Ratio"
        else format_currency(row[col]) if pd.notna(row[col]) else "—",
        axis=1,
    )

st.dataframe(tracker_display, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD — EXACT INPUT FORMAT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("#### Download Forecast")
st.markdown(
    "Output file matches the input `receivables_tracker.csv` format exactly: "
    "13 columns, one row per DPD bucket × grade × month."
)

# The projection_full DataFrame is already in the correct format
# Columns: month_date, dpd_bucket, grade, balance, count, gco_amount,
#          recovery_amount, nco_amount, loss_rate, total_ecl_simple,
#          total_ecl_dcf, alll_ratio_simple, alll_ratio_dcf

buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    # Sheet 1: Projected receivables (exact input format)
    projection_full.to_excel(writer, sheet_name="Projected Receivables", index=False)

    # Sheet 2: Historical + Projected combined
    hist_recv = recv.copy()
    hist_recv["source"] = "historical"
    proj_out = projection_full.copy()
    proj_out["source"] = "projected"
    combined = pd.concat([hist_recv, proj_out], ignore_index=True)
    combined.to_excel(writer, sheet_name="Combined (Hist+Proj)", index=False)

    # Sheet 3: Flow rates by grade
    fr_table.to_excel(writer, sheet_name="Flow Rates by Grade", index=False)

    # Sheet 4: Assumptions
    assumptions_rows = []
    for g in GRADE_ORDER:
        assumptions_rows.append({
            "grade": g,
            "origination_monthly": grade_originations.get(g, 0),
            "liquidation_factor": grade_liquidation.get(g, 0.02),
            "ftr": grade_summaries[g]["ftr"],
            "starting_balance": grade_summaries[g]["starting_balance"],
        })
    assumptions_df = pd.DataFrame(assumptions_rows)
    assumptions_df.to_excel(writer, sheet_name="Grade Assumptions", index=False)

    # Sheet 5: Global assumptions
    global_assumptions = pd.DataFrame([{
        "mode": mode_label,
        "scenario": scenario_key,
        "recovery_rate": recovery_rate,
        "discount_rate": discount_rate,
        "horizon_years": horizon_years,
        "snapshot_date": str(snapshot_date.date()),
        "starting_balance": starting_balance,
        "total_monthly_origination": total_origination,
    }])
    global_assumptions.to_excel(writer, sheet_name="Global Assumptions", index=False)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        label="Download Forecast (Excel)",
        data=buf.getvalue(),
        file_name=f"receivables_forecast_{mode_label.lower().replace(' ', '_')}_{scenario_key}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with col_dl2:
    csv_buf = io.StringIO()
    projection_full.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download Forecast (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"receivables_forecast_{scenario_key}.csv",
        mime="text/csv",
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECTED RECEIVABLES & ECL OVER TIME
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### Projected Receivables & ECL Over Time")

fig = go.Figure()
for bucket in ["Current", "30+", "60+", "90+", "120+", "150+", "180+"]:
    if bucket in projection.columns:
        fig.add_trace(go.Scatter(
            x=projection["month"], y=projection[bucket],
            name=bucket, stackgroup="one",
            line=dict(color=DPD_COLORS.get(bucket, "#999")),
        ))

fig.add_trace(go.Scatter(
    x=projection["month"], y=projection["Cumulative_ECL"],
    name="Cumulative ECL", yaxis="y2",
    line=dict(color="#e74c3c", width=3, dash="dot"),
))

fig.update_layout(
    title="Projected Balance by DPD Bucket with Cumulative ECL",
    template="plotly_white", height=500,
    xaxis_title="Projection Month",
    yaxis_title="Balance ($)",
    yaxis2=dict(title="Cumulative ECL ($)", overlaying="y", side="right"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

# ═══════════════════════════════════════════════════════════════════════════════
# GCO / NCO / RECOVERY OVER TIME
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### GCO / NCO / Recovery Over Time")

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=projection["month"], y=projection["NCO"],
        name="Net Charge-Off", stackgroup="one",
        line=dict(color="#e74c3c"),
    ))
    fig.add_trace(go.Scatter(
        x=projection["month"], y=projection["Recovery"],
        name="Recovery", stackgroup="one",
        line=dict(color="#27ae60"),
    ))
    fig.update_layout(
        title="Monthly GCO Breakdown (NCO + Recovery)",
        template="plotly_white", height=400,
        xaxis_title="Month", yaxis_title="Amount ($)",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=projection["month"], y=projection["Cumulative_NCO"],
        name="Cumulative NCO",
        line=dict(color="#e74c3c", width=2),
        fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=projection["month"], y=projection["Cumulative_ECL"],
        name="Cumulative ECL",
        line=dict(color="#2c3e50", width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Cumulative NCO & ECL",
        template="plotly_white", height=400,
        xaxis_title="Month", yaxis_title="Cumulative Amount ($)",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# ALLL RATIO & LOSS RATE TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### ALLL Ratio & Loss Rate Trajectory")

col_alll1, col_alll2 = st.columns(2)

with col_alll1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=projection["month"], y=projection["ALLL_Ratio"],
        mode="lines", line=dict(color="#2c3e50", width=2),
        name="ALLL Ratio (ECL / Balance)",
        fill="tozeroy", fillcolor="rgba(52, 152, 219, 0.1)",
    ))
    if is_cecl:
        fig.add_vline(x=24, line_dash="dash", line_color="orange",
                      annotation_text="Phase 1→2")
        fig.add_vline(x=36, line_dash="dash", line_color="red",
                      annotation_text="Phase 2→3")
    fig.update_layout(
        title="ALLL Ratio (Cumulative ECL / Outstanding Balance)",
        template="plotly_white", height=400,
        xaxis_title="Projection Month",
        yaxis_title="ALLL Ratio", yaxis_tickformat=".1%",
        yaxis_range=[0, min(1.0, float(projection["ALLL_Ratio"].max()) * 1.1)],
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

with col_alll2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=projection["month"], y=projection["Loss_Rate_vs_Start"],
        mode="lines", line=dict(color="#e74c3c", width=2),
        name="Cumulative Loss Rate",
        fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.1)",
    ))
    if is_cecl:
        fig.add_vline(x=24, line_dash="dash", line_color="orange",
                      annotation_text="Phase 1→2")
        fig.add_vline(x=36, line_dash="dash", line_color="red",
                      annotation_text="Phase 2→3")
    fig.update_layout(
        title="Cumulative Loss Rate (vs Starting Balance)",
        template="plotly_white", height=400,
        xaxis_title="Projection Month",
        yaxis_title="Cumulative Loss %", yaxis_tickformat=".1%",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# FLOW RATE PROJECTIONS (3-phase visualization)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### Flow Rate Projections")
st.markdown(
    f"**{mode_label}** — "
    + ("Three-phase: 24mo macro-adjusted → 12mo reversion → historical baseline"
       if is_cecl else "Flat scenario rates across entire projection horizon")
)

# Use Grade B as representative for the flow rate timeline chart
rep_grade = "B"
hist_r = grade_summaries[rep_grade]["hist_rates"]
scen_r = grade_summaries[rep_grade]["scen_rates"]

fr_timeline = []
for month in range(1, n_months + 1):
    if is_cecl:
        if month <= 24:
            rates = scen_r
        elif month <= 36:
            w = (36 - month) / 12.0
            rates = {k: w * scen_r.get(k, 0) + (1 - w) * hist_r.get(k, 0) for k in RATE_COLS}
        else:
            rates = hist_r
    else:
        rates = grade_summaries[rep_grade]["rates"]
    for col, trans in zip(RATE_COLS, TRANSITIONS):
        fr_timeline.append({"month": month, "transition": trans, "rate": rates.get(col, 0)})

fr_timeline_df = pd.DataFrame(fr_timeline)

col_fr1, col_fr2 = st.columns(2)

entry_df = fr_timeline_df[fr_timeline_df["transition"] == "Current→30+"]
roll_df = fr_timeline_df[fr_timeline_df["transition"] != "Current→30+"]

with col_fr1:
    fig = px.line(entry_df, x="month", y="rate", color="transition",
                  title=f"Entry Rate: Current → 30+ DPD (Grade {rep_grade})")
    fig.update_layout(
        template="plotly_white", height=400,
        xaxis_title="Month", yaxis_title="Flow Rate", yaxis_tickformat=".2%",
        showlegend=False,
    )
    if is_cecl:
        fig.add_vline(x=24, line_dash="dash", line_color="orange", annotation_text="Phase 1→2")
        fig.add_vline(x=36, line_dash="dash", line_color="red", annotation_text="Phase 2→3")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

with col_fr2:
    fig = px.line(roll_df, x="month", y="rate", color="transition",
                  title=f"DPD Roll Rates (Grade {rep_grade})")
    fig.update_layout(
        template="plotly_white", height=400,
        xaxis_title="Month", yaxis_title="Flow Rate", yaxis_tickformat=".1%",
        legend_title="Transition",
    )
    if is_cecl:
        fig.add_vline(x=24, line_dash="dash", line_color="orange", annotation_text="Phase 1→2")
        fig.add_vline(x=36, line_dash="dash", line_color="red", annotation_text="Phase 2→3")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# ECL BY GRADE & SCENARIO (from pre-computed data)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### ECL by Grade & Scenario")
col3, col4 = st.columns(2)

with col3:
    grade_ecl = ecl_scen[ecl_scen["scenario"] == scenario_key].copy()
    if len(grade_ecl) > 0:
        fig = go.Figure(go.Bar(
            x=grade_ecl["grade"], y=grade_ecl["total_ecl"],
            marker_color=[GRADE_COLORS.get(g, "#999") for g in grade_ecl["grade"]],
            text=[format_currency(v) for v in grade_ecl["total_ecl"]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"ECL by Grade ({scenario_key.title()})",
            template="plotly_white", height=400, yaxis_title="ECL ($)",
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

with col4:
    if len(grade_ecl) > 0:
        fig = go.Figure(go.Bar(
            x=grade_ecl["grade"], y=grade_ecl["alll_ratio"],
            marker_color=[GRADE_COLORS.get(g, "#999") for g in grade_ecl["grade"]],
            text=[f"{v:.2%}" for v in grade_ecl["alll_ratio"]],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"ALLL Ratio by Grade ({scenario_key.title()})",
            template="plotly_white", height=400,
            yaxis_title="ALLL Ratio", yaxis_tickformat=".1%",
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### ECL Comparison Across FEG Views")

feg_labels = {
    "baseline": "Pre-FEG (Pure Model)",
    "central": "Central (Baseline)",
    "stress": "Post-FEG (Weighted)",
}
ecl_compare = []
for scen in ["baseline", "central", "stress"]:
    s_data = ecl_scen[ecl_scen["scenario"] == scen]
    ecl_compare.append({
        "Scenario": feg_labels.get(scen, scen.title()),
        "Total ECL": s_data["total_ecl"].sum(),
        "Total EAD": s_data["total_ead"].sum(),
        "ALLL Ratio": s_data["total_ecl"].sum() / max(s_data["total_ead"].sum(), 1),
    })
ecl_compare_df = pd.DataFrame(ecl_compare)

col7, col8 = st.columns(2)
with col7:
    fig = go.Figure(go.Bar(
        x=ecl_compare_df["Scenario"], y=ecl_compare_df["Total ECL"],
        marker_color=["#3498db", "#f39c12", "#e74c3c"],
        text=[format_currency(v) for v in ecl_compare_df["Total ECL"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Total ECL by Scenario", template="plotly_white",
        height=400, yaxis_title="ECL ($)",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

with col8:
    fig = go.Figure(go.Bar(
        x=ecl_compare_df["Scenario"], y=ecl_compare_df["ALLL Ratio"],
        marker_color=["#3498db", "#f39c12", "#e74c3c"],
        text=[f"{v:.1%}" for v in ecl_compare_df["ALLL Ratio"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="ALLL Ratio by Scenario", template="plotly_white",
        height=400, yaxis_title="ALLL Ratio", yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

# ═══════════════════════════════════════════════════════════════════════════════
# ASSUMPTIONS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
with st.expander("Current Assumptions & Methodology"):
    st.markdown(f"""
**Global Parameters:**

| Parameter | Value |
|-----------|-------|
| Mode | {mode_label} |
| Scenario | {scenario_key.title()} |
| Recovery Rate | {recovery_rate:.0%} |
| Discount Rate | {discount_rate:.0%} |
| Horizon | {horizon_years} years ({n_months} months) |
| Snapshot | {snapshot_date.strftime('%Y-%m')} |
| Starting Balance | {format_currency(starting_balance)} |

**Per-Grade Originations & Liquidation:**
""")

    grade_table = "| Grade | Starting Balance | Monthly Origination | Liquidation | FTR |\n"
    grade_table += "|-------|-----------------|--------------------:|------------:|-----:|\n"
    for g in GRADE_ORDER:
        gs = grade_summaries[g]
        grade_table += f"| {g} | {format_currency(gs['starting_balance'])} | {format_currency(gs['origination'])} | {gs['liquidation']:.1%} | {gs['ftr']:.4%} |\n"

    st.markdown(grade_table)

    st.markdown(f"""
**Methodology:**
- **Per-Grade Projection:** Each grade runs independently with its own flow rates, originations, and liquidation
- **Flow Rates:** From `flow_rates_by_scenario.csv` — grade × scenario specific
- **CECL 3-Phase:** Phase 1 = scenario-stressed (24mo), Phase 2 = linear reversion (12mo), Phase 3 = baseline
- **Output Format:** Matches input `receivables_tracker.csv` exactly (13 columns, per-grade rows)
- **GCO:** 180+ DPD balance × GCO flow rate
- **NCO:** GCO × (1 - Recovery Rate)
- **FTR:** Product of all sequential flow rates (Current → GCO)
""")
