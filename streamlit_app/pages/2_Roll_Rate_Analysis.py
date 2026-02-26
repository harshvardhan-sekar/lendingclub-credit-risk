"""Page 2: Roll-Rate Analysis — Receivables tracker, flow rates, Sankey diagram.

V6 Roadmap Requirements:
- st.info() synthetic panel disclaimer (exact wording from V6)
- Filters: Grade, Vintage Year, Period range
- Receivables tracker table (institutional format, downloadable Excel)
- Flow rate transition matrix (heatmap)
- Sankey diagram showing delinquency flow
- Flow rate trend lines over time
- Delinquency bucket balances over time (stacked bar)
- Flow-Through Rate by grade
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
from utils.data_loader import load_receivables_tracker, load_roll_rates, load_flow_through_rate
from utils.styles import (
    inject_custom_css, sidebar_disclaimer, kpi_card, format_currency,
    DPD_COLORS, GRADE_COLORS, GRADE_ORDER,
)

st.set_page_config(page_title="Roll-Rate Analysis", layout="wide")
inject_custom_css()
sidebar_disclaimer()

st.title("Roll-Rate Analysis")
st.markdown("Delinquency bucket transitions and receivables tracking across the portfolio lifecycle.")

# V6 Disclaimer
st.info(
    "**Flow Rate Data Note:** Flow rates are computed from synthetically reconstructed "
    "monthly DPD status. These represent forward (worsening) transitions only. Curing rates "
    "are not observable from the available data. In a production environment with monthly "
    "payment tapes, the same framework would incorporate two-way transitions and curing rates."
)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

recv = load_receivables_tracker()
roll = load_roll_rates()
ftr = load_flow_through_rate()

# Normalize column names (handle all naming conventions from different code versions)
_col_renames = {}
if "flow_rate" in roll.columns and "roll_rate" not in roll.columns:
    _col_renames["flow_rate"] = "roll_rate"
if "from_balance" in roll.columns and "balance_amount" not in roll.columns:
    _col_renames["from_balance"] = "balance_amount"
if "rate" in roll.columns and "roll_rate" not in roll.columns and "flow_rate" not in roll.columns:
    _col_renames["rate"] = "roll_rate"
if "balance" in roll.columns and "balance_amount" not in roll.columns and "from_balance" not in roll.columns:
    _col_renames["balance"] = "balance_amount"
if _col_renames:
    roll = roll.rename(columns=_col_renames)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: Filters
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("### Filters")

# Grade filter
selected_grades = st.sidebar.multiselect("Grade", GRADE_ORDER, default=GRADE_ORDER)

# Period range filter
all_months = sorted(recv["month_date"].unique())
if len(all_months) >= 2:
    # Default to full range — late months may only have "Current" loans as portfolio winds down
    period_start = st.sidebar.select_slider(
        "Period Start",
        options=all_months,
        value=all_months[0],
        format_func=lambda x: x.strftime("%Y-%m") if hasattr(x, "strftime") else str(x),
    )
    period_end = st.sidebar.select_slider(
        "Period End",
        options=all_months,
        value=all_months[-1],
        format_func=lambda x: x.strftime("%Y-%m") if hasattr(x, "strftime") else str(x),
    )
else:
    period_start, period_end = all_months[0], all_months[-1]

# Apply filters
recv_f = recv[
    (recv["grade"].isin(selected_grades)) &
    (recv["month_date"] >= period_start) &
    (recv["month_date"] <= period_end)
]
roll_f = roll[
    (roll["grade"].isin(selected_grades)) &
    (roll["month_date"] >= period_start) &
    (roll["month_date"] <= period_end)
]
ftr_f = ftr[ftr["grade"].isin(selected_grades)]

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Portfolio Flow Metrics")
avg_ftr_by_grade = ftr_f.groupby("grade")["ftr"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi_card("Avg FTR (A)", f"{avg_ftr_by_grade.get('A', 0):.4%}")
with c2:
    kpi_card("Avg FTR (C)", f"{avg_ftr_by_grade.get('C', 0):.4%}")
with c3:
    kpi_card("Avg FTR (E)", f"{avg_ftr_by_grade.get('E', 0):.4%}")
with c4:
    kpi_card("Avg FTR (G)", f"{avg_ftr_by_grade.get('G', 0):.4%}")
with c5:
    total_bal = recv_f.groupby("month_date")["balance"].sum()
    latest_bal = total_bal.iloc[-1] if len(total_bal) > 0 else 0
    kpi_card("Latest Total Balance", format_currency(latest_bal))

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# RECEIVABLES TRACKER TABLE (Institutional Format)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Receivables Tracker (Institutional Format)")
st.markdown("Monthly balances by DPD bucket with flow rates and NCO.")

bucket_order = ["Current", "30+", "60+", "90+", "120+", "150+", "180+", "GCO"]

# Aggregate across selected grades
recv_agg = recv_f.groupby(["month_date", "dpd_bucket"]).agg(
    balance=("balance", "sum"),
    count=("count", "sum"),
    gco_amount=("gco_amount", "sum"),
    recovery_amount=("recovery_amount", "sum"),
    nco_amount=("nco_amount", "sum"),
).reset_index()

# Pivot for display
pivot_bal = recv_agg.pivot_table(index="dpd_bucket", columns="month_date", values="balance", aggfunc="sum")
pivot_bal = pivot_bal.reindex([b for b in bucket_order if b in pivot_bal.index])
pivot_bal.columns = [c.strftime("%Y-%m") if hasattr(c, "strftime") else str(c) for c in pivot_bal.columns]

# Show 24 columns centered on peak activity for readability
if pivot_bal.shape[1] > 24:
    # Find the window with the most non-zero values across delinquent buckets
    non_current = pivot_bal.loc[pivot_bal.index != "Current"]
    col_activity = non_current.fillna(0).sum(axis=0)
    if col_activity.max() > 0:
        peak_idx = col_activity.values.argmax()
        start = max(0, peak_idx - 12)
        end = min(pivot_bal.shape[1], start + 24)
        start = max(0, end - 24)  # adjust if near the end
        pivot_bal = pivot_bal.iloc[:, start:end]
    else:
        pivot_bal = pivot_bal.iloc[:, :24]

# ── Section 1: Dollar Receivables ──
st.markdown("**Dollar Receivables**")
pivot_display = pivot_bal.map(lambda x: format_currency(x) if pd.notna(x) else "—")
st.dataframe(pivot_display, use_container_width=True)

# ── Section 2: Flow Rates ──
# Compute flow rates from dollar receivables (month-over-month bucket transitions)
st.markdown("**Flow Rates** (forward transitions from prior month)")

forward_pairs = [
    ("Current", "30+", "30+ FR"),
    ("30+", "60+", "60+ FR"),
    ("60+", "90+", "90+ FR"),
    ("90+", "120+", "120+ FR"),
    ("120+", "150+", "150+ FR"),
    ("150+", "180+", "180+ FR"),
]

# Use the same columns (months) as pivot_bal
months_list = list(pivot_bal.columns)
fr_rows = {}

for from_b, to_b, label in forward_pairs:
    row_vals = {}
    for i in range(1, len(months_list)):
        prev_month = months_list[i - 1]
        curr_month = months_list[i]
        prev_val = pivot_bal.loc[from_b, prev_month] if from_b in pivot_bal.index else 0
        curr_val = pivot_bal.loc[to_b, curr_month] if to_b in pivot_bal.index else 0
        if pd.notna(prev_val) and prev_val > 0 and pd.notna(curr_val):
            rate = curr_val / prev_val
            row_vals[curr_month] = min(rate, 1.0)  # Cap at 100% — values >100% are artifacts
        else:
            row_vals[curr_month] = np.nan
    fr_rows[label] = row_vals

# GCO flow rate: use month-over-month CHANGE in GCO balance / prior month 180+ balance
# The GCO row in receivables tracker is cumulative (total ever-charged-off),
# so we need the monthly increment, not the absolute amount.
gco_row_vals = {}
if "GCO" in pivot_bal.index:
    gco_bals = pivot_bal.loc["GCO"]
    for i in range(1, len(months_list)):
        prev_month = months_list[i - 1]
        curr_month = months_list[i]
        prev_180 = pivot_bal.loc["180+", prev_month] if "180+" in pivot_bal.index and prev_month in pivot_bal.columns else 0
        # FIX (Issue #11): pd.Series.get() silently returns NaN when the key exists
        # but its value is NaN (the default 0 only fires when the key is absent).
        # Use direct indexing + explicit pd.isna() guard instead.
        prev_gco = float(gco_bals[prev_month]) if prev_month in gco_bals.index else 0.0
        curr_gco = float(gco_bals[curr_month]) if curr_month in gco_bals.index else 0.0
        prev_gco = 0.0 if pd.isna(prev_gco) else prev_gco
        curr_gco = 0.0 if pd.isna(curr_gco) else curr_gco
        gco_increment = curr_gco - prev_gco if pd.notna(curr_gco) and pd.notna(prev_gco) else 0
        # If GCO didn't increase, use a proxy from roll rates if available
        if gco_increment <= 0:
            gco_increment = 0
        if pd.notna(prev_180) and prev_180 > 0 and gco_increment > 0:
            gco_fr = min(gco_increment / prev_180, 1.0)  # Cap at 100%
            gco_row_vals[curr_month] = gco_fr
        else:
            gco_row_vals[curr_month] = np.nan
else:
    for i in range(1, len(months_list)):
        gco_row_vals[months_list[i]] = np.nan
fr_rows["GCO FR"] = gco_row_vals

fr_df = pd.DataFrame(fr_rows).T
# Only keep columns starting from the 2nd month (can't compute for first month)
fr_display = fr_df.map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
st.dataframe(fr_display, use_container_width=True)

# ── Section 3: Flow-Through Rates ──
st.markdown("**Flow-Through Rates** (cumulative product: Current → each bucket)")

ftr_labels = ["FTR 30", "FTR 60", "FTR 90", "FTR 120", "FTR 150", "FTR 180", "FTR GCO"]
fr_keys = ["30+ FR", "60+ FR", "90+ FR", "120+ FR", "150+ FR", "180+ FR", "GCO FR"]

ftr_rows = {}
for j, (ftr_label, _) in enumerate(zip(ftr_labels, fr_keys)):
    row_vals = {}
    for month in fr_df.columns:
        cumulative = 1.0
        valid = True
        for k in range(j + 1):
            val = fr_df.loc[fr_keys[k], month] if fr_keys[k] in fr_df.index and month in fr_df.columns else np.nan
            if pd.notna(val):
                cumulative *= val
            else:
                valid = False
                break
        row_vals[month] = cumulative if valid else np.nan
    ftr_rows[ftr_label] = row_vals

ftr_df = pd.DataFrame(ftr_rows).T
ftr_display = ftr_df.map(lambda x: f"{x:.4%}" if pd.notna(x) else "—")
st.dataframe(ftr_display, use_container_width=True)

# ── Excel Download (all 3 sections) ──
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    pivot_bal.to_excel(writer, sheet_name="Dollar Receivables")
    fr_df.to_excel(writer, sheet_name="Flow Rates")
    ftr_df.to_excel(writer, sheet_name="Flow-Through Rates")
    # NCO sheet
    nco_pivot = recv_agg.pivot_table(index="dpd_bucket", columns="month_date", values="nco_amount", aggfunc="sum")
    nco_pivot.columns = [c.strftime("%Y-%m") if hasattr(c, "strftime") else str(c) for c in nco_pivot.columns]
    if nco_pivot.shape[1] > 24:
        nco_pivot = nco_pivot.iloc[:, -24:]
    nco_pivot.to_excel(writer, sheet_name="NCO")

st.download_button(
    label="Download Receivables Tracker (Excel)",
    data=buf.getvalue(),
    file_name="receivables_tracker.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SANKEY DIAGRAM — Delinquency Flow
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Delinquency Flow (Sankey Diagram)")
st.markdown("Average balance flow through delinquency stages across selected period.")

# Build Sankey from roll rates (average across period)
# KEY FIX: Use flow amount (balance × rate) not raw from_balance.
# Raw from_balance makes Current dwarf everything since it's 99%+ of the portfolio.
_bal_col = next((c for c in ["balance_amount", "from_balance", "balance"] if c in roll_f.columns), None)
_rate_col = next((c for c in ["roll_rate", "flow_rate", "rate"] if c in roll_f.columns), None)

if _bal_col and _rate_col:
    roll_avg = roll_f.groupby(["from_bucket", "to_bucket"]).agg(
        avg_balance=(_bal_col, "mean"),
        avg_rate=(_rate_col, "mean"),
    ).reset_index()
else:
    roll_avg = pd.DataFrame(columns=["from_bucket", "to_bucket", "avg_balance", "avg_rate"])

# Filter to forward transitions only (exclude stay-in-same)
roll_fwd = roll_avg[roll_avg["from_bucket"] != roll_avg["to_bucket"]].copy()
# Compute actual dollar flow = balance × flow_rate
roll_fwd["flow_amount"] = roll_fwd["avg_balance"] * roll_fwd["avg_rate"]
# Filter to meaningful flows
roll_fwd = roll_fwd[roll_fwd["flow_amount"] > 0]

if len(roll_fwd) > 0:
    # Build node labels
    all_nodes = list(set(roll_fwd["from_bucket"].tolist() + roll_fwd["to_bucket"].tolist()))
    # Order nodes
    node_order = [b for b in bucket_order if b in all_nodes]
    node_idx = {n: i for i, n in enumerate(node_order)}

    # Node colors
    node_colors = [DPD_COLORS.get(n, "#999") for n in node_order]

    # Links — use flow_amount (actual dollars transitioning)
    valid = []
    for _, row in roll_fwd.iterrows():
        fb, tb = row["from_bucket"], row["to_bucket"]
        if fb in node_idx and tb in node_idx:
            valid.append((node_idx[fb], node_idx[tb], row["flow_amount"]))
    if valid:
        s_list, t_list, v_list = zip(*valid)

        # Color links based on severity
        link_colors = []
        for s, t, v in valid:
            severity = t / max(len(node_order) - 1, 1)
            r = int(52 + (231 - 52) * severity)
            g = int(152 - (152 - 76) * severity)
            b = int(219 - (219 - 60) * severity)
            link_colors.append(f"rgba({r},{g},{b},0.5)")

        fig = go.Figure(go.Sankey(
            node=dict(
                pad=20, thickness=25,
                label=node_order,
                color=node_colors,
            ),
            link=dict(
                source=list(s_list),
                target=list(t_list),
                value=list(v_list),
                color=link_colors,
            ),
        ))
        fig.update_layout(
            title="Average Dollar Flow Through Delinquency Stages",
            template="plotly_white", height=500,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
else:
    st.info("Insufficient data for Sankey diagram with current filters.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FLOW RATE TRANSITION MATRIX (Heatmap)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Average Flow Rate Transition Matrix")

roll_matrix = roll_f.groupby(["from_bucket", "to_bucket"])["roll_rate"].mean().reset_index()
from_buckets = [b for b in bucket_order if b in roll_matrix["from_bucket"].unique()]
to_buckets = [b for b in bucket_order if b in roll_matrix["to_bucket"].unique()]

if len(from_buckets) > 0 and len(to_buckets) > 0:
    matrix = roll_matrix.pivot_table(index="from_bucket", columns="to_bucket", values="roll_rate")
    matrix = matrix.reindex(index=from_buckets, columns=to_buckets)

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=list(matrix.columns),
        y=list(matrix.index),
        texttemplate="%{z:.1%}",
        colorscale="RdYlGn_r",
        showscale=True,
    ))
    fig.update_layout(
        title="Average Roll Rate: From → To Bucket",
        template="plotly_white", height=450,
        xaxis_title="To Bucket", yaxis_title="From Bucket",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FLOW RATE TRENDS OVER TIME + FTR BY GRADE
# ══════════════════════════════════════════════════════════════════════════════

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Flow Rate Trends Over Time")
    st.markdown("Key forward flow rates tracked month-over-month (early warning indicators).")

    # Compute forward flow rates from roll data
    forward_transitions = [
        ("Current", "30+"), ("30+", "60+"), ("60+", "90+"),
        ("90+", "120+"), ("120+", "150+"), ("150+", "180+"), ("180+", "GCO"),
    ]
    fwd_rates = roll_f[roll_f.apply(
        lambda r: (r["from_bucket"], r["to_bucket"]) in forward_transitions, axis=1
    )].copy()
    fwd_rates["transition"] = fwd_rates["from_bucket"] + " → " + fwd_rates["to_bucket"]
    fwd_rates["month_date"] = pd.to_datetime(fwd_rates["month_date"])

    # Average across grades for trend
    fwd_trend = fwd_rates.groupby(["month_date", "transition"])["roll_rate"].mean().reset_index()
    fwd_trend = fwd_trend.sort_values("month_date")

    if len(fwd_trend) > 0:
        fig = px.line(
            fwd_trend, x="month_date", y="roll_rate", color="transition",
            title="Forward Flow Rates Over Time",
        )
        fig.update_layout(
            template="plotly_white", height=420,
            yaxis_title="Flow Rate", yaxis_tickformat=".1%",
            xaxis_title="", legend_title="Transition",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

with col2:
    st.markdown("### Flow-Through Rate by Grade")
    st.markdown("FTR = product of all flow rates (Current → GCO). Lower is better.")

    ftr_monthly = ftr_f.groupby(["month_date", "grade"])["ftr"].mean().reset_index()
    ftr_monthly["month_date"] = pd.to_datetime(ftr_monthly["month_date"])
    ftr_monthly = ftr_monthly.dropna(subset=["ftr"]).sort_values("month_date")

    if len(ftr_monthly) > 0:
        fig = px.line(
            ftr_monthly, x="month_date", y="ftr", color="grade",
            color_discrete_map=GRADE_COLORS,
            title="Monthly Flow-Through Rate by Grade",
        )
        fig.update_layout(
            template="plotly_white", height=420,
            yaxis_title="Flow-Through Rate", yaxis_tickformat=".3%",
            xaxis_title="", legend_title="Grade",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# DELINQUENCY BUCKET BALANCES OVER TIME (Stacked Bar)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Delinquency Bucket Balances Over Time")

dpd_ts = recv_f.groupby(["month_date", "dpd_bucket"])["balance"].sum().reset_index()
dpd_ts["dpd_bucket"] = pd.Categorical(dpd_ts["dpd_bucket"], categories=bucket_order, ordered=True)
dpd_ts = dpd_ts.sort_values(["month_date", "dpd_bucket"])

# Subsample months for readability if too many
months_available = sorted(dpd_ts["month_date"].unique())
if len(months_available) > 48:
    # Show every 3rd month
    months_show = months_available[::3]
    dpd_ts = dpd_ts[dpd_ts["month_date"].isin(months_show)]

if len(dpd_ts) > 0:
    fig = px.bar(
        dpd_ts, x="month_date", y="balance", color="dpd_bucket",
        color_discrete_map=DPD_COLORS,
        category_orders={"dpd_bucket": bucket_order},
        title="Stacked Balance by DPD Bucket",
    )
    fig.update_layout(
        template="plotly_white", height=450,
        barmode="stack",
        xaxis_title="", yaxis_title="Balance ($)",
        legend_title="DPD Bucket",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
