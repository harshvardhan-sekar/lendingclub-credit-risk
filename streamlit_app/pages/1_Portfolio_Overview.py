"""Page 1: Portfolio Overview — Executive-level view of portfolio health.

V6 Roadmap Requirements:
- KPI Cards: Total Outstanding, Default Rate, ALLL Ratio, Flow-Through Rate, NCO Ratio, 30+ DPD Rate, Weighted Avg Interest Rate
- Sidebar filters: Date range, Grade, Term, Purpose, State
- Charts:
  1. Default rate by grade (bar)
  2. Portfolio composition by grade (donut)
  3. Monthly origination volume (dual-axis line)
  4. Flow-Through Rate trend (line with alert threshold)
  5. Delinquency trend (stacked area: Current, 30+, 60+, 90+)
  6. Geographic default rate (choropleth by state)
  7. Default rate by purpose (horizontal bar)
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.data_loader import (
    load_strategy_analysis, load_ecl_postfeg, load_ecl_central,
    load_sensitivity_results, load_loans_sample,
    load_flow_through_rate, load_receivables_tracker,
)
from utils.styles import (
    inject_custom_css, sidebar_disclaimer, kpi_card, format_currency,
    GRADE_COLORS, GRADE_ORDER, DPD_COLORS,
)
from utils.charts import bar_by_grade, donut_chart

st.set_page_config(page_title="Portfolio Overview", layout="wide")
inject_custom_css()
sidebar_disclaimer()

st.title("Portfolio Overview")
st.markdown("Executive summary of the LendingClub consumer loan portfolio — the first view a CRO would see.")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

strategy = load_strategy_analysis()
sens = load_sensitivity_results()
ps = sens.get("portfolio_summary", {})
loans = load_loans_sample()
ftr_data = load_flow_through_rate()
recv = load_receivables_tracker()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: Filters
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("### Filters")

# Date range filter
if "issue_d" in loans.columns and loans["issue_d"].notna().any():
    min_date = loans["issue_d"].min().to_pydatetime()
    max_date = loans["issue_d"].max().to_pydatetime()
    date_range = st.sidebar.date_input(
        "Origination Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter loans by origination date.",
    )
else:
    date_range = None

# Grade filter
selected_grades = st.sidebar.multiselect(
    "Grade", GRADE_ORDER, default=GRADE_ORDER,
    help="Select credit grades.",
)

# Term filter
all_terms = sorted(loans["term"].dropna().unique()) if "term" in loans.columns else [36, 60]
selected_terms = st.sidebar.multiselect(
    "Term (months)", all_terms, default=all_terms,
    help="Loan term in months.",
)

# Purpose filter
all_purposes = sorted(loans["purpose"].dropna().unique()) if "purpose" in loans.columns else []
selected_purposes = st.sidebar.multiselect(
    "Purpose", all_purposes, default=all_purposes,
    help="Loan purpose category.",
)

# State filter
all_states = sorted(loans["addr_state"].dropna().unique()) if "addr_state" in loans.columns else []
selected_states = st.sidebar.multiselect(
    "State", all_states, default=[],
    help="Filter by borrower state (leave empty for all).",
)

# ══════════════════════════════════════════════════════════════════════════════
# APPLY FILTERS TO LOAN DATA
# ══════════════════════════════════════════════════════════════════════════════

filtered = loans.copy()

if date_range and len(date_range) == 2:
    filtered = filtered[
        (filtered["issue_d"] >= pd.Timestamp(date_range[0])) &
        (filtered["issue_d"] <= pd.Timestamp(date_range[1]))
    ]

if selected_grades:
    filtered = filtered[filtered["grade"].isin(selected_grades)]

if selected_terms and "term" in filtered.columns:
    filtered = filtered[filtered["term"].isin(selected_terms)]

if selected_purposes and "purpose" in filtered.columns:
    filtered = filtered[filtered["purpose"].isin(selected_purposes)]

if selected_states and "addr_state" in filtered.columns:
    filtered = filtered[filtered["addr_state"].isin(selected_states)]

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTED METRICS
# ══════════════════════════════════════════════════════════════════════════════

total_balance = filtered["funded_amnt"].sum()
total_loans_n = len(filtered)
default_col = "default" if "default" in filtered.columns else "default_flag"
overall_default_rate = filtered[default_col].mean() if default_col in filtered.columns else 0
# Balance-weighted average interest rate (not simple mean)
if "int_rate" in filtered.columns and "funded_amnt" in filtered.columns:
    _w = filtered["funded_amnt"]
    avg_int_rate = (filtered["int_rate"] * _w).sum() / _w.sum() if _w.sum() > 0 else 0
elif "int_rate" in filtered.columns:
    avg_int_rate = filtered["int_rate"].mean()
else:
    avg_int_rate = 0

# NCO from receivables tracker
# GCO bucket is cumulative — compute monthly NCO as month-over-month GCO balance increment
recv_active = recv[recv["dpd_bucket"] != "GCO"]
gco_monthly = recv[recv["dpd_bucket"] == "GCO"].groupby("month_date")["balance"].sum().sort_index()
gco_increments = gco_monthly.diff().clip(lower=0)  # Monthly new charge-offs

recv_total = recv_active.groupby("month_date").agg(
    balance=("balance", "sum"),
).reset_index().sort_values("month_date")
recv_total = recv_total.set_index("month_date")
recv_total["monthly_nco"] = gco_increments
recv_total = recv_total.fillna(0).reset_index()

# Annualized NCO ratio — use months with meaningful performing balance
recv_meaningful = recv_total[recv_total["balance"] > recv_total["balance"].max() * 0.05]
if len(recv_meaningful) >= 12:
    last_12 = recv_meaningful.tail(12)
    annual_nco = last_12["monthly_nco"].sum()
    avg_balance = last_12["balance"].mean()
    nco_ratio = annual_nco / max(avg_balance, 1)
elif len(recv_meaningful) > 0:
    annual_nco = recv_meaningful["monthly_nco"].sum() * (12 / len(recv_meaningful))
    avg_balance = recv_meaningful["balance"].mean()
    nco_ratio = annual_nco / max(avg_balance, 1)
else:
    nco_ratio = 0

# FTR — use last meaningful value (exclude months where portfolio has wound down to FTR≈1.0)
ftr_agg = ftr_data.groupby("month_date")["ftr"].mean().dropna()
ftr_meaningful = ftr_agg[ftr_agg < 0.99]
latest_ftr = ftr_meaningful.iloc[-1] if len(ftr_meaningful) > 0 else (ftr_agg.iloc[-1] if len(ftr_agg) > 0 else 0)

# 30+ DPD rate (from receivables — use meaningful period, exclude GCO terminal bucket)
# GCO is not delinquency; it's charged-off. 30+ DPD = sum(30+ thru 180+) / total active balance
recv_active_latest = recv_active.copy()
recv_active_latest = recv_active_latest.sort_values("month_date")
# Use month with highest total activity (not tail-end runoff where only Current remains)
month_bals = recv_active_latest.groupby("month_date")["balance"].sum()
meaningful_months = month_bals[month_bals > month_bals.max() * 0.05]
if len(meaningful_months) > 0:
    latest_active_month = meaningful_months.index[-1]
else:
    latest_active_month = recv["month_date"].max()
latest_recv = recv_active[recv_active["month_date"] == latest_active_month]
total_bal_latest = latest_recv["balance"].sum()
dpd30_plus_bal = latest_recv[latest_recv["dpd_bucket"].isin(
    ["30+", "60+", "90+", "120+", "150+", "180+"]
)]["balance"].sum()
dpd30_rate = dpd30_plus_bal / max(total_bal_latest, 1)

# ALLL ratio from portfolio summary
alll_ratio = ps.get("postfeg_alll", 0)

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Total Funded", format_currency(total_balance))
with c2:
    kpi_card("Overall Default Rate", f"{overall_default_rate:.2%}")
with c3:
    kpi_card("Wtd Avg Interest Rate", f"{avg_int_rate:.2f}%")
with c4:
    kpi_card("ALLL Ratio", f"{alll_ratio:.2%}")

c5, c6, c7, c8 = st.columns(4)
with c5:
    kpi_card("Flow-Through Rate", f"{latest_ftr:.3%}",
             delta="Current → GCO product")
with c6:
    kpi_card("NCO Ratio (Ann.)", f"{nco_ratio:.2%}")
with c7:
    kpi_card("30+ DPD Rate", f"{dpd30_rate:.2%}",
             delta="Early delinquency indicator")
with c8:
    kpi_card("Total Loans", f"{total_loans_n:,.0f}")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: Default Rate by Grade + Portfolio Composition (Donut)
# ══════════════════════════════════════════════════════════════════════════════

col1, col2 = st.columns(2)

with col1:
    # Default rate by grade from filtered loan data
    grade_stats = filtered.groupby("grade").agg(
        n_loans=(default_col, "count"),
        defaults=(default_col, "sum"),
        total_funded=("funded_amnt", "sum"),
    ).reindex(GRADE_ORDER).dropna()
    grade_stats["default_rate"] = grade_stats["defaults"] / grade_stats["n_loans"].replace(0, np.nan)

    fig = px.bar(
        grade_stats.reset_index(), x="grade", y="default_rate",
        color="grade", color_discrete_map=GRADE_COLORS,
        title="Default Rate by Grade",
    )
    fig.update_layout(
        template="plotly_white", height=400,
        showlegend=False, yaxis_title="Default Rate",
        yaxis_tickformat=".1%", xaxis_title="Grade",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

with col2:
    grade_balance = filtered.groupby("grade")["funded_amnt"].sum().reindex(GRADE_ORDER).dropna()
    colors = [GRADE_COLORS.get(g, "#999") for g in grade_balance.index]
    fig = donut_chart(
        labels=grade_balance.index.tolist(),
        values=grade_balance.values.tolist(),
        title="Portfolio Composition by Grade (Balance)",
        colors=colors,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: Origination Volume + FTR Trend
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.markdown("### Monthly Origination Volume")
    if "issue_d" in filtered.columns and "vintage_year" in filtered.columns:
        monthly = filtered.set_index("issue_d").resample("M").agg(
            volume=("funded_amnt", "sum"),
            count=("funded_amnt", "count"),
            default_rate=(default_col, "mean"),
        ).reset_index()
        monthly = monthly[monthly["volume"] > 0]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["issue_d"], y=monthly["volume"],
            name="Funded Amount",
            marker_color="#3498db", opacity=0.7,
        ))
        fig.add_trace(go.Scatter(
            x=monthly["issue_d"], y=monthly["default_rate"],
            name="Default Rate",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#e74c3c", width=2),
        ))
        fig.update_layout(
            template="plotly_white", height=420,
            yaxis=dict(title="Funded Amount ($)", side="left"),
            yaxis2=dict(title="Default Rate", overlaying="y", side="right", tickformat=".1%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
    else:
        st.info("Origination date not available.")

with col4:
    st.markdown("### Flow-Through Rate Trend")
    st.markdown("Current → GCO product rate. Amber alert threshold at 0.5%.")

    ftr_agg_df = ftr_data.groupby("month_date")["ftr"].mean().reset_index()
    ftr_agg_df["month_date"] = pd.to_datetime(ftr_agg_df["month_date"])
    ftr_agg_df = ftr_agg_df.dropna(subset=["ftr"]).sort_values("month_date")

    if len(ftr_agg_df) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ftr_agg_df["month_date"], y=ftr_agg_df["ftr"],
            mode="lines",
            line=dict(color="#2c3e50", width=2),
            name="FTR",
            fill="tozeroy",
            fillcolor="rgba(52, 152, 219, 0.15)",
        ))
        # Amber threshold
        fig.add_hline(
            y=0.005, line_dash="dash", line_color="#f39c12",
            annotation_text="Amber Threshold (0.5%)",
            annotation_position="top right",
        )
        fig.update_layout(
            template="plotly_white", height=420,
            yaxis_title="Flow-Through Rate",
            yaxis_tickformat=".2%",
            xaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
    else:
        st.info("Flow-through rate data not available.")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3: Delinquency Stacked Area + Geographic Choropleth
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
col5, col6 = st.columns(2)

with col5:
    st.markdown("### Delinquency Trend")
    st.markdown("Stacked area showing balance by DPD bucket over time.")

    # Build delinquency time series from receivables tracker
    dpd_order = ["Current", "30+", "60+", "90+", "120+", "150+", "180+", "GCO"]
    dpd_ts = recv.groupby(["month_date", "dpd_bucket"])["balance"].sum().reset_index()
    dpd_ts["month_date"] = pd.to_datetime(dpd_ts["month_date"])
    # Filter to relevant grades if selected
    if selected_grades and len(selected_grades) < 7:
        recv_filt = recv[recv["grade"].isin(selected_grades)]
        dpd_ts = recv_filt.groupby(["month_date", "dpd_bucket"])["balance"].sum().reset_index()
        dpd_ts["month_date"] = pd.to_datetime(dpd_ts["month_date"])

    dpd_ts["dpd_bucket"] = pd.Categorical(dpd_ts["dpd_bucket"], categories=dpd_order, ordered=True)
    dpd_ts = dpd_ts.sort_values(["month_date", "dpd_bucket"])

    if len(dpd_ts) > 0:
        fig = px.area(
            dpd_ts, x="month_date", y="balance", color="dpd_bucket",
            color_discrete_map=DPD_COLORS,
            category_orders={"dpd_bucket": dpd_order},
            title="Balance by Delinquency Bucket Over Time",
        )
        fig.update_layout(
            template="plotly_white", height=420,
            xaxis_title="", yaxis_title="Balance ($)",
            legend_title="DPD Bucket",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

with col6:
    st.markdown("### Geographic Default Rate")
    if "addr_state" in filtered.columns:
        state_stats = filtered.groupby("addr_state").agg(
            n_loans=(default_col, "count"),
            defaults=(default_col, "sum"),
        ).reset_index()
        state_stats["default_rate"] = state_stats["defaults"] / state_stats["n_loans"].replace(0, np.nan)
        state_stats = state_stats[state_stats["n_loans"] >= 50]  # Min sample threshold

        fig = px.choropleth(
            state_stats,
            locations="addr_state",
            locationmode="USA-states",
            color="default_rate",
            scope="usa",
            color_continuous_scale="RdYlGn_r",
            title="Default Rate by State",
            labels={"default_rate": "Default Rate", "addr_state": "State"},
        )
        fig.update_layout(
            template="plotly_white", height=420,
            coloraxis_colorbar=dict(tickformat=".1%"),
            geo=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
    else:
        st.info("State data not available for geographic visualization.")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4: Default Rate by Purpose (Full Width)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### Default Rate by Purpose")

if "purpose" in filtered.columns:
    purpose_stats = filtered.groupby("purpose").agg(
        n_loans=(default_col, "count"),
        defaults=(default_col, "sum"),
        total_funded=("funded_amnt", "sum"),
    ).reset_index()
    purpose_stats["default_rate"] = purpose_stats["defaults"] / purpose_stats["n_loans"].replace(0, np.nan)
    purpose_stats = purpose_stats.sort_values("default_rate", ascending=True)

    fig = px.bar(
        purpose_stats, x="default_rate", y="purpose",
        orientation="h",
        title="Default Rate by Loan Purpose",
        color="default_rate",
        color_continuous_scale="RdYlGn_r",
        labels={"default_rate": "Default Rate", "purpose": "Loan Purpose"},
    )
    fig.update_layout(
        template="plotly_white", height=450,
        xaxis_tickformat=".1%",
        coloraxis_colorbar=dict(tickformat=".1%"),
        yaxis=dict(categoryorder="total ascending"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

# ══════════════════════════════════════════════════════════════════════════════
# THREE FEG VIEWS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### ECL Summary — Three FEG Views")

fc1, fc2, fc3 = st.columns(3)
with fc1:
    kpi_card("Central ECL", format_currency(ps.get("central_ecl", 0)),
             f"ALLL: {ps.get('central_alll', 0):.1%}")
with fc2:
    kpi_card("Pre-FEG ECL", format_currency(ps.get("prefeg_ecl", 0)),
             f"ALLL: {ps.get('prefeg_alll', 0):.1%}")
with fc3:
    kpi_card("Post-FEG ECL", format_currency(ps.get("postfeg_ecl", 0)),
             f"ALLL: {ps.get('postfeg_alll', 0):.1%}")

# Grade-level strategy summary table
st.markdown("---")
st.markdown("### Grade-Level Summary")

strat_display = strategy.copy()
strat_display["default_rate"] = strat_display["default_rate"].apply(lambda x: f"{x:.2%}")
strat_display["alll_ratio"] = strat_display["alll_ratio"].apply(lambda x: f"{x:.2%}")
strat_display["avg_int_rate"] = strat_display["avg_int_rate"].apply(lambda x: f"{x:.2f}%")
strat_display["total_balance"] = strat_display["total_balance"].apply(lambda x: format_currency(x))
strat_display["net_margin_pct"] = strat_display["net_margin_pct"].apply(lambda x: f"{x:.2f}%")
strat_display = strat_display[["grade", "n_loans", "total_balance", "avg_int_rate",
                                "default_rate", "alll_ratio", "net_margin_pct"]]
strat_display.columns = ["Grade", "Loans", "Total Balance", "Avg Rate",
                          "Default Rate", "ALLL Ratio", "Net Margin"]
st.dataframe(strat_display, use_container_width=True, hide_index=True)
