"""Page 3: Vintage Performance — MOB-indexed vintage curves with metric toggle.

V6 Roadmap Requirements:
- Metric toggle: Cumulative Default Rate / Marginal PD / Cumulative Loss Rate
- Filters: Grades, Vintages
- Vintage curves chart (Y=cum default, X=MOB, each line=vintage year)
- Marginal PD curves by grade
- Vintage comparison table (MOB 6/12/18/24/36)
- Seasoning pattern chart (avg default rate by MOB across all vintages)
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
    load_vintage_curves, load_vintage_curves_mob,
    load_marginal_pd_by_grade, load_seasoning_pattern,
)
from utils.styles import (
    inject_custom_css, sidebar_disclaimer, kpi_card, format_currency,
    GRADE_COLORS, GRADE_ORDER,
)

st.set_page_config(page_title="Vintage Performance", layout="wide")
inject_custom_css()
sidebar_disclaimer()

st.title("Vintage Performance Analysis")
st.markdown("Cumulative default curves by origination vintage over time (months on book).")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

vintage_summary = load_vintage_curves()
vintage_mob = load_vintage_curves_mob()
marginal_pd = load_marginal_pd_by_grade()
seasoning = load_seasoning_pattern()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: Filters & Metric Toggle
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("### Metric")
metric = st.sidebar.radio(
    "Select Metric",
    ["Cumulative Default Rate", "Cumulative Loss Rate", "Marginal PD"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

all_vintages = sorted(vintage_mob["vintage_year"].unique())
selected_vintages = st.sidebar.multiselect(
    "Vintages", all_vintages,
    default=all_vintages,
    help="Select vintage years to display.",
)

available_grades = sorted(marginal_pd["grade"].unique()) if "grade" in marginal_pd.columns else GRADE_ORDER
selected_grades = st.sidebar.multiselect("Grades", available_grades, default=available_grades)

mob_max = st.sidebar.slider("Max MOB (months)", 6, 60, 36, 3,
                             help="Maximum months on book to display.")

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════

vs = vintage_summary.copy()
if selected_vintages:
    vs = vs[vs["vintage_year"].isin(selected_vintages)]

total_loans = vs["total_loans"].sum()
total_defaults = vs["defaults"].sum()
overall_dr = total_defaults / max(total_loans, 1)
total_funded = vs["total_funded"].sum()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    kpi_card("Vintages Selected", str(len(selected_vintages)))
with c2:
    kpi_card("Total Loans", f"{total_loans:,.0f}")
with c3:
    kpi_card("Total Funded", format_currency(total_funded))
with c4:
    kpi_card("Overall Default Rate", f"{overall_dr:.2%}")
with c5:
    worst_vintage = vs.loc[vs["default_rate"].idxmax()] if len(vs) > 0 else None
    if worst_vintage is not None:
        kpi_card("Worst Vintage", f"{int(worst_vintage['vintage_year'])} ({worst_vintage['default_rate']:.1%})")
    else:
        kpi_card("Worst Vintage", "N/A")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# VINTAGE CURVES (Main Chart)
# ══════════════════════════════════════════════════════════════════════════════

if metric == "Cumulative Default Rate":
    st.markdown("### Cumulative Default Rate by Vintage")
    st.markdown("Each line represents one origination vintage. X-axis = months on book (MOB).")

    vdata = vintage_mob[
        (vintage_mob["vintage_year"].isin(selected_vintages)) &
        (vintage_mob["mob"] <= mob_max)
    ].copy()

    if len(vdata) > 0:
        fig = px.line(
            vdata, x="mob", y="cum_default_rate",
            color="vintage_year",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Cumulative Default Rate by Vintage (MOB Curves)",
        )
        fig.update_layout(
            template="plotly_white", height=500,
            xaxis_title="Months on Book (MOB)",
            yaxis_title="Cumulative Default Rate",
            yaxis_tickformat=".1%",
            legend_title="Vintage",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

elif metric == "Cumulative Loss Rate":
    st.markdown("### Cumulative Loss Rate by Vintage")
    st.markdown("Cumulative funded amount of defaults / total cohort funded amount.")

    vdata = vintage_mob[
        (vintage_mob["vintage_year"].isin(selected_vintages)) &
        (vintage_mob["mob"] <= mob_max)
    ].copy()

    if len(vdata) > 0:
        fig = px.line(
            vdata, x="mob", y="cum_loss_rate",
            color="vintage_year",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Cumulative Loss Rate by Vintage (MOB Curves)",
        )
        fig.update_layout(
            template="plotly_white", height=500,
            xaxis_title="Months on Book (MOB)",
            yaxis_title="Cumulative Loss Rate",
            yaxis_tickformat=".1%",
            legend_title="Vintage",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

else:  # Marginal PD
    st.markdown("### Marginal PD by Grade")
    st.markdown("Monthly incremental default probability by grade (hazard rate). 6-month rolling average applied.")

    mp = marginal_pd.copy()
    if selected_vintages:
        mp = mp[mp["vintage_year"].isin(selected_vintages)]
    if selected_grades:
        mp = mp[mp["grade"].isin(selected_grades)]
    mp = mp[mp["mob"] <= mob_max]

    mp_avg = mp.groupby(["grade", "mob"])["marginal_pd"].mean().reset_index()
    mp_avg = mp_avg.sort_values(["grade", "mob"])
    mp_avg["marginal_pd_smooth"] = mp_avg.groupby("grade")["marginal_pd"].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )

    if len(mp_avg) > 0:
        fig = px.line(
            mp_avg, x="mob", y="marginal_pd_smooth",
            color="grade",
            color_discrete_map=GRADE_COLORS,
            title="Marginal PD by Grade (6-Month Rolling Average)",
        )
        fig.update_layout(
            template="plotly_white", height=500,
            xaxis_title="Months on Book (MOB)",
            yaxis_title="Marginal PD (Monthly)",
            yaxis_tickformat=".3%",
            legend_title="Grade",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# VINTAGE COMPARISON TABLE (MOB milestones)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Vintage Comparison Table")
st.markdown("Cumulative default rate at key MOB milestones.")

mob_milestones = [6, 12, 18, 24, 36]
mob_milestones = [m for m in mob_milestones if m <= mob_max]

comp_rows = []
for vy in sorted(selected_vintages):
    vy_data = vintage_mob[vintage_mob["vintage_year"] == vy]
    row = {"Vintage": int(vy)}
    for m in mob_milestones:
        m_data = vy_data[vy_data["mob"] == m]
        if len(m_data) > 0:
            row[f"MOB {m}"] = f"{m_data.iloc[0]['cum_default_rate']:.2%}"
        else:
            row[f"MOB {m}"] = "—"
    row["Terminal"] = f"{vy_data['cum_default_rate'].max():.2%}" if len(vy_data) > 0 else "—"
    comp_rows.append(row)

comp_df = pd.DataFrame(comp_rows)
st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MARGINAL PD CURVES BY GRADE (always shown when not in Marginal PD mode)
# ══════════════════════════════════════════════════════════════════════════════

if metric != "Marginal PD":
    st.markdown("### Marginal PD Curves by Grade")
    st.markdown("Monthly incremental default probability across selected vintages (6-month smoothed).")

    mp = marginal_pd.copy()
    if selected_vintages:
        mp = mp[mp["vintage_year"].isin(selected_vintages)]
    if selected_grades:
        mp = mp[mp["grade"].isin(selected_grades)]
    mp = mp[mp["mob"] <= mob_max]

    mp_avg = mp.groupby(["grade", "mob"])["marginal_pd"].mean().reset_index()
    mp_avg = mp_avg.sort_values(["grade", "mob"])
    mp_avg["marginal_pd_smooth"] = mp_avg.groupby("grade")["marginal_pd"].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )

    col1, col2 = st.columns(2)
    with col1:
        if len(mp_avg) > 0:
            fig = px.line(
                mp_avg, x="mob", y="marginal_pd_smooth",
                color="grade", color_discrete_map=GRADE_COLORS,
                title="Marginal PD by Grade (Smoothed)",
            )
            fig.update_layout(
                template="plotly_white", height=400,
                xaxis_title="MOB", yaxis_title="Marginal PD",
                yaxis_tickformat=".3%",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

    with col2:
        mp2 = marginal_pd.copy()
        if selected_vintages:
            mp2 = mp2[mp2["vintage_year"].isin(selected_vintages)]
        if selected_grades:
            mp2 = mp2[mp2["grade"].isin(selected_grades)]
        mp2 = mp2[mp2["mob"] <= mob_max]
        mp_vy = mp2.groupby(["vintage_year", "mob"])["marginal_pd"].mean().reset_index()
        mp_vy = mp_vy.sort_values(["vintage_year", "mob"])
        mp_vy["marginal_pd_smooth"] = mp_vy.groupby("vintage_year")["marginal_pd"].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )

        if len(mp_vy) > 0:
            fig = px.line(
                mp_vy, x="mob", y="marginal_pd_smooth",
                color="vintage_year",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Marginal PD by Vintage (Smoothed)",
            )
            fig.update_layout(
                template="plotly_white", height=400,
                xaxis_title="MOB", yaxis_title="Marginal PD",
                yaxis_tickformat=".3%",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

    st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SEASONING PATTERN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Seasoning Pattern")
st.markdown(
    "Average cumulative default rate by MOB across all vintages — "
    "shows the typical lifecycle curve for LendingClub personal loans."
)

sea = seasoning[seasoning["mob"] <= mob_max].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=sea["mob"], y=sea["avg_cum_default_rate"],
    mode="lines+markers",
    line=dict(color="#e74c3c", width=3),
    marker=dict(size=4),
    name="Avg Cumulative Default Rate",
    fill="tozeroy",
    fillcolor="rgba(231, 76, 60, 0.1)",
))
fig.update_layout(
    title="Seasoning Curve — Average Default Rate by MOB",
    template="plotly_white", height=400,
    xaxis_title="Months on Book (MOB)",
    yaxis_title="Avg Cumulative Default Rate",
    yaxis_tickformat=".1%",
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# VINTAGE SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Vintage Summary")

vs_display = vintage_summary[vintage_summary["vintage_year"].isin(selected_vintages)].copy()
vs_display = vs_display.sort_values("vintage_year")
vs_display["default_rate"] = vs_display["default_rate"].apply(lambda x: f"{x:.2%}")
vs_display["total_funded"] = vs_display["total_funded"].apply(lambda x: format_currency(x))
vs_display.columns = ["Vintage", "Defaults", "Total Loans", "Default Rate", "Total Funded"]
st.dataframe(vs_display, use_container_width=True, hide_index=True)
