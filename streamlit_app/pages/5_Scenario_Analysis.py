"""Page 5: Macro Scenario Analysis — Stress at flow-rate level.

V6 Roadmap Requirements:
- st.info() data limitation disclaimer
- Scenario weight sliders (Baseline/Mild/Stress) that sum to 100%
- Flow Rate Stress by Scenario table (baseline vs mild vs stress per bucket)
- Compounding effect callout
- Side-by-side scenario comparison
- Forward macro paths (UNRATE, HPI, DFF × 3 scenarios)
- Sensitivity sliders: Unemployment, Recovery Rate, Flow Rate Multiplier
- Weighted ECL across scenarios
- Mean reversion visualization
- Elasticity summary
- Sensitivity tornado
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
    load_macro_scenarios, load_forward_macro_paths, load_quarterly_stress_multipliers,
    load_ecl_by_scenario, load_sensitivity_results, load_multi_factor_elasticities,
    load_flow_rates_by_scenario,
)
from utils.styles import (
    inject_custom_css, sidebar_disclaimer, kpi_card, format_currency,
    SCENARIO_COLORS, GRADE_COLORS, GRADE_ORDER,
)

st.set_page_config(page_title="Scenario Analysis", layout="wide")
inject_custom_css()
sidebar_disclaimer()

st.title("Scenario Analysis & Stress Testing")
st.markdown("V6 multi-factor stress testing with flow-rate-level scenario stress and FEG framework.")

st.info(
    "**Scenario Analysis Data Note:** Flow rate stress is applied to synthetically "
    "derived rates. The compounding mathematics is exact; the base rates are approximate."
)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

macro = load_macro_scenarios()
fwd_paths = load_forward_macro_paths()
stress_mult = load_quarterly_stress_multipliers()
ecl_scen = load_ecl_by_scenario()
sens = load_sensitivity_results()
elasticities = load_multi_factor_elasticities()
flow_scen = load_flow_rates_by_scenario()
ps = sens.get("portfolio_summary", {})

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: Scenario Weights & Controls
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("### Scenario Weights")
st.sidebar.markdown("Adjust probability weights (should sum to 100%).")

w_baseline = st.sidebar.slider("Baseline Weight (%)", 0, 100, 75, 5)
w_mild = st.sidebar.slider("Mild Downturn Weight (%)", 0, 100, 15, 5)
w_stress = st.sidebar.slider("Stress Weight (%)", 0, 100, 10, 5)

total_weight = w_baseline + w_mild + w_stress
if total_weight != 100:
    st.sidebar.warning(f"Weights sum to {total_weight}%, not 100%. Results will be normalized.")

weights_norm = {
    "baseline": w_baseline / max(total_weight, 1),
    "mild": w_mild / max(total_weight, 1),
    "stress": w_stress / max(total_weight, 1),
}

st.sidebar.markdown("---")
st.sidebar.markdown("### Sensitivity Sliders")

slider_unemployment = st.sidebar.slider(
    "Unemployment Rate (%)", 3.0, 10.0, 4.2, 0.1,
    help="Adjust unemployment assumption — watch ECL update.",
)
slider_recovery = st.sidebar.slider(
    "Recovery Rate (%)", 5.0, 30.0, 17.0, 1.0,
    help="Adjust recovery rate — watch LGD and ECL adjust.",
)
slider_fr_multiplier = st.sidebar.slider(
    "Flow Rate Stress Multiplier", 0.85, 1.25, 1.00, 0.01,
    help="Multiplicative stress applied to ALL flow rates.",
)

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW: Three FEG Views
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Three FEG Views")

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Central ECL", format_currency(ps.get("central_ecl", 0)),
             f"ALLL: {ps.get('central_alll', 0):.1%}")
with c2:
    kpi_card("Pre-FEG ECL", format_currency(ps.get("prefeg_ecl", 0)),
             f"ALLL: {ps.get('prefeg_alll', 0):.1%}")
with c3:
    kpi_card("Post-FEG ECL", format_currency(ps.get("postfeg_ecl", 0)),
             f"ALLL: {ps.get('postfeg_alll', 0):.1%}")
with c4:
    variance = ps.get("postfeg_ecl", 0) - ps.get("central_ecl", 0)
    kpi_card("Post-FEG vs Central", format_currency(variance),
             f"+{variance / max(ps.get('central_ecl', 1), 1) * 100:.0f}%")

feg_ok = ps.get("feg_ordering_correct", False)
if feg_ok:
    st.success("FEG Ordering Validated: Pre-FEG (Baseline) < Central (Macro-Adjusted) < Post-FEG (Weighted)")
else:
    st.error("FEG Ordering Check Failed — investigate scenario calibration.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FLOW RATE STRESS BY SCENARIO TABLE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Flow Rate Stress by Scenario")
st.markdown(
    "Individual flow rates under each scenario. Stress is multiplicative — "
    "compounding across buckets produces non-linear FTR changes."
)

fr_cols = ["flow_rate_30", "flow_rate_60", "flow_rate_90", "flow_rate_120",
           "flow_rate_150", "flow_rate_180", "flow_rate_gco"]
fr_labels = {"flow_rate_30": "30+ FR", "flow_rate_60": "60+ FR", "flow_rate_90": "90+ FR",
             "flow_rate_120": "120+ FR", "flow_rate_150": "150+ FR", "flow_rate_180": "180+ FR",
             "flow_rate_gco": "GCO FR"}

# Show portfolio-average flow rates per scenario
scenarios_list = ["baseline", "central", "mild", "stress"]
fr_summary_rows = []
for scen in scenarios_list:
    scen_data = flow_scen[flow_scen["scenario"] == scen]
    if len(scen_data) == 0:
        continue
    row = {"Scenario": scen.title()}
    for col in fr_cols:
        if col in scen_data.columns:
            row[fr_labels[col]] = f"{scen_data[col].mean():.2%}"
    row["FTR"] = f"{scen_data['flow_through_rate'].mean():.3%}"
    fr_summary_rows.append(row)

if fr_summary_rows:
    fr_df = pd.DataFrame(fr_summary_rows)
    st.dataframe(fr_df, use_container_width=True, hide_index=True)

    # Compounding effect callout
    baseline_ftr = flow_scen[flow_scen["scenario"] == "baseline"]["flow_through_rate"].mean()
    stress_ftr = flow_scen[flow_scen["scenario"] == "stress"]["flow_through_rate"].mean()
    if baseline_ftr > 0:
        pct_increase = (stress_ftr / baseline_ftr - 1) * 100
        st.markdown(
            f"> **Compounding Effect:** Baseline FTR = {baseline_ftr:.3%} → "
            f"Stress FTR = {stress_ftr:.3%} → **{pct_increase:+.0f}% increase** "
            f"(due to multiplicative compounding across 7 flow rate transitions)"
        )

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# WEIGHTED ECL ACROSS SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Probability-Weighted ECL")

# Compute ECL per scenario
ecl_by_scen = {}
for scen in ["central", "mild", "stress"]:
    scen_ecl = ecl_scen[ecl_scen["scenario"] == scen]["total_ecl"].sum()
    ecl_by_scen[scen] = scen_ecl

# Apply user weights (map baseline→central for weighting)
weighted_ecl = (
    weights_norm.get("baseline", 0.75) * ecl_by_scen.get("central", 0)
    + weights_norm.get("mild", 0.15) * ecl_by_scen.get("mild", 0)
    + weights_norm.get("stress", 0.10) * ecl_by_scen.get("stress", 0)
)

# Apply slider adjustments
# Unemployment sensitivity: approximate linear impact from elasticity
unemp_elast = elasticities.get("UNRATE", {}).get("elasticity", 0.5)
unemp_baseline = 4.2
unemp_adj = 1 + unemp_elast * (slider_unemployment - unemp_baseline) / unemp_baseline

# Recovery sensitivity: higher recovery → lower ECL
recovery_baseline = 17.0
recovery_adj = (100 - slider_recovery) / (100 - recovery_baseline)

# Flow rate multiplier: direct multiplicative impact on ECL
fr_adj = slider_fr_multiplier

adjusted_ecl = weighted_ecl * unemp_adj * recovery_adj * fr_adj

wc1, wc2, wc3, wc4 = st.columns(4)
with wc1:
    kpi_card("Central ECL", format_currency(ecl_by_scen.get("central", 0)),
             f"Weight: {weights_norm['baseline']:.0%}")
with wc2:
    kpi_card("Mild Downturn ECL", format_currency(ecl_by_scen.get("mild", 0)),
             f"Weight: {weights_norm['mild']:.0%}")
with wc3:
    kpi_card("Stress ECL", format_currency(ecl_by_scen.get("stress", 0)),
             f"Weight: {weights_norm['stress']:.0%}")
with wc4:
    kpi_card("Weighted ECL (Adjusted)", format_currency(adjusted_ecl),
             f"Unemployment: {slider_unemployment}% | Recovery: {slider_recovery}%")

# Bar chart comparison
ecl_compare = pd.DataFrame([
    {"Scenario": "Central", "ECL": ecl_by_scen.get("central", 0), "Color": SCENARIO_COLORS["central"]},
    {"Scenario": "Mild", "ECL": ecl_by_scen.get("mild", 0), "Color": SCENARIO_COLORS["mild"]},
    {"Scenario": "Stress", "ECL": ecl_by_scen.get("stress", 0), "Color": SCENARIO_COLORS["stress"]},
    {"Scenario": "Weighted (Adj.)", "ECL": adjusted_ecl, "Color": "#2c3e50"},
])

fig = go.Figure()
fig.add_trace(go.Bar(
    x=ecl_compare["Scenario"], y=ecl_compare["ECL"],
    marker_color=ecl_compare["Color"].tolist(),
    text=[format_currency(v) for v in ecl_compare["ECL"]],
    textposition="outside",
))
fig.update_layout(
    title="ECL Across Scenarios vs Probability-Weighted Result",
    template="plotly_white", height=450,
    yaxis_title="ECL ($)",
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FORWARD MACRO PATHS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### 8-Quarter Forward Macro Paths")
st.markdown("Central uses actual 2019 FRED values. Mild/Stress are constructed recession trajectories.")

macro_vars = ["UNRATE", "CSUSHPINSA", "DFF"]
var_labels = {
    "UNRATE": "Unemployment Rate (%)",
    "CSUSHPINSA": "Case-Shiller HPI",
    "DFF": "Federal Funds Rate (%)",
}

cols = st.columns(3)
for i, var in enumerate(macro_vars):
    with cols[i]:
        fig = px.line(
            fwd_paths, x="quarter_label", y=var, color="scenario",
            color_discrete_map=SCENARIO_COLORS,
            title=var_labels.get(var, var),
            markers=True,
        )
        baseline = macro.get("baseline_levels", {}).get(var)
        if baseline:
            fig.add_hline(
                y=baseline, line_dash="dot", line_color="gray",
                annotation_text=f"Baseline: {baseline}",
            )
        fig.update_layout(
            template="plotly_white", height=350,
            xaxis_tickangle=45, showlegend=(i == 0),
            yaxis_title=var_labels.get(var, var),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MEAN REVERSION VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Mean Reversion After Forecast Horizon")
st.markdown(
    "Macro variables revert to long-run mean after the 8-quarter explicit forecast horizon. "
    "This shows the reversion path assumed in CECL Phase 2 (12-quarter reversion)."
)

baseline_levels = macro.get("baseline_levels", {})
if len(fwd_paths) > 0 and baseline_levels:
    # Build extended paths showing reversion
    reversion_quarters = 12
    last_q = fwd_paths["quarter_label"].max() if "quarter_label" in fwd_paths.columns else "Q8"

    fig = go.Figure()
    for var, label in var_labels.items():
        if var not in fwd_paths.columns:
            continue
        bl = baseline_levels.get(var, 0)
        # Show stress path then revert
        stress_path = fwd_paths[fwd_paths["scenario"] == "stress"][["quarter_label", var]].copy()
        if len(stress_path) == 0:
            continue

        # Extend with reversion
        last_val = stress_path[var].iloc[-1] if len(stress_path) > 0 else bl
        reversion = [last_val + (bl - last_val) * (i / reversion_quarters)
                     for i in range(1, reversion_quarters + 1)]
        rev_labels = [f"R{i}" for i in range(1, reversion_quarters + 1)]

        all_q = stress_path["quarter_label"].tolist() + rev_labels
        all_v = stress_path[var].tolist() + reversion

        fig.add_trace(go.Scatter(
            x=all_q, y=all_v,
            name=label, mode="lines+markers",
        ))
        fig.add_hline(y=bl, line_dash="dot", line_color="gray", opacity=0.5)

    # Mark the reversion phase — use shape with xref="x" for categorical axis
    try:
        all_x_labels = stress_path["quarter_label"].tolist() + rev_labels
        r1_idx = all_x_labels.index("R1")
        r_end_idx = all_x_labels.index(f"R{reversion_quarters}")
        fig.add_shape(
            type="rect",
            x0=r1_idx - 0.5, x1=r_end_idx + 0.5,
            y0=0, y1=1, yref="paper",
            fillcolor="rgba(52, 152, 219, 0.08)",
            layer="below", line_width=0,
        )
        fig.add_annotation(
            x=(r1_idx + r_end_idx) / 2, y=1.05, yref="paper",
            text="Reversion Phase", showarrow=False,
            font=dict(size=12, color="#3498db"),
        )
    except Exception:
        pass  # Graceful fallback if categorical axis indexing fails
    fig.update_layout(
        template="plotly_white", height=400,
        title="Stress Scenario with Mean Reversion (12-Quarter)",
        xaxis_title="Quarter", yaxis_title="Value",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# QUARTERLY STRESS MULTIPLIERS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Quarterly Composite Stress Multipliers")

if "composite_multiplier" in stress_mult.columns:
    mult_col = "composite_multiplier"
elif "multiplier" in stress_mult.columns:
    mult_col = "multiplier"
else:
    mult_cols = [c for c in stress_mult.columns if "mult" in c.lower() or "composite" in c.lower()]
    mult_col = mult_cols[0] if mult_cols else None

if mult_col:
    fig = px.line(
        stress_mult, x="quarter_label", y=mult_col, color="scenario",
        color_discrete_map=SCENARIO_COLORS,
        title="Composite Stress Multiplier by Quarter",
        markers=True,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(template="plotly_white", height=400, yaxis_title="Multiplier")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY TORNADO
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Sensitivity Tornado (5-Factor)")
st.markdown("Impact of individual factor shocks on portfolio ECL.")

tornado_data = sens.get("tornado_data", [])
if tornado_data:
    df_t = pd.DataFrame(tornado_data)

    # Use low_ecl and high_ecl if available
    if "low_ecl" in df_t.columns and "high_ecl" in df_t.columns:
        df_t["range"] = df_t["high_ecl"] - df_t["low_ecl"]
        df_t = df_t.sort_values("range", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df_t["factor"], x=df_t["low_ecl"],
            orientation="h", name="Low Scenario",
            marker_color="#3498db",
            text=[format_currency(v) for v in df_t["low_ecl"]],
            textposition="inside",
        ))
        fig.add_trace(go.Bar(
            y=df_t["factor"], x=df_t["high_ecl"],
            orientation="h", name="High Scenario",
            marker_color="#e74c3c",
            text=[format_currency(v) for v in df_t["high_ecl"]],
            textposition="inside",
        ))
        fig.update_layout(
            title="ECL Sensitivity Tornado (V6 — 5 Factors)",
            template="plotly_white", height=400,
            barmode="overlay",
            xaxis_title="ECL ($)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

    elif "low" in df_t.columns and "high" in df_t.columns:
        df_t["range"] = df_t["high"] - df_t["low"]
        df_t = df_t.sort_values("range", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df_t["factor"], x=df_t["low"],
            orientation="h", name="Low", marker_color="#3498db",
        ))
        fig.add_trace(go.Bar(
            y=df_t["factor"], x=df_t["high"],
            orientation="h", name="High", marker_color="#e74c3c",
        ))
        fig.update_layout(
            title="ECL Sensitivity Tornado",
            template="plotly_white", height=400,
            barmode="overlay", xaxis_title="ECL ($)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})
else:
    st.info("No tornado sensitivity data available.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-FACTOR ELASTICITIES
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Multi-Factor Elasticities (Ridge Regression)")

elast_rows = []
interpretations = {
    "UNRATE": "Higher unemployment → higher default entry rate",
    "CSUSHPINSA": "Rising home prices → lower defaults (wealth effect)",
    "DFF": "Higher rates → higher debt service → more defaults",
}
for var, stats in elasticities.items():
    elast_rows.append({
        "Variable": var,
        "Elasticity": f"{stats.get('elasticity', 0):+.4f}",
        "Slope": f"{stats.get('slope', 0):.6f}",
        "R²": f"{stats.get('r_squared', 0):.3f}",
        "Interpretation": interpretations.get(var, ""),
    })

if elast_rows:
    st.dataframe(pd.DataFrame(elast_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# FLOW RATE STRESS BY GRADE (DETAILED TABLE)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("### Flow Rate Detail by Grade × Scenario")

grade_filter = st.multiselect("Select grades:", GRADE_ORDER, default=GRADE_ORDER, key="scen_grades")

detail = flow_scen[flow_scen["grade"].isin(grade_filter)].copy()
detail = detail.sort_values(["grade", "scenario"])

display_cols = ["grade", "scenario", "stress_multiplier"] + fr_cols + ["flow_through_rate"]
rename_map = {
    "grade": "Grade", "scenario": "Scenario", "stress_multiplier": "Stress Mult.",
    "flow_through_rate": "FTR",
}
rename_map.update(fr_labels)

detail_display = detail[display_cols].rename(columns=rename_map)

# Format percentages
for col in list(fr_labels.values()) + ["FTR"]:
    if col in detail_display.columns:
        detail_display[col] = detail_display[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "—")

st.dataframe(detail_display, use_container_width=True, hide_index=True)
