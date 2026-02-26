"""Page 6: Model Monitoring — RAG dashboard with discrimination, calibration, stability.

V6 Roadmap Requirements:
- RAG Dashboard with color-coded status table
- Gini Over Time line chart with green/amber/red zones
- PSI by period bar chart with thresholds
- CSI/VDI by feature (horizontal bar)
- Calibration Check: predicted PD vs actual default by decile
- Out-of-Time Performance: separate metrics by validation period
- Benchmark Population tab (external PSI comparison)
- Model comparison table (Scorecard vs XGBoost vs LightGBM)
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.data_loader import load_rag_status, load_pd_metrics, load_psi_by_period, load_csi_by_feature
from utils.styles import inject_custom_css, sidebar_disclaimer, kpi_card, RAG_COLORS
from utils.charts import rag_status_styled

st.set_page_config(page_title="Model Monitoring", layout="wide")
inject_custom_css()
sidebar_disclaimer()

st.title("Model Monitoring Dashboard")
st.markdown("RAG (Red/Amber/Green) status tracking for PD scorecard discrimination, calibration, and stability metrics.")

st.info(
    "**Metric Source Note:** PD model metrics (Gini, AUC, KS, PSI) are fully based on real "
    "observed data. ECL-related metrics use synthetically derived flow rates."
)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = APP_DIR.parent
RESULTS = PROJECT_DIR / "data" / "results"


def _find_file(filename: str, subdir: str = "08_validation") -> Path | None:
    """Find file in subdirectory or top-level results, return None if not found."""
    for path in [RESULTS / subdir / filename, RESULTS / filename]:
        if path.exists():
            return path
    return None


def _load_json_safe(filename: str, subdir: str = "08_validation") -> dict:
    """Load JSON from subdirectory with top-level fallback."""
    path = _find_file(filename, subdir)
    if path is None:
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _load_csv_safe(filename: str, subdir: str = "08_validation") -> pd.DataFrame:
    """Load CSV from subdirectory with top-level fallback."""
    path = _find_file(filename, subdir)
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


try:
    rag = load_rag_status()
except Exception:
    rag = pd.DataFrame(columns=["Metric", "Value", "RAG_Status", "Symbol",
                                  "Threshold_Green", "Threshold_Amber", "Threshold_Red"])

if "RAG_Status" in rag.columns:
    rag["RAG_Status"] = rag["RAG_Status"].str.strip().str.lower()

try:
    metrics = load_pd_metrics()
except Exception:
    metrics = {}

try:
    psi = load_psi_by_period()
except Exception:
    psi = pd.DataFrame(columns=["Period", "PSI", "RAG"])

if "RAG" in psi.columns:
    psi["RAG"] = psi["RAG"].str.strip().str.lower()

try:
    csi = load_csi_by_feature()
except Exception:
    csi = pd.DataFrame(columns=["Feature", "CSI", "RAG"])

if "RAG" in csi.columns:
    csi["RAG"] = csi["RAG"].str.strip().str.lower()

# Load additional data for new sections
validation_report = _load_json_safe("validation_report.json")
model_comparison = _load_json_safe("model_comparison.json")
ext_psi = _load_csv_safe("external_validation_psi.csv")

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Model Health Summary")

rag_counts = rag["RAG_Status"].value_counts()
green_count = rag_counts.get("green", 0)
amber_count = rag_counts.get("amber", 0)
red_count = rag_counts.get("red", 0)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    test_gini = metrics.get("test_gini_pct", 0)
    kpi_card("Test Gini", f"{test_gini:.1f}%")
with c2:
    kpi_card("Green Metrics", str(green_count), f"of {len(rag)} total", True)
with c3:
    kpi_card("Amber Metrics", str(amber_count), f"of {len(rag)} total", False)
with c4:
    kpi_card("Red Metrics", str(red_count), f"of {len(rag)} total", False)
with c5:
    health = "Healthy" if red_count == 0 and amber_count <= 2 else "Review" if red_count == 0 else "Action Required"
    kpi_card("Overall Status", health)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# RAG STATUS TABLE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### RAG Status Table")
st.markdown("Thresholds follow institutional MRM (Model Risk Management) standards.")

if len(rag) > 0:
    rag_html = rag_status_styled(rag)
    st.markdown(rag_html, unsafe_allow_html=True)
else:
    st.warning("RAG status data not available. Check that rag_status_table.csv exists in data/results/.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# GINI OVER TIME (with zones)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Gini Coefficient Over Time")
st.markdown("Model discrimination tracked across validation periods with RAG zones.")

# Build Gini timeline from available data
gini_timeline = []
# Train
train_m = metrics.get("train", {})
if isinstance(train_m, dict) and "Gini" in train_m:
    gini_timeline.append({"Period": "Train (2007-2014)", "Gini": train_m["Gini"] * 100, "Model": "Scorecard"})

# Validation
val_m = metrics.get("validation", {})
if isinstance(val_m, dict) and "Gini" in val_m:
    gini_timeline.append({"Period": "Validation (2016)", "Gini": val_m["Gini"] * 100, "Model": "Scorecard"})

# Test
test_m = metrics.get("test", {})
if isinstance(test_m, dict) and "Gini" in test_m:
    gini_timeline.append({"Period": "Test (2017-2018)", "Gini": test_m["Gini"] * 100, "Model": "Scorecard"})

# XGBoost comparison
for model_name, model_key in [("XGBoost", "xgboost_full"), ("LightGBM", "lightgbm_full")]:
    mc = model_comparison.get(model_key, {})
    for split, label in [("test", "Test (2017-2018)")]:
        sm = mc.get(split, {})
        if "Gini" in sm:
            gini_timeline.append({"Period": label, "Gini": sm["Gini"] * 100, "Model": model_name})

if gini_timeline:
    gini_df = pd.DataFrame(gini_timeline)

    fig = go.Figure()

    # RAG zones
    fig.add_hrect(y0=42, y1=70, fillcolor="rgba(39, 174, 96, 0.1)", line_width=0,
                  annotation_text="Green Zone (≥42%)", annotation_position="top left")
    fig.add_hrect(y0=35, y1=42, fillcolor="rgba(243, 156, 18, 0.1)", line_width=0,
                  annotation_text="Amber Zone (35-42%)")
    fig.add_hrect(y0=0, y1=35, fillcolor="rgba(231, 76, 60, 0.1)", line_width=0,
                  annotation_text="Red Zone (<35%)")

    for model in gini_df["Model"].unique():
        mdf = gini_df[gini_df["Model"] == model]
        fig.add_trace(go.Scatter(
            x=mdf["Period"], y=mdf["Gini"],
            mode="lines+markers",
            name=model,
            marker=dict(size=10),
        ))

    fig.update_layout(
        template="plotly_white", height=450,
        yaxis_title="Gini (%)", yaxis_range=[25, 55],
        xaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PSI BY PERIOD + CSI BY FEATURE (Side by Side)
# ══════════════════════════════════════════════════════════════════════════════

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Population Stability Index (PSI)")
    st.markdown("PSI < 0.10 (Green) | 0.10–0.25 (Amber) | > 0.25 (Red)")

    if len(psi) == 0:
        st.warning("PSI data not available.")
    else:
        psi_colors = [RAG_COLORS.get(r, "#999") for r in psi["RAG"]]
        fig = go.Figure(go.Bar(
            x=psi["Period"], y=psi["PSI"],
            marker_color=psi_colors,
            text=[f"{v:.4f}" for v in psi["PSI"]],
            textposition="outside",
        ))
        fig.add_hline(y=0.10, line_dash="dash", line_color="#f39c12", annotation_text="Amber: 0.10")
        fig.add_hline(y=0.25, line_dash="dash", line_color="#e74c3c", annotation_text="Red: 0.25")
        fig.update_layout(
            title="PSI by Validation Period",
            template="plotly_white", height=400, yaxis_title="PSI",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

with col2:
    st.markdown("### Characteristic Stability Index (CSI)")
    if len(csi) == 0:
        st.warning("CSI data not available.")
    else:
        csi_sorted = csi.sort_values("CSI", ascending=True)
        csi_colors = [RAG_COLORS.get(r, "#999") for r in csi_sorted["RAG"]]
        fig = go.Figure(go.Bar(
            y=csi_sorted["Feature"], x=csi_sorted["CSI"],
            orientation="h",
            marker_color=csi_colors,
            text=[f"{v:.4f}" for v in csi_sorted["CSI"]],
            textposition="outside",
        ))
        fig.add_vline(x=0.10, line_dash="dash", line_color="#f39c12")
        fig.add_vline(x=0.25, line_dash="dash", line_color="#e74c3c")
        fig.update_layout(
            title="CSI by Feature",
            template="plotly_white", height=max(400, len(csi) * 30),
            xaxis_title="CSI",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION CHECK — Predicted PD vs Actual by Decile
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Calibration Check — Predicted vs Actual Default Rate")
st.markdown("Predicted PD vs observed default rate by score decile. Good calibration = points near 45° line.")

cal_metrics = validation_report.get("sections", {}).get("calibration", {}).get("metrics", {})

# Check for calibration image first (check subdir then top-level)
cal_img = _find_file("calibration_decile.png") or RESULTS / "calibration_decile.png"
grade_cal_img = _find_file("grade_calibration.png") or RESULTS / "grade_calibration.png"

col_cal1, col_cal2 = st.columns(2)

with col_cal1:
    if cal_img.exists():
        st.image(str(cal_img), caption="Calibration by Score Decile", use_container_width=True)
    else:
        # Generate from metrics
        brier_sc = cal_metrics.get("brier_scorecard", 0)
        brier_xgb = cal_metrics.get("brier_xgboost", 0)
        st.markdown(f"**Brier Score (Scorecard):** {brier_sc:.4f}")
        st.markdown(f"**Brier Score (XGBoost):** {brier_xgb:.4f}")

        hl_stat = cal_metrics.get("hosmer_lemeshow_stat", 0)
        hl_p = cal_metrics.get("hosmer_lemeshow_p", 0)
        if hl_p < 0.05:
            st.warning(f"Hosmer-Lemeshow: χ² = {hl_stat:.1f}, p = {hl_p:.4f} (Significant — poor calibration)")
        else:
            st.success(f"Hosmer-Lemeshow: χ² = {hl_stat:.1f}, p = {hl_p:.4f} (Good calibration)")

with col_cal2:
    if grade_cal_img.exists():
        st.image(str(grade_cal_img), caption="Calibration by Grade", use_container_width=True)
    else:
        st.info("Grade calibration chart not available.")

# Calibration metrics summary
st.markdown("**Calibration Metrics:**")
brier_sc = cal_metrics.get("brier_scorecard", 0)
brier_xgb = cal_metrics.get("brier_xgboost", 0)
hl_stat = cal_metrics.get("hosmer_lemeshow_stat", 0)
hl_p = cal_metrics.get("hosmer_lemeshow_p", 0)

cal_df = pd.DataFrame([
    {"Metric": "Brier Score (Scorecard)", "Value": f"{brier_sc:.4f}", "Assessment": "Lower is better"},
    {"Metric": "Brier Score (XGBoost)", "Value": f"{brier_xgb:.4f}", "Assessment": "Lower is better"},
    {"Metric": "Hosmer-Lemeshow χ²", "Value": f"{hl_stat:.1f}", "Assessment": "p < 0.05 → poor calibration"},
    {"Metric": "Hosmer-Lemeshow p-value", "Value": f"{hl_p:.4f}",
     "Assessment": "RED" if hl_p < 0.01 else ("AMBER" if hl_p < 0.05 else "GREEN")},
])
st.dataframe(cal_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# OUT-OF-TIME PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Out-of-Time Performance")
st.markdown("Model discrimination and calibration on separate time periods (train vs validation vs test).")

# Build OOT table from metrics and model_comparison
oot_rows = []

# Scorecard
for split, label in [("train", "Train (2007-2014)"), ("validation", "Validation (2016)"), ("test", "Test (2017-2018)")]:
    sm = metrics.get(split, {})
    if isinstance(sm, dict):
        oot_rows.append({
            "Period": label,
            "Model": "Scorecard",
            "AUC": f"{sm.get('AUC', 0):.4f}",
            "Gini": f"{sm.get('Gini', 0):.4f}",
            "KS": f"{sm.get('KS', 0):.4f}",
            "Brier": f"{sm.get('Brier', 0):.4f}",
        })

# XGBoost
for split, label in [("train", "Train"), ("validation", "Validation"), ("test", "Test")]:
    mc = model_comparison.get("xgboost_full", {}).get(split, {})
    if mc:
        oot_rows.append({
            "Period": label,
            "Model": "XGBoost",
            "AUC": f"{mc.get('AUC', 0):.4f}",
            "Gini": f"{mc.get('Gini', 0):.4f}",
            "KS": f"{mc.get('KS', 0):.4f}",
            "Brier": f"{mc.get('Brier', 0):.4f}",
        })

if oot_rows:
    oot_df = pd.DataFrame(oot_rows)
    st.dataframe(oot_df, use_container_width=True, hide_index=True)

    # Degradation check
    sc_train_auc = metrics.get("train", {}).get("AUC", 0) if isinstance(metrics.get("train"), dict) else 0
    sc_test_auc = metrics.get("test", {}).get("AUC", 0) if isinstance(metrics.get("test"), dict) else 0
    gap = sc_train_auc - sc_test_auc
    if gap < 0.03:
        st.success(f"Train-Test AUC Gap: {gap:.4f} (GREEN — minimal overfitting)")
    elif gap < 0.05:
        st.warning(f"Train-Test AUC Gap: {gap:.4f} (AMBER — moderate degradation)")
    else:
        st.error(f"Train-Test AUC Gap: {gap:.4f} (RED — significant degradation)")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK POPULATION (External PSI)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Benchmark Population Comparison")
st.markdown("Score distribution comparison against 2014 benchmark population (external validation).")

if len(ext_psi) > 0:
    col_bp1, col_bp2 = st.columns(2)

    with col_bp1:
        # Score distribution comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ext_psi["bin_lower"].astype(str) + "-" + ext_psi["bin_upper"].astype(str),
            y=ext_psi["expected_pct"],
            name="Expected (Model)",
            marker_color="#3498db", opacity=0.7,
        ))
        fig.add_trace(go.Bar(
            x=ext_psi["bin_lower"].astype(str) + "-" + ext_psi["bin_upper"].astype(str),
            y=ext_psi["actual_pct"],
            name="Actual (Benchmark 2014)",
            marker_color="#e74c3c", opacity=0.7,
        ))
        fig.update_layout(
            title="Score Distribution: Model vs Benchmark",
            template="plotly_white", height=400,
            barmode="group", xaxis_tickangle=45,
            yaxis_title="Proportion", yaxis_tickformat=".1%",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

    with col_bp2:
        # PSI by bin
        fig = go.Figure(go.Bar(
            x=ext_psi["bin_lower"].astype(str) + "-" + ext_psi["bin_upper"].astype(str),
            y=ext_psi["psi_bin"],
            marker_color="#e74c3c",
            text=[f"{v:.3f}" for v in ext_psi["psi_bin"]],
            textposition="outside",
        ))
        fig.add_hline(y=0.10, line_dash="dash", line_color="#f39c12")
        fig.update_layout(
            title="PSI Contribution by Score Bin",
            template="plotly_white", height=400,
            xaxis_tickangle=45, yaxis_title="PSI (bin)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "scrollZoom": True})

    total_ext_psi = ext_psi["total_psi"].iloc[0] if "total_psi" in ext_psi.columns else ext_psi["psi_bin"].sum()
    if total_ext_psi > 0.25:
        st.error(f"External PSI = {total_ext_psi:.3f} (RED — significant distribution shift vs benchmark)")
    elif total_ext_psi > 0.10:
        st.warning(f"External PSI = {total_ext_psi:.3f} (AMBER — moderate shift)")
    else:
        st.success(f"External PSI = {total_ext_psi:.3f} (GREEN — stable)")
else:
    st.info("Benchmark population data not available.")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### Model Comparison (Scorecard vs ML Ensemble)")

model_rows = []
for model_name, model_key in [
    ("Logistic Regression (Scorecard)", "logistic_regression"),
    ("XGBoost (Full)", "xgboost_full"),
    ("LightGBM (Full)", "lightgbm_full"),
    ("XGBoost (Selected)", "xgboost_selected"),
    ("LightGBM (Selected)", "lightgbm_selected"),
]:
    mc_data = model_comparison.get(model_key, {})
    test_data = mc_data.get("test", {})
    if test_data:
        model_rows.append({
            "Model": model_name,
            "Test AUC": f"{test_data.get('AUC', 0):.4f}",
            "Test Gini": f"{test_data.get('Gini', 0):.4f}",
            "Test KS": f"{test_data.get('KS', 0):.4f}",
            "Features": mc_data.get("n_features", "—"),
        })

if model_rows:
    st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# RAG THRESHOLDS REFERENCE
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
with st.expander("RAG Threshold Reference"):
    st.markdown("""
    | Metric | Green | Amber | Red |
    |--------|-------|-------|-----|
    | Gini (Scorecard) | ≥ 42% | 35-42% | < 35% |
    | Gini (ML Ensemble) | ≥ 46% | 40-46% | < 40% |
    | PSI | < 0.10 | 0.10-0.25 | > 0.25 |
    | CSI (per feature) | < 0.10 | 0.10-0.25 | > 0.25 |
    | VDI | < 0.10 | 0.10-0.25 | > 0.25 |
    | Train-Test AUC Gap | < 0.03 | 0.03-0.05 | > 0.05 |
    | Hosmer-Lemeshow p | > 0.05 | 0.01-0.05 | < 0.01 |
    """)
