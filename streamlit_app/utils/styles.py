"""Consistent styling, colors, and KPI card components for the Streamlit app."""

import streamlit as st

# ── Grade Colors ──
GRADE_COLORS = {
    "A": "#2ecc71", "B": "#27ae60", "C": "#f39c12",
    "D": "#e67e22", "E": "#e74c3c", "F": "#c0392b", "G": "#8e44ad",
}

GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]

# ── Scenario Colors ──
SCENARIO_COLORS = {
    "central": "#3498db", "mild": "#f39c12", "stress": "#e74c3c",
    "baseline": "#95a5a6", "pre_feg": "#7f8c8d", "post_feg": "#2c3e50",
}

# ── RAG Colors ──
RAG_COLORS = {"green": "#27ae60", "amber": "#f39c12", "red": "#e74c3c"}

# ── DPD Bucket Colors ──
DPD_COLORS = {
    "Current": "#2ecc71", "30+": "#f1c40f", "60+": "#f39c12",
    "90+": "#e67e22", "120+": "#e74c3c", "150+": "#c0392b",
    "180+": "#8e44ad", "GCO": "#2c3e50",
}


def inject_custom_css():
    """Inject custom CSS for consistent styling."""
    st.markdown("""
    <style>
        .kpi-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #1f77b4;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .kpi-value {
            font-size: 28px;
            font-weight: 700;
            color: #2c3e50;
            margin: 4px 0;
        }
        .kpi-label {
            font-size: 13px;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .kpi-delta {
            font-size: 12px;
            margin-top: 4px;
        }
        .kpi-delta.positive { color: #27ae60; }
        .kpi-delta.negative { color: #e74c3c; }
        .disclaimer-box {
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 12px;
            font-size: 12px;
            color: #856404;
            margin-bottom: 16px;
        }
        .section-header {
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 8px;
            margin-bottom: 16px;
        }
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str = None, delta_positive: bool = True):
    """Render a styled KPI metric card."""
    delta_html = ""
    if delta:
        cls = "positive" if delta_positive else "negative"
        delta_html = f'<div class="kpi-delta {cls}">{delta}</div>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def sidebar_disclaimer():
    """Add synthetic panel disclaimer to sidebar."""
    st.sidebar.markdown("""
    <div class="disclaimer-box">
        <strong>Data Limitation:</strong> Flow rates derived from synthetic monthly panel
        reconstruction using terminal loan outcomes. Production implementation would use
        observed monthly payment data with actual transition observations.
    </div>
    """, unsafe_allow_html=True)


def format_currency(value: float) -> str:
    """Format large dollar amounts."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"${value / 1e3:.0f}K"
    return f"${value:,.0f}"


def format_pct(value: float, decimals: int = 2) -> str:
    """Format percentage."""
    return f"{value * 100:.{decimals}f}%" if value < 1 else f"{value:.{decimals}f}%"
