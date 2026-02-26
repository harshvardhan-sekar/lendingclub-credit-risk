"""
LendingClub Credit Risk Analytics Platform
==========================================
7-page Streamlit app for institutional-grade credit risk analysis.

Run: streamlit run streamlit_app/app.py
"""

import streamlit as st
from utils.styles import inject_custom_css, sidebar_disclaimer

# ── Page Config ──
st.set_page_config(
    page_title="LendingClub Credit Risk Analytics",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

# ── Sidebar ──
st.sidebar.title("Credit Risk Analytics")
st.sidebar.markdown("---")
try:
    from utils.data_loader import load_strategy_analysis as _load_sa
    _sa = _load_sa()
    _total_loans = _sa["n_loans"].sum()
    _total_bal = _sa["total_balance"].sum()
    st.sidebar.markdown(f"""
**LendingClub Portfolio**
Consumer unsecured personal loans (2007-2018)
~{_total_loans / 1e6:.1f}M loans, ~${_total_bal / 1e9:.0f}B total funded
""")
except Exception:
    st.sidebar.markdown("""
**LendingClub Portfolio**
Consumer unsecured personal loans (2007-2018)
""")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Data Note:** Flow rates derived from synthetic monthly panel "
    "reconstruction. PD, LGD, and EAD models use real observed data. "
    "Production implementation would use observed monthly payment data."
)
sidebar_disclaimer()
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit | [GitHub Repo](https://github.com/harshvardhan-sekar/lendingclub-credit-risk)")

# ── Main Landing Page ──
st.title("LendingClub Credit Risk Analytics Platform")
st.markdown("""
### Institutional-Grade CECL & Stress Testing Framework

This platform provides a comprehensive credit risk analytics suite built on LendingClub's
consumer loan portfolio. It mirrors the analytical framework used at institutional lenders
for CECL compliance, stress testing, and portfolio management.

---

**Navigate using the sidebar** to explore:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Analytics Pages:**
    - **Portfolio Overview** — Executive KPIs, grade distribution, default rates
    - **Roll-Rate Analysis** — DPD bucket transitions, receivables tracker
    - **Vintage Performance** — Cumulative default curves by cohort
    - **ECL Forecasting** — Dual-mode projector with interactive assumptions
    """)

with col2:
    st.markdown("""
    **Advanced Pages:**
    - **Scenario Analysis** — FEG framework, macro stress, sensitivity tornado
    - **Model Monitoring** — RAG dashboard with Gini/PSI/CSI/VDI tracking
    - **AI Analyst** — Claude-powered portfolio Q&A (coming soon)
    """)

st.markdown("---")

# Quick KPIs on landing page
st.subheader("Quick Portfolio Snapshot")

try:
    from utils.data_loader import load_strategy_analysis, load_ecl_postfeg, load_sensitivity_results

    strategy = load_strategy_analysis()
    ecl_pf = load_ecl_postfeg()
    sens = load_sensitivity_results()
    ps = sens.get("portfolio_summary", {})

    total_balance = strategy["total_balance"].sum()
    avg_default = (strategy["default_rate"] * strategy["n_loans"]).sum() / strategy["n_loans"].sum()
    total_loans = strategy["n_loans"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Portfolio", f"${total_balance / 1e9:.2f}B")
    c2.metric("Total Loans", f"{total_loans:,.0f}")
    c3.metric("Avg Default Rate", f"{avg_default:.1%}")
    c4.metric("Post-FEG ECL", f"${ps.get('postfeg_ecl', 0) / 1e9:.2f}B")
    c5.metric("ALLL Ratio", f"{ps.get('postfeg_alll', 0):.1%}")

    feg_ok = ps.get("feg_ordering_correct", False)
    if feg_ok:
        st.success("FEG Ordering Validated: Pre-FEG (Baseline) < Central (Macro-Adjusted) < Post-FEG (Weighted)")
    else:
        st.warning("FEG Ordering: Check scenario results")

except Exception as e:
    st.info(f"Load data files to see portfolio snapshot. ({e})")
