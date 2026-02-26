"""Page 7: AI Analyst — Claude-powered portfolio Q&A with full V6 context.

V6 Roadmap: PortfolioAnalystBot with:
- Claude API integration (anthropic library)
- Portfolio context pre-loaded from data files
- File upload for ad-hoc analysis
- Conversation history within session
- V6 system prompt with dataset metadata, data limitations, key concepts
"""

import os
import streamlit as st
import json
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.styles import inject_custom_css, sidebar_disclaimer

st.set_page_config(page_title="AI Analyst", layout="wide")
inject_custom_css()
sidebar_disclaimer()

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = APP_DIR.parent
RESULTS = PROJECT_DIR / "data" / "results"

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONTEXT LOADER
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data
def load_portfolio_context():
    """Load portfolio data for AI analyst context."""
    context = {}

    # Strategy analysis (grade-level summary)
    try:
        sa = pd.read_csv(RESULTS / "strategy_analysis.csv")
        context["portfolio_summary"] = {
            "total_loans": int(sa["n_loans"].sum()),
            "total_balance": float(sa["total_balance"].sum()),
            "grades": sa.to_dict(orient="records"),
        }
    except Exception:
        context["portfolio_summary"] = {"note": "Data not loaded"}

    # ECL by scenario
    try:
        ecl = pd.read_csv(RESULTS / "ecl_by_scenario.csv")
        context["ecl_by_scenario"] = ecl.groupby("scenario").agg(
            total_ecl=("total_ecl", "sum"),
            total_ead=("total_ead", "sum"),
        ).to_dict(orient="index")
    except Exception:
        context["ecl_by_scenario"] = {}

    # Sensitivity results
    try:
        with open(RESULTS / "sensitivity_results.json") as f:
            sens = json.load(f)
        context["sensitivity"] = sens.get("portfolio_summary", {})
    except Exception:
        context["sensitivity"] = {}

    # Flow rates
    try:
        fr = pd.read_csv(RESULTS / "flow_rates_by_scenario.csv")
        baseline = fr[fr["scenario"] == "baseline"]
        context["flow_rates_baseline"] = baseline.to_dict(orient="records")[:7]
    except Exception:
        context["flow_rates_baseline"] = []

    # Flow-through rate
    try:
        ftr = pd.read_csv(RESULTS / "07_ecl_flow_rates/flow_through_rate.csv")
        avg_ftr = ftr.groupby("grade")["ftr"].mean().to_dict()
        context["avg_ftr_by_grade"] = avg_ftr
    except Exception:
        context["avg_ftr_by_grade"] = {}

    # Model metrics
    try:
        with open(RESULTS / "08_validation" / "pd_scorecard_metrics.json") as f:
            context["model_metrics"] = json.load(f)
    except Exception:
        context["model_metrics"] = {}

    # Model comparison
    try:
        with open(RESULTS / "08_validation" / "model_comparison.json") as f:
            mc = json.load(f)
        context["model_comparison_summary"] = {
            k: v.get("test", {}) for k, v in mc.items() if isinstance(v, dict)
        }
    except Exception:
        context["model_comparison_summary"] = {}

    # Macro scenarios
    try:
        with open(RESULTS / "macro_scenarios.json") as f:
            macro = json.load(f)
        context["macro_scenarios"] = {
            k: v for k, v in macro.items() if k != "baseline_levels"
        }
    except Exception:
        context["macro_scenarios"] = {}

    # Multi-factor elasticities
    try:
        with open(RESULTS / "multi_factor_elasticities.json") as f:
            context["elasticities"] = json.load(f)
    except Exception:
        context["elasticities"] = {}

    return context


def build_system_prompt(portfolio_context: dict) -> str:
    """Build the V6 system prompt with portfolio context."""
    return f"""You are an expert Credit Risk Analyst embedded in the LendingClub
Risk Analytics Platform. You have deep expertise in:

- Consumer credit risk modeling (PD, EAD, LGD, ECL)
- CECL/IFRS-9 frameworks and DCF-based loss estimation
- Portfolio monitoring and model validation (Gini, PSI, CSI, VDI)
- Roll-rate analysis and delinquency migration
- Flow-Through Rate concept (Current → GCO cumulative product)
- Vintage analysis and seasoning patterns
- Macroeconomic scenario analysis with FEG framework
- Loss forecasting and reserve estimation
- Dual-mode forecasting (Operational vs. CECL)
- Competing risks (default vs. prepayment)
- Flow-rate-level stress testing (multiplicative adjustments)
- LGD analysis with recovery and collection fee components
- Synthetic monthly panel reconstruction and limitations

LENDINGCLUB DATASET KNOWLEDGE (V6):
- Total usable loans: 2,260,668 (after footer removal, terminal statuses)
- Total columns in raw dataset: 151
- Usable columns after cleaning: ~65-75 features
- Expected default rate: 19.96% (on terminal loans)
- Loan status values (9 unique):
  * Terminal: Charged Off, Default, Fully Paid
  * Non-terminal: Current, In Grace Period, Late (16-30 days), Late (31-120 days)
  * Policy non-conforming: Does not meet the credit policy. Status:Fully Paid/Charged Off
- Available for LGD analysis: 'recoveries', 'collection_recovery_fee'

DATA LIMITATIONS (V6 — CRITICAL KNOWLEDGE):
The LendingClub public dataset provides loan-level terminal outcomes, NOT monthly
payment history. Monthly DPD status is synthetically reconstructed:
- Fully Paid loans: assumed current every month until payoff
- Charged Off loans: back-calculated from last_pymnt_d (30→60→90→120 DPD progression)
- Current/Late at snapshot: mapped from terminal loan_status

This means:
- Flow rates are FORWARD-ONLY (Current → 30+ → ... → GCO)
- Curing is UNOBSERVABLE (we cannot see delinquent → current transitions)
- Dollar balances in delinquent buckets are APPROXIMATE (scheduled, not actual)
- PD, LGD, and EAD models use REAL observed data (not synthetic)

When answering questions about flow rates or receivables, always note that these
are derived from synthetic reconstruction. When discussing ECL, note that the
framework is production-ready but dollar amounts are approximate.

KEY CONCEPTS YOU SHOULD KNOW:

1. FLOW-THROUGH RATE:
   - Cumulative product of all flow rates from Current to GCO
   - Early warning indicator if trending up
   - Formula: FR_30 × FR_60 × FR_90 × ... × FR_GCO
   - Example: 0.028 × 0.382 × 0.701 × 0.85 × 0.90 × 0.92 × 0.95 = 0.468%

2. FEG FRAMEWORK (Three Scenarios):
   - Pre-FEG: Pure model output, no macro overlay
   - Central (FEG): Baseline macro scenario applied to flow rates
   - Post-FEG: Weighted average across all scenarios + qualitative adjustment

3. DUAL-MODE FORECASTING:
   - Operational Mode: 6-month rolling average flow rates extended flat
   - CECL Mode: R&S period (macro-adjusted) → Reversion → Long-run historical average

4. FLOW-RATE-LEVEL STRESS:
   - Stress multipliers applied to individual flow rates, not final ECL
   - Example: 15% stress on each rate → ~75% increase in flow-through (compounding)
   - Preserves non-linear delinquency dynamics

5. LGD FORMULAS:
   - Primary: LGD = 1 - ((recoveries - collection_recovery_fee) / EAD)
   - Cross-check: LGD_simple = 1 - (total_rec_prncp / EAD)
   - Both columns 100% populated in dataset

6. COMPETING RISKS:
   - Portfolio loses balance via default (GCO) and prepayment (liquidation factor)
   - Both must be modeled for accurate ECL projection

7. SYNTHETIC PANEL CONSTRUCTION:
   - Monthly DPD status reconstructed from terminal outcomes + amortization
   - Performing loans: assumed Current until payoff
   - Defaulted loans: back-calculated from last_pymnt_d
   - Cannot observe curing events (delinquent → current)
   - Framework is production-ready; only the input data granularity differs

INTERVIEW FRAMING:
If asked about data limitations, explain: "The public LendingClub dataset provides
origination features and terminal outcomes. I reconstructed approximate monthly DPD
status to demonstrate the flow rate framework. The PD, LGD, and EAD models use real
data. In production with monthly payment tapes, the same framework would incorporate
two-way transitions and curing rates."

CURRENT PORTFOLIO DATA:
{json.dumps(portfolio_context, indent=2, default=str)}

When answering questions:
- Reference specific numbers from the portfolio data
- Provide actionable insights, not just observations
- When asked about trends, explain the likely drivers
- When asked for recommendations, frame them as strategy decisions
- Use proper credit risk terminology
- Explain the impact of flow-through rate changes
- Distinguish between Pre-FEG, Central, and Post-FEG results
- Explain dual-mode forecasting choice tradeoffs
- Explain how recoveries and collection fees affect LGD
- Be transparent about synthetic data vs. observed data
- If asked to generate a report, format it professionally

If the user uploads a file, analyze it in the context of the existing
portfolio data. Compare metrics, identify anomalies, and provide insights.
"""


def parse_uploaded_file(file) -> str:
    """Parse uploaded CSV/Excel/JSON file into a string summary."""
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        elif file.name.endswith(".json"):
            return json.dumps(json.load(file), indent=2)
        else:
            return file.read().decode("utf-8")

        summary = f"Shape: {df.shape}\n"
        summary += f"Columns: {list(df.columns)}\n"
        summary += f"Data types:\n{df.dtypes.to_string()}\n\n"
        summary += f"Summary statistics:\n{df.describe().to_string()}\n\n"
        summary += f"First 20 rows:\n{df.head(20).to_string()}"
        return summary
    except Exception as e:
        return f"Error parsing file: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

st.title("AI Analyst")
st.markdown("### Claude-Powered Portfolio Q&A")
st.markdown(
    "Ask natural language questions about the credit risk portfolio, "
    "ECL projections, model performance, and scenario analysis."
)

# ── API Key Check ──
api_key = None
try:
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
except Exception:
    pass

if not api_key:
    api_key = st.sidebar.text_input(
        "Anthropic API Key",
        type="password",
        help="Enter your Anthropic API key to enable the AI Analyst. "
             "You can also set it in .streamlit/secrets.toml.",
    )

# ── Sidebar: Capabilities & File Upload ──
st.sidebar.markdown("---")
st.sidebar.markdown("### AI Analyst Capabilities")
st.sidebar.markdown("""
- Portfolio Q&A with live data
- Executive summary generation
- Scenario narrative explanations
- Flow-through rate analysis
- What-if stress testing
- File upload for ad-hoc analysis
""")

uploaded_file = st.sidebar.file_uploader(
    "Upload file for analysis",
    type=["csv", "xlsx", "xls", "json", "txt"],
    help="Upload a data file and ask the AI to analyze it in portfolio context.",
)

# ── Load Portfolio Context ──
portfolio_context = load_portfolio_context()

# ── Quick Stats ──
ps = portfolio_context.get("portfolio_summary", {})
total_loans = ps.get("total_loans", 0)
total_bal = ps.get("total_balance", 0)

if total_loans > 0:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Portfolio Loans", f"{total_loans:,.0f}")
    with c2:
        st.metric("Total Balance", f"${total_bal / 1e9:.2f}B")
    with c3:
        sens = portfolio_context.get("sensitivity", {})
        ecl_val = sens.get("postfeg_ecl", 0)
        st.metric("Post-FEG ECL", f"${ecl_val / 1e9:.2f}B" if ecl_val else "N/A")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CHAT INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

# Initialize session state for conversation
if "ai_messages" not in st.session_state:
    st.session_state.ai_messages = []

# Display conversation history
for msg in st.session_state.ai_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Sample Questions ──
if len(st.session_state.ai_messages) == 0:
    st.markdown("**Try asking:**")
    sample_qs = [
        "What is our current flow-through rate and how does it compare across grades?",
        "Compare Pre-FEG vs Post-FEG ECL estimates and explain the difference.",
        "Write a quarterly loss forecast memo for the CFO.",
        "What happens to ECL if unemployment rises by 2%?",
        "Summarize the model monitoring RAG status — are any metrics flagged?",
    ]
    cols = st.columns(len(sample_qs))
    for i, q in enumerate(sample_qs):
        with cols[i]:
            if st.button(q[:40] + "...", key=f"sample_{i}", use_container_width=True):
                st.session_state._pending_question = q
                st.rerun()

# Check for pending sample question
pending = st.session_state.pop("_pending_question", None)

# Chat input
user_input = st.chat_input("Ask a question about the portfolio...")

# Use pending question if no direct input
if pending and not user_input:
    user_input = pending

if user_input:
    # Add file context if uploaded
    if uploaded_file is not None:
        file_content = parse_uploaded_file(uploaded_file)
        full_message = (
            f"[User uploaded file: {uploaded_file.name}]\n\n"
            f"File contents:\n{file_content}\n\n"
            f"User message: {user_input}"
        )
    else:
        full_message = user_input

    # Display user message
    st.session_state.ai_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)

            # Build messages for API
            api_messages = []
            for msg in st.session_state.ai_messages:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"] if msg["role"] == "assistant" else full_message
                    if msg == st.session_state.ai_messages[-1] else msg["content"],
                })

            with st.chat_message("assistant"):
                with st.spinner("Analyzing portfolio data..."):
                    response = client.messages.create(
                        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                        max_tokens=4096,
                        system=build_system_prompt(portfolio_context),
                        messages=api_messages,
                    )
                    assistant_text = response.content[0].text
                    st.markdown(assistant_text)

            st.session_state.ai_messages.append(
                {"role": "assistant", "content": assistant_text}
            )

        except ImportError:
            with st.chat_message("assistant"):
                st.error(
                    "The `anthropic` library is not installed. "
                    "Install it with: `pip install anthropic`"
                )
        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"API Error: {e}")
    else:
        # No API key — provide a helpful static response
        with st.chat_message("assistant"):
            st.warning(
                "No API key configured. To enable the AI Analyst:\n\n"
                "1. Enter your Anthropic API key in the sidebar, or\n"
                "2. Add it to `.streamlit/secrets.toml`:\n"
                "```\nANTHROPIC_API_KEY = \"your-key-here\"\n```\n\n"
                "The AI Analyst uses Claude to answer portfolio questions "
                "with pre-loaded context from all data files."
            )
        st.session_state.ai_messages.append({
            "role": "assistant",
            "content": "Please configure an Anthropic API key to use the AI Analyst.",
        })

# ── Clear Conversation ──
if len(st.session_state.ai_messages) > 0:
    if st.sidebar.button("Clear Conversation"):
        st.session_state.ai_messages = []
        st.rerun()

# ── Context Info (Expander) ──
with st.expander("Portfolio Context Loaded"):
    st.json({
        "data_sources": list(portfolio_context.keys()),
        "total_loans": ps.get("total_loans", "N/A"),
        "total_balance": ps.get("total_balance", "N/A"),
        "ecl_scenarios": list(portfolio_context.get("ecl_by_scenario", {}).keys()),
        "model_metrics_available": bool(portfolio_context.get("model_metrics")),
        "macro_scenarios": list(portfolio_context.get("macro_scenarios", {}).keys()),
        "ftr_grades": list(portfolio_context.get("avg_ftr_by_grade", {}).keys()),
    })
