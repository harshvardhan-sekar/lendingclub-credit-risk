# LendingClub Risk Analytics Platform — Streamlit App Technical Architecture V5

## Changes from V4

This version incorporates data-driven corrections and enhancements based on analysis of the LendingClub dataset:

1. **LGD Formula Updates:** Updated all Loss Given Default (LGD) formulas to explicitly include `recoveries` and `collection_recovery_fee` columns, with net recovery = recoveries - collection_recovery_fee. Both columns are 100% populated in the dataset.

2. **Dataset Configuration Metadata:** Added comprehensive `DATASET_CONFIG` to constants.py documenting actual data characteristics: 2,260,668 usable rows (after footer removal), 19.96% default rate on terminal loans, 9 unique loan_status values with documented terminal/non-terminal/drop categories, and all 14 empty sec_app_* columns.

3. **Data Preprocessing Pipeline:** Formalized the exact data cleaning sequence in a dedicated section showing load, drop, parse, filter, target creation, merge, and save steps.

4. **Benchmark Population Validation:** Added new tab in Model Monitoring (Page 6) for external benchmark validation using benchmark_population_2014.csv (200K records) with PSI and calibration checks.

5. **AI Analyst Context Enhancement:** Updated chatbot system prompt with specific dataset knowledge: 2,260,668 loans, ~65-75 usable columns after cleaning, 9 unique loan_status values, and benchmark_population_2014.csv availability.

6. **New Originations Visibility (V5):** New Originations input is now contextually visible — hidden in CECL mode (hardcoded to $0) because CECL reserves are for the existing portfolio; new loans get Day 1 CECL at origination.

7. **FEG/Stress Applies to Both Modes (V5):** Documented explicitly that FEG toggle and flow-rate-level stress apply to both Operational and CECL modes; the dimensions are orthogonal (mode determines baseline computation, FEG determines stress application).

8. **Recovery Rate Connection to LGD Model (V5):** Updated recovery_rate assumption to reference its connection to the LGD model's portfolio-level output (LGD ≈ 83% → recovery = 17%, confirmed from LendingClub 10-K).

9. **Generic Framing (V5):** Replaced all HSBC-specific references with generic institutional framing to create a portable, interview-ready tool.

---

## Overview

This document specifies the complete technical architecture for the LendingClub Risk Analytics Platform — a Streamlit-based portfolio management and loss forecasting tool inspired by PyCraft. The tool has two main components:

1. **Core Forecasting Engine:** Takes receivables data as input and projects forward 10 years of portfolio balances, GCO, NCO, recoveries, flow rates, and ECL — exactly replicating what PyCraft does. Now with dual-mode forecasting (operational vs. CECL) and three-scenario FEG framework.
2. **AI Analysis Layer:** An embedded Claude-powered chatbot that can analyze uploaded files, answer portfolio questions, run on-the-fly sensitivity analysis, and generate executive reports.

---

## Architecture Overview

```
app/
├── streamlit_app.py                 # Main entry point, navigation, global config
├── pages/
│   ├── 01_portfolio_overview.py     # Dashboard: KPIs, composition, default rates, flow-through rate
│   ├── 02_roll_rate_analysis.py     # Flow rates, receivables tracker, flow-through rates
│   ├── 03_vintage_performance.py    # Cumulative default curves by vintage × MOB
│   ├── 04_ecl_forecasting.py        # PyCraft core: dual-mode forecasting, FEG toggle, assumption upload/export
│   ├── 05_scenario_analysis.py      # Macro scenarios, stress at flow rate level, sensitivity
│   ├── 06_model_monitoring.py       # Gini/PSI/CSI/VDI tracking with RAG + benchmark population validation
│   └── 07_ai_analyst.py             # Claude-powered chatbot
├── components/
│   ├── charts.py                    # Reusable Plotly chart components
│   ├── tables.py                    # Formatted DataFrame display components
│   ├── metrics.py                   # KPI card components (including flow-through rate)
│   ├── chatbot.py                   # AI chatbot interface and API integration
│   └── file_handler.py              # File upload parsing (CSV, Excel, JSON)
├── engine/
│   ├── flow_rate_engine.py          # Flow rate computation, receivables tracker, projection
│   ├── ecl_projector.py             # Forward-looking ECL projection (redesigned dual-mode)
│   ├── liquidation.py               # Portfolio liquidation/runoff modeling
│   ├── macro_overlay.py             # FRED integration, flow-rate-level stress, scenario weighting
│   ├── vintage_analyzer.py          # Vintage curve computation and comparison
│   ├── prepayment.py                # Prepayment model for competing risks
│   └── flow_through_calculator.py   # Flow-through rate computation across DPD buckets
├── utils/
│   ├── data_loader.py               # Load and cache data for Streamlit
│   ├── session_state.py             # Manage Streamlit session state
│   ├── constants.py                 # App-wide constants, color palettes, thresholds, dataset config
│   └── formatters.py                # Number formatting, currency, percentages
└── assets/
    ├── logo.png                     # App logo
    └── custom.css                   # Custom styling
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA INPUTS                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ LendingClub  │  │  User Upload │  │    FRED API      │  │
│  │   Dataset    │  │ (Receivables │  │ (Macro Forecasts)│  │
│  │  (Processed) │  │    File)     │  │                  │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                    │            │
└─────────┼─────────────────┼────────────────────┼────────────┘
          │                 │                    │
          ▼                 ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    PROCESSING ENGINE                        │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Roll-Rate   │  │     ECL      │  │     Macro        │  │
│  │   Engine     │──│  Projector   │──│    Overlay        │  │
│  │              │  │  (Dual-Mode) │  │  (Flow-Rate Stress)  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Vintage     │  │ Liquidation  │  │   Flow-Through   │  │
│  │  Analyzer    │  │   Engine     │  │  Rate Calculator │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Prepayment   │  │ Validation   │  │  Competing Risks │  │
│  │   Model      │  │   Engine     │  │  (Default vs PP) │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                      │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │Portfolio │ │Roll-Rate │ │ Vintage  │ │    ECL       │  │
│  │Dashboard │ │ Analysis │ │  Curves  │ │  Forecaster  │  │
│  │ (+FTR)   │ │ (+FTR)   │ │          │ │ (Dual-Mode)  │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────────┐   │
│  │ Macro    │ │  Model   │ │    AI Analyst            │   │
│  │Scenarios │ │ Monitor  │ │  (Claude Chatbot)        │   │
│  │(FEG, FRT)│ │  (RAG)   │ │                          │   │
│  └──────────┘ └──────────┘ └──────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Preprocessing Pipeline

The Streamlit app loads and processes raw LendingClub data via the following pipeline:

### Step 1: Load Raw Data
- Load CSV from data/raw/LendingClub dataset
- Skip or filter footer rows (33 rows to skip, total usable: 2,260,668 rows)
- Verify expected columns and row count

### Step 2: Drop Immediate-Drop Columns
- **14 empty sec_app_* columns:** `sec_app_fico_range_low`, `sec_app_fico_range_high`, `sec_app_earliest_cr_line`, `sec_app_inq_last_6mths`, `sec_app_mort_acc`, `sec_app_open_acc`, `sec_app_revol_util`, `sec_app_open_act_il`, `sec_app_num_rev_accts`, `sec_app_chargeoff_within_12_mths`, `sec_app_collections_12_mths_ex_med`, `sec_app_mths_since_last_major_derog`, `revol_bal_joint`, `member_id`
- **Non-feature columns:** `id`, `url`, `desc`, `pymnt_plan`, `policy_code`
- **Hardship/Settlement (~15 columns, near-empty):** `hardship_type`, `hardship_reason`, `hardship_status`, `deferral_term`, `hardship_amount`, `hardship_start_date`, `hardship_end_date`, `payment_plan_start_date`, `hardship_length`, `hardship_dpd`, `hardship_loan_status`, `hardship_payoff_balance_amount`, `hardship_last_payment_amount`, `orig_projected_additional_accrued_interest`, `debt_settlement_flag_date`, `settlement_status`, `settlement_date`, `settlement_amount`, `settlement_percentage`, `settlement_term`

### Step 3: Parse Column Values
- **term column:** Strip whitespace and convert to integer (36 or 60 months)
- **emp_length column:** Convert from text to numeric using mapping:
  - '< 1 year' → 0, '1 year' → 1, ..., '10+ years' → 10
- **int_rate and revol_util:** Already float64 — NO % stripping needed (confirmed from V4 profiling; values are e.g. 13.99, 29.7)
- **Date columns (issue_d, earliest_cr_line, etc.):** Parse from 'MMM-YYYY' format to datetime

### Step 4: Filter to Terminal Loan Status
- **Terminal Statuses (retain):** 'Charged Off', 'Default', 'Fully Paid'
- **Non-Terminal (drop):** 'Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)'
- **Policy non-conforming (drop):** 'Does not meet the credit policy. Status:Fully Paid', 'Does not meet the credit policy. Status:Charged Off'
- Result: ~19.96% default rate (Charged Off + Default) on remaining loans

### Step 5: Create Target Variable
- **default:** Binary flag (1 if loan_status in ['Charged Off', 'Default'], else 0)
- Result: 2,260,668 × 1 binary target

### Step 6: Merge FRED Macro Data
- Extract issue_d month and year
- Download matching FRED data (unemployment, GDP growth, HPI) for that month
- Merge on issue_d month

### Step 7: Save Processed Data
- Output: parquet file with ~65-75 usable features, 2,260,668 rows, terminal statuses only

---

## Dataset Configuration & Known Quirks

Add to `utils/constants.py`:

```python
DATASET_CONFIG = {
    'total_usable_rows': 2_260_668,
    'footer_rows_to_skip': 33,
    'expected_default_rate': 0.1996,  # 19.96% after filtering to terminal
    'terminal_statuses': {
        'default': ['Charged Off', 'Default'],
        'non_default': ['Fully Paid'],
        'drop': ['Current', 'In Grace Period', 'Late (16-30 days)',
                 'Late (31-120 days)',
                 'Does not meet the credit policy. Status:Fully Paid',
                 'Does not meet the credit policy. Status:Charged Off']
    },
    'columns_to_drop_immediate': [
        # 14 completely empty sec_app_* columns
        'sec_app_fico_range_low', 'sec_app_fico_range_high',
        'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths',
        'sec_app_mort_acc', 'sec_app_open_acc', 'sec_app_revol_util',
        'sec_app_open_act_il', 'sec_app_num_rev_accts',
        'sec_app_chargeoff_within_12_mths',
        'sec_app_collections_12_mths_ex_med',
        'sec_app_mths_since_last_major_derog',
        'revol_bal_joint', 'member_id',
        # Non-feature columns
        'id', 'url', 'desc', 'pymnt_plan', 'policy_code',
        # Near-empty hardship/settlement (~15 columns)
        'hardship_type', 'hardship_reason', 'hardship_status',
        'deferral_term', 'hardship_amount', 'hardship_start_date',
        'hardship_end_date', 'payment_plan_start_date',
        'hardship_length', 'hardship_dpd', 'hardship_loan_status',
        'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
        'orig_projected_additional_accrued_interest',
        'debt_settlement_flag_date', 'settlement_status',
        'settlement_date', 'settlement_amount',
        'settlement_percentage', 'settlement_term'
    ],
    'date_format': '%b-%Y',  # 'Dec-2015' format
    'term_values': [36, 60],  # After stripping and parsing
    'emp_length_mapping': {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
        '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
        '8 years': 8, '9 years': 9, '10+ years': 10
    },
    'numeric_no_conversion': ['int_rate', 'revol_util'],  # Already float64, no % stripping
    'fred_series': ['UNRATE', 'CSUSHPINSA', 'A191RL1Q225SBEA', 'CPIAUCSL', 'DFF', 'UMCSENT'],
    'fred_series_descriptions': {
        'UNRATE': 'Unemployment rate (monthly)',
        'CSUSHPINSA': 'Case-Shiller Home Price Index (monthly, not seasonally adjusted)',
        'A191RL1Q225SBEA': 'Real GDP growth rate (quarterly — forward-fill within quarter)',
        'CPIAUCSL': 'Consumer Price Index (monthly)',
        'DFF': 'Effective Federal Funds Rate (monthly)',
        'UMCSENT': 'University of Michigan Consumer Sentiment Index (monthly)'
    }
}
```

---

## Page-by-Page Specification

### Page 1: Portfolio Overview Dashboard

**Purpose:** Executive-level view of portfolio health — the first thing a CRO or Head of Credit Strategy would look at.

**New Features (V3):**
- **Flow-Through Rate KPI:** Display cumulative flow rates (Current → GCO) as an early warning signal
- Trend line showing flow-through rate over time
- Cross-check against PD model outputs

**Layout:**
```
┌──────────────────────────────────────────────────────┐
│  LendingClub Risk Analytics Platform                 │
├────────┬────────┬────────┬────────┬─────────────────┤
│ Total  │Default │ ALLL   │ Flow-  │ NCO             │
│Outstand│ Rate   │Ratio   │Through │ Ratio           │
│ $XXB   │ X.X%   │ X.X%   │ X.XX%  │ X.X%            │
│        │        │        │Rate    │                 │
├────────┴────────┴────────┴────────┴─────────────────┤
│                                                      │
│  ┌─────────────────────┐ ┌─────────────────────────┐ │
│  │ Default Rate by     │ │ Portfolio Composition    │ │
│  │ Grade (Bar Chart)   │ │ by Grade (Donut Chart)  │ │
│  └─────────────────────┘ └─────────────────────────┘ │
│                                                      │
│  ┌─────────────────────┐ ┌─────────────────────────┐ │
│  │ Monthly Origination │ │ Flow-Through Rate Trend │ │
│  │ Volume Over Time    │ │ (Line Chart + Alert)    │ │
│  └─────────────────────┘ └─────────────────────────┘ │
│                                                      │
│  ┌─────────────────────┐ ┌─────────────────────────┐ │
│  │ Delinquency Trend   │ │ Geographic Default Rate │ │
│  │ (Stacked Area)      │ │ Heatmap (Choropleth)    │ │
│  └─────────────────────┘ └─────────────────────────┘ │
│                                                      │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Default Rate by Purpose (Horizontal Bar)         │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Key Metrics (KPI Cards):**
- Total Outstanding Balance (sum of funded_amnt for active loans)
- Overall Default Rate (%)
- Weighted Average Interest Rate
- **Flow-Through Rate (NEW):** Current → GCO product = FR₁ × FR₂ × ... × FR_GCO
- ALLL Ratio (ECL / Total Outstanding)
- NCO Ratio (annualized net charge-offs / average outstanding)
- 30+ DPD Rate (early delinquency indicator)

**Filters (Sidebar):**
- Date range (origination period)
- Grade filter (A-G, multi-select)
- Term filter (36 months, 60 months)
- Purpose filter
- State filter

**Charts:**
1. Default rate by grade (A-G) — bar chart, should be monotonically increasing
2. Portfolio composition by grade — donut/pie chart showing balance concentration
3. Monthly origination volume — line chart with volume on left axis, default rate on right axis
4. **Flow-Through Rate Trend (NEW)** — line chart with early warning threshold (e.g., if > 0.5%, flag as amber)
5. Delinquency waterfall — stacked area chart: Current, 30 DPD, 60 DPD, 90+ DPD over time
6. Geographic default rate — choropleth map by state
7. Default rate by purpose — horizontal bar chart

**Implementation Notes:**
- Use `st.cache_data` for all data loading
- Use Plotly for interactive charts
- KPI cards use `st.metric()` with delta indicators (month-over-month change)
- Flow-through rate computation: multiply all sequential flow rates from receivables tracker

---

### Page 2: Roll-Rate Analysis

**Purpose:** Show how balances flow through delinquency stages month-over-month. This is the core deliverable from the loss forecasting team — the receivables tracker.

**New Features (V3):**
- **Flow-Through Rate Row (NEW):** Below individual flow rates, add cumulative diagonal multiplication
  - Flow Through Rate (Current → 30 DPD) = FR₁
  - Flow Through Rate (Current → 60 DPD) = FR₁ × FR₂
  - ... continuing through to GCO

**PyCraft Connection:** This directly replicates the receivables tracker that categorizes portfolio balances into delinquency buckets (Current, 30+, 60+, 90+, 120+, 150+, 180+ DPD) with GCO, recoveries, and NCO. The flow rates computed from these buckets are what PyCraft used to project forward.

**Layout:**
```
┌──────────────────────────────────────────────────────┐
│  Roll-Rate Analysis                                  │
├──────────────────────────────────────────────────────┤
│  Filters: [Grade ▾] [Vintage Year ▾] [Period ▾]     │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────────────────────────────────────┐ │
│  │ RECEIVABLES TRACKER TABLE (Institutional Format) │
│  │                                                  │ │
│  │         Jan-18   Feb-18   Mar-18   ...           │ │
│  │ ─── Dollar Receivables ───                       │ │
│  │ Current $XXM     $XXM     $XXM                   │ │
│  │ 30 DPD  $XXM     $XXM     $XXM                   │ │
│  │ 60 DPD  $XXM     $XXM     $XXM                   │ │
│  │ 90 DPD  $XXM     $XXM     $XXM                   │ │
│  │ 120 DPD $XXM     $XXM     $XXM                   │ │
│  │ 150 DPD $XXM     $XXM     $XXM                   │ │
│  │ 180 DPD $XXM     $XXM     $XXM                   │ │
│  │ Acct Ct  XXX      XXX      XXX                   │ │
│  │ GCO     $XXM     $XXM     $XXM                   │ │
│  │ Recovery $XXM    $XXM     $XXM                   │ │
│  │ NCO     $XXM     $XXM     $XXM                   │ │
│  │ ─── Flow Rates ───                              │ │
│  │ 30+ FR   X.X%    X.X%     X.X%                   │ │
│  │ 60+ FR   X.X%    X.X%     X.X%                   │ │
│  │ 90+ FR   X.X%    X.X%     X.X%                   │ │
│  │ 120+ FR  X.X%    X.X%     X.X%                   │ │
│  │ 150+ FR  X.X%    X.X%     X.X%                   │ │
│  │ 180+ FR  X.X%    X.X%     X.X%                   │ │
│  │ GCO FR   X.X%    X.X%     X.X%                   │ │
│  │ ─── Flow-Through Rates (NEW) ───                │ │
│  │ FTR 30   X.XX%   X.XX%    X.XX%                 │ │
│  │ FTR 60   X.XX%   X.XX%    X.XX%                 │ │
│  │ FTR 90   X.XX%   X.XX%    X.XX%                 │ │
│  │ FTR 120  X.XX%   X.XX%    X.XX%                 │ │
│  │ FTR 150  X.XX%   X.XX%    X.XX%                 │ │
│  │ FTR 180  X.XX%   X.XX%    X.XX%                 │ │
│  │ FTR GCO  X.XX%   X.XX%    X.XX%                 │ │
│  └──────────────────────────────────────────────────┘ │
│                                                      │
│  ┌─────────────────────┐ ┌─────────────────────────┐ │
│  │ Flow Rate Trends    │ │ Delinquency Flow        │ │
│  │ Over Time           │ │ (Sankey Diagram)        │ │
│  │ (Line Chart)        │ │                         │ │
│  └─────────────────────┘ └─────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Flow Rate Computation (engine/flow_rate_engine.py):**

**Important distinction:** The industry often conflates "roll rates" and "flow rates." True roll rates track individual accounts across buckets month-over-month (requires account-level longitudinal data, produces an N×N transition matrix). Flow rates simply compare consecutive bucket balances in consecutive months as simple ratios. The OCC's Comptroller's Handbook acknowledges that most banks use the flow rate approach operationally: "for ease of calculation, roll rate analysis assumes all dollars at the end of a period flow from the prior period bucket." This is what institutions used in the receivables tracker.

```python
def build_receivables_tracker(df, period_col='month', status_col='dpd_bucket',
                               balance_col='outstanding_balance', grade=None):
    """
    Build the receivables tracker in institutional format.

    Aggregates account-level data into monthly dollar balances by DPD bucket.
    Structure: rows = DPD buckets + account counts + GCO + Recovery + NCO,
               columns = monthly periods (YYYYMM)

    DPD buckets: Current, 30 DPD, 60 DPD, 90 DPD, 120 DPD, 150 DPD, 180+ DPD

    Returns:
        pd.DataFrame: Receivables tracker with dollar balances by bucket by month
    """

def compute_flow_rates(receivables_tracker):
    """
    Compute flow rates as simple bucket-to-bucket ratios.

    Flow rates are computed from the receivables tracker (aggregate balances),
    NOT from account-level tracking. This is the standard operational approach
    used at most large banks.

    Formula for each month:
        30+ Flow Rate  = 30 DPD balance (this month) / Current balance (last month)
        60+ Flow Rate  = 60 DPD balance (this month) / 30 DPD balance (last month)
        90+ Flow Rate  = 90 DPD balance (this month) / 60 DPD balance (last month)
        120+ Flow Rate = 120 DPD balance (this month) / 90 DPD balance (last month)
        150+ Flow Rate = 150 DPD balance (this month) / 120 DPD balance (last month)
        180+ Flow Rate = 180 DPD balance (this month) / 150 DPD balance (last month)
        GCO Flow Rate  = GCO (this month) / 180 DPD balance (last month)

    Returns:
        pd.DataFrame: Flow rates by bucket by month (same column structure
                      as receivables tracker, displayed below dollar rows)
    """

def compute_flow_rate_trends(receivables_tracker, window=6):
    """
    Compute rolling average flow rates to track stability over time.

    Args:
        receivables_tracker: output from build_receivables_tracker
        window: rolling average window in months (default 6)

    Returns:
        pd.DataFrame: smoothed flow rates over time for trend analysis
    """

def project_balances(current_balances, flow_rates, n_months,
                     liquidation_factor=0.0, new_origination=0.0):
    """
    Project balances forward using flow rates.
    This is the core PyCraft projection logic.

    For each projected month t:
        Projected_30DPD(t) = flow_rate_30 × Current_balance(t-1)
        Projected_60DPD(t) = flow_rate_60 × 30DPD_balance(t-1)
        ...
        Projected_GCO(t)   = flow_rate_gco × 180DPD_balance(t-1)

    Then apply liquidation factor and add new originations to Current.

    Args:
        current_balances: dict of {bucket: balance} — current receivables snapshot
        flow_rates: dict of {bucket: rate} — average or scenario-adjusted flow rates
        n_months: number of months to project (120 for 10 years)
        liquidation_factor: monthly portfolio runoff rate (e.g., 0.02 = 2%/month)
        new_origination: monthly new loan origination amount

    Returns:
        pd.DataFrame: projected balances by bucket × month
    """
```

**Receivables Tracker Format (institutional standard):**

The tracker should be a downloadable Excel file with this structure:
- **Top section (Dollar Receivables):** DPD buckets (Current, 30 DPD, 60 DPD, 90 DPD, 120 DPD, 150 DPD, 180+ DPD) + Account counts + GCO + Recovery + NCO
- **Flow Rates section:** Flow rate for each bucket transition, computed as simple ratios from the dollar receivables above
- **Flow-Through Rates section (NEW):** Cumulative products of flow rates, showing end-to-end migration rates
- Columns: Monthly periods (YYYYMM)
- Filterable by: Grade, Vintage, Portfolio segment

**Charts:**
1. Flow rate trend lines over time (e.g., 60+ flow rate month-over-month — key early warning indicator)
2. Receivables tracker table (interactive, filterable)
3. Sankey diagram showing balance flow through delinquency stages
4. Delinquency bucket balances over time (stacked bar)

---

### Page 3: Vintage Performance

**Purpose:** Track cumulative default rates by origination vintage over time (MOB). This directly replicates the Sherwood PD curves from my prior role.

**Prior Role Connection:** The Sherwood Lifetime Loss model used a Product Type × MOB grid to compute marginal and cumulative PD. Here we use Grade × MOB, which is the equivalent for unsecured personal loans.

**Layout:**
```
┌──────────────────────────────────────────────────────┐
│  Vintage Performance Analysis                        │
├──────────────────────────────────────────────────────┤
│  Filters: [Grades ▾] [Vintages ▾] [Metric ▾]        │
│  Metric: ○ Cumulative Default Rate  ○ Marginal PD   │
│          ○ Cumulative Loss Rate                      │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────────────────────────────────────┐ │
│  │ VINTAGE CURVES (Main Chart)                      │ │
│  │                                                  │ │
│  │  Y: Cumulative Default Rate (%)                  │ │
│  │  X: Month on Book (MOB)                          │ │
│  │                                                  │ │
│  │  Each line = one vintage year                    │ │
│  │  Color-coded: 2007(red) 2008(orange) ... 2018    │ │
│  │                                                  │ │
│  │  Smoothed with 6-month rolling average           │ │
│  └──────────────────────────────────────────────────┘ │
│                                                      │
│  ┌─────────────────────┐ ┌─────────────────────────┐ │
│  │ Marginal PD Curves  │ │ Vintage Comparison      │ │
│  │ (by Grade)          │ │ Table                   │ │
│  │                     │ │ (MOB 12/24/36 defaults) │ │
│  └─────────────────────┘ └─────────────────────────┘ │
│                                                      │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Seasoning Pattern (Default rate by MOB,          │ │
│  │ averaged across vintages — shows typical         │ │
│  │ lifecycle curve for LC personal loans)            │ │
│  └──────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

**Vintage Analysis Engine (engine/vintage_analyzer.py):**

```python
def compute_vintage_curves(df, vintage_col='issue_year', mob_col='mob',
                           default_col='default', balance_col='outstanding'):
    """
    Compute cumulative default rate curves by vintage.

    For each vintage cohort:
    1. Track the number (or balance) of loans at each MOB
    2. Track cumulative defaults up to each MOB
    3. Cumulative Default Rate = Cumulative Defaults / Original Cohort Size

    Returns:
        pd.DataFrame: vintage × MOB matrix of cumulative default rates
    """

def compute_marginal_pd(df, vintage_col='issue_year', mob_col='mob',
                         default_col='default', prepaid_col='prepaid'):
    """
    Compute marginal PD at each MOB (mirrors Sherwood methodology).

    Marginal PD = Defaults at MOB_n / (Active Accounts at MOB_n - Prepaid at MOB_n)

    Apply 6-month rolling average smoothing.

    Returns:
        pd.DataFrame: vintage × MOB matrix of marginal PDs
    """

def identify_underperforming_vintages(vintage_curves, benchmark_mob=24):
    """
    Compare each vintage's cumulative default rate at benchmark_mob
    against the portfolio average. Flag vintages that are >20% worse.

    Returns root cause analysis hints (origination volume, grade mix,
    macro conditions at origination).
    """
```

**Key Features:**
- Toggle between cumulative default rate, marginal PD, and cumulative loss rate
- Filter by grade to see vintage curves within each risk segment
- 6-month rolling average smoothing (as used in Sherwood)
- Vintage comparison table: default rate at MOB 6, 12, 18, 24, 36 for each vintage
- Seasoning pattern: average default rate by MOB across all vintages — shows the typical lifecycle curve

---

### Page 4: ECL Forecasting Engine (The PyCraft Core) — MAJOR OVERHAUL

**Purpose:** This is the heart of the tool. It takes a receivables snapshot and projects forward 10 years of receivables, GCO, NCO, recoveries, flow rates, and ECL — exactly what PyCraft does. Now features dual-mode forecasting (Operational vs. CECL) and FEG three-scenario framework.

**PyCraft Connection:** PyCraft was a Django-based tool used during Annual Operations Planning / Financial Resource Planning. It took receivables files as input, applied liquidation factors and blended ECL rates, and output 10-year forecasts of portfolio balances and losses.

**V3 Major Changes:**
1. **Dual-Mode Toggle:** Operational (PyCraft-style) vs. CECL (ASC 326 compliant)
2. **FEG Three-Scenario Toggle:** Pre-FEG / Central (FEG) / Post-FEG
3. **Assumption Upload/Export:** Excel-based workflow for audit trail
4. **Liquidation Factor UI:** Operational mode (single slider) vs. CECL mode (term-level inputs)
5. **Flow-Through Rate KPI** displayed prominently

**Layout:**
```
┌──────────────────────────────────────────────────────┐
│  ECL Forecasting Engine                              │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─ MODE SELECTION ──────────────────────────────────┐│
│  │ ○ Operational Forecast (PyCraft)                  ││
│  │ ○ CECL Reserve Estimation (ASC 326)               ││
│  │                                                  ││
│  │ ┌─ FEG Scenario Selection ──────────────────────┐││
│  │ │ ○ Pre-FEG (Pure Model)                        │││
│  │ │ ○ Central (Baseline Macro)                    │││
│  │ │ ○ Post-FEG (Weighted Scenarios)               │││
│  │ └─────────────────────────────────────────────────┘││
│  └────────────────────────────────────────────────────┘│
│                                                      │
│  ┌─────────────── INPUT PANEL ──────────────────────┐│
│  │                                                  ││
│  │  Data Source: ○ Use LendingClub Dataset           ││
│  │               ○ Upload Receivables File           ││
│  │                                                  ││
│  │  [Upload Receivables File (.csv/.xlsx)]           ││
│  │                                                  ││
│  │  [📥 Upload Assumptions (.xlsx)] [📤 Export]      ││
│  │                                                  ││
│  │  ┌─── Assumptions ──────────────────────────┐    ││
│  │  │                                           │    ││
│  │  │ OPERATIONAL MODE:                         │    ││
│  │  │ Liquidation Factor (monthly): [__2.0__%]  │    ││
│  │  │                                           │    ││
│  │  │ CECL MODE (if selected):                  │    ││
│  │  │ Liquidation by Term:                      │    ││
│  │  │   36-month rate: [__3.2__% monthly]       │    ││
│  │  │   60-month rate: [__2.1__% monthly]       │    ││
│  │  │ Vintage-Age Differentiation: ○ Yes / ○ No │    ││
│  │  │                                           │    ││
│  │  │ New Monthly Originations: [$__50M___]     │    ││
│  │  │ (Visible in Operational mode only;        │    ││
│  │  │  hidden in CECL mode — hardcoded to $0)   │    ││
│  │  │                                           │    ││
│  │  │ Recovery Rate:              [__17___%]    │    ││
│  │  │ Discount Rate (annual):     [__12___%]    │    ││
│  │  │ Projection Horizon:         [10 years]    │    ││
│  │  │                                           │    ││
│  │  │ [▶ Run Forecast]                          │    ││
│  │  └───────────────────────────────────────────┘    ││
│  └──────────────────────────────────────────────────┘│
│                                                      │
│  ┌─────────────── OUTPUT PANEL ─────────────────────┐│
│  │                                                  ││
│  │  ┌──────────┬──────────┬──────────┬──────────┐  ││
│  │  │ Year 1   │ Year 3   │ Year 5   │ Year 10  │  ││
│  │  │ ECL      │ ECL      │ ECL      │ ECL      │  ││
│  │  │ $XXM     │ $XXM     │ $XXM     │ $XXM     │  ││
│  │  └──────────┴──────────┴──────────┴──────────┘  ││
│  │                                                  ││
│  │  ┌──────────┬──────────────────────────────────┐ ││
│  │  │ Flow-    │ X.XX% → Early Warning if         │ ││
│  │  │ Through  │ above threshold                  │ ││
│  │  │ Rate     │                                  │ ││
│  │  └──────────┴──────────────────────────────────┘ ││
│  │                                                  ││
│  │  ┌───────────────────────────────────────────┐   ││
│  │  │ Projected Receivables & ECL Over Time     │   ││
│  │  │ (Dual-axis: Receivables declining,        │   ││
│  │  │  ECL reserve path)                        │   ││
│  │  └───────────────────────────────────────────┘   ││
│  │                                                  ││
│  │  ┌───────────────────────────────────────────┐   ││
│  │  │ Projected GCO / NCO / Recovery Over Time  │   ││
│  │  │ (Stacked area chart)                      │   ││
│  │  └───────────────────────────────────────────┘   ││
│  │                                                  ││
│  │  ┌───────────────────────────────────────────┐   ││
│  │  │ ALLL Ratio Trajectory                     │   ││
│  │  │ (ECL / Outstanding over time)             │   ││
│  │  └───────────────────────────────────────────┘   ││
│  │                                                  ││
│  │  ┌───────────────────────────────────────────┐   ││
│  │  │ Projected Receivables Tracker Table       │   ││
│  │  │ (Downloadable Excel — same format as      │   ││
│  │  │  institutional receivables tracker)       │   ││
│  │  │                                           │   ││
│  │  │  [📥 Download Forecast (.xlsx)]           │   ││
│  │  └───────────────────────────────────────────┘   ││
│  └──────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

**Loss Given Default (LGD) Formula — Updated for V4:**

The LGD computation now explicitly includes recoveries and collection recovery fees:

**Primary LGD formula:**
```
LGD = 1 - ((recoveries - collection_recovery_fee) / EAD)
```

Where:
- `recoveries`: post-charge-off cash recovered (100% populated in dataset)
- `collection_recovery_fee`: fee paid to recovery agent (100% populated in dataset)
- `net_recovery` = recoveries - collection_recovery_fee
- EAD: Exposure at Default (original loan amount or balance at default)

**Cross-check formula:**
```
LGD_simple = 1 - (total_rec_prncp / EAD)
```

Where `total_rec_prncp` is total principal recovered (100% populated in dataset).

Both columns are 100% populated in the LendingClub dataset, allowing full LGD analysis without missing values.

**Redesigned ECL Projection Engine (engine/ecl_projector.py):**

```python
class ECLProjector:
    """
    Core forecasting engine — the PyCraft equivalent (V3 Redesigned).

    Takes a receivables snapshot and assumptions, projects forward
    using flow rates (simple bucket-to-bucket ratios), and computes
    ECL at each future month.

    DUAL-MODE DESIGN:
    1. Operational Mode (PyCraft-style):
       - Uses 6-month rolling average of historical flow rates
       - Applies flat across entire projection horizon
       - Simple, operationally practical
       - Used for: AOP, FRP, internal planning

    2. CECL Mode (ASC 326 compliant):
       - Phase 1 (R&S period): Macro-adjusted flow rates
       - Phase 2 (reversion period): Gradual transition to historical averages
       - Phase 3 (remaining horizon): Long-run historical averages
       - Used for: CECL reserve estimation, regulatory reporting

    IMPORTANT: Flow rates are NOT passed in at __init__. Instead, they are
    COMPUTED from the receivables tracker during the forecasting process.

    NOTE ON FEG FRAMEWORK:
    The FEG toggle (Pre-FEG/Central/Post-FEG) applies to BOTH Operational and CECL modes.
    Mode determines HOW baseline flow rates are computed (rolling avg for Operational,
    three-phase approach for CECL). FEG determines WHETHER and HOW stress is applied.
    Key difference: In Operational mode, stressed rates extend flat for the full horizon.
    In CECL mode, stressed rates only apply during Phase 1 (R&S period) then revert.
    """

    def __init__(self, pd_model=None, lgd_model=None):
        """
        Initialize the ECL projector with optional PD and LGD models.

        Args:
            pd_model: trained PD model (optional — for ECL computation)
            lgd_model: trained LGD model (or scalar LGD assumption)

        No flow_rates parameter — they are computed from the receivables tracker.
        """
        self.pd_model = pd_model
        self.lgd_model = lgd_model
        self.receivables_tracker = None
        self.flow_rates_historical = None
        self.flow_rates_forecast = None
        self.mode = 'operational'  # 'operational' or 'cecl'
        self.feg_scenario = 'central'  # 'pre-feg', 'central', 'post-feg'

    def load_receivables(self, data, format='tracker'):
        """
        Load the receivables tracker file.
        The tracker contains BOTH:
        - Dollar receivables by DPD bucket by month (top section)
        - Historical flow rates by bucket by month (bottom section)
        These flow rates are already in the file because they're
        simple ratios computed from the dollar receivables.

        Args:
            data: DataFrame or path to Excel file in institutional format
            format: 'tracker' for institutional-style Excel format (default)

        Expected structure:
        - Row groups for: Dollar Receivables (buckets), Account Counts, GCO, Recovery, NCO
        - Row groups for: Flow Rates (by transition)
        - Columns: Monthly periods
        """
        self.receivables_tracker = data
        # Extract historical flow rates from the tracker
        self._extract_flow_rates_from_tracker()

    def _extract_flow_rates_from_tracker(self):
        """
        Parse the receivables tracker and extract historical flow rates.
        Flow rates are already in the file as computed ratios.
        """
        # Implementation: find the "Flow Rates" section and extract
        # Compute rolling average (default 6 months)
        pass

    def set_mode(self, mode='operational', **kwargs):
        """
        Set the forecasting mode.

        Args:
            mode: 'operational' or 'cecl'

            For 'operational' mode:
                kwargs: liquidation_factor (float, default 0.02)

            For 'cecl' mode:
                kwargs:
                    liquidation_factor_36m (float, default 0.032)
                    liquidation_factor_60m (float, default 0.021)
                    vintage_age_differentiation (bool, default False)
                    rs_period_months (int, default 24)
                    reversion_method (str, 'straight_line', 'immediate', 'stepped')
                    reversion_period_months (int, default 12)
        """
        self.mode = mode

    def set_feg_scenario(self, scenario='central'):
        """
        Set the FEG scenario. Applies to BOTH Operational and CECL modes.

        Args:
            scenario: 'pre-feg' (no macro), 'central' (baseline), or 'post-feg' (weighted)

        NOTE: This is orthogonal to mode selection. In both modes:
        - Pre-FEG: empirical flow rates with no macro overlay
        - Central: apply baseline macro scenario to flow rates
        - Post-FEG: weighted average flow rates across scenarios

        In Operational mode, the chosen FEG rates extend flat for the full horizon.
        In CECL mode, FEG adjustment applies during Phase 1 (R&S), then reverts.
        """
        self.feg_scenario = scenario

    def compute_forecast_flow_rates(self, lookback_months=6,
                                     method='extend',
                                     rs_period_months=24,
                                     reversion_method='straight_line',
                                     reversion_period_months=12,
                                     historical_avg_lookback_months=60):
        """
        Compute flow rates for the forecast horizon.
        Two modes:

        method='extend' (OPERATIONAL MODE, PyCraft style):
            - Take 6-month rolling average of flow rates
            - Apply flat across entire projection horizon
            - Simple, operationally practical
            - Used for: AOP, FRP, internal planning

        method='cecl' (CECL MODE, ASC 326 compliant):
            - Phase 1 (months 1 to rs_period): Use macro-adjusted
              flow rates (6-month avg + scenario overlay)
            - Phase 2 (reversion_period): Gradually transition from
              Phase 1 rates to long-run historical averages
            - Phase 3 (remaining horizon): Pure historical averages,
              NO adjustment for current/future conditions
            - Used for: CECL reserve estimation, regulatory reporting

        Args:
            lookback_months: window for computing historical average (default 6)
            method: 'extend' or 'cecl' (matches self.mode)
            rs_period_months: R&S (Reasonable and Supportable) period for CECL
            reversion_method: how to transition from Phase 1 to Phase 3
                - 'straight_line': linear interpolation
                - 'immediate': jump to Phase 3 after R&S
                - 'stepped': step function (quarterly transitions)
            reversion_period_months: duration of Phase 2 reversion
            historical_avg_lookback_months: lookback window for long-run average

        Returns:
            dict: flow_rates_forecast with monthly rates for entire horizon
        """
        if method == 'extend':
            # OPERATIONAL: flat 6-month rolling average
            rolling_avg = self._compute_rolling_avg_flow_rates(
                self.flow_rates_historical,
                window=lookback_months
            )
            # Extend flat across 120 months (10 years)
            self.flow_rates_forecast = {
                bucket: [rolling_avg[bucket]] * 120
                for bucket in rolling_avg.keys()
            }

        elif method == 'cecl':
            # CECL: Phase 1 (R&S) → Phase 2 (reversion) → Phase 3 (long-run)
            phase1_rates = self._compute_phase1_rates(rs_period_months)
            phase3_rates = self._compute_phase3_rates(historical_avg_lookback_months)

            self.flow_rates_forecast = self._blend_phases(
                phase1_rates, phase3_rates,
                rs_period_months, reversion_period_months,
                reversion_method
            )

        return self.flow_rates_forecast

    def load_assumptions(self, excel_file):
        """
        Load assumptions from institutional-format Excel file.

        Expected structure:
        - Sheet: "Assumptions"
        - Rows: Liquidation Factor, New Originations, Recovery Rate,
                Discount Rate, Mode, FEG Scenario, etc.
        """
        pass

    def export_assumptions(self, excel_file):
        """
        Export current Streamlit settings as Excel file for audit trail.
        """
        pass

    def set_assumptions(self, liquidation_factor=0.02, new_originations=0,
                        recovery_rate=0.17, discount_rate=0.12):
        """
        Set projection assumptions (operational mode).

        Args:
            liquidation_factor: monthly runoff rate (prepayments + maturities)
                               e.g., 0.02 = 2% of current balance exits each month
            new_originations: monthly new loan volume ($)
                             NOTE: Only visible in Operational mode. In CECL mode,
                             this is hidden and hardcoded to $0 because CECL reserves
                             are for the existing portfolio; new loans get Day 1 CECL
                             at origination.
            recovery_rate: expected recovery on charged-off balances
                          Default value (0.17) is seeded from the LGD model's
                          portfolio-level output: LGD ≈ 83% → recovery = 17%,
                          confirmed from LendingClub 10-K. The slider allows
                          override for sensitivity/stress testing.
                          This connects the LGD model (loan-level, from
                          Notebooks 05/06) to the flow-rate projection engine
                          (portfolio-level).
            discount_rate: annual discount rate for DCF computation
        """
        pass

    def project(self, n_months=120):
        """
        Run the forward projection. This is the core PyCraft algorithm (now dual-mode).

        For each month t = 1 to n_months:
            1. Apply flow rates to project balances into next DPD bucket:
               - 30_DPD(t) = flow_rate_30(t) × Current(t-1)
               - 60_DPD(t) = flow_rate_60(t) × 30_DPD(t-1)
               - ... through each bucket
               - GCO(t)    = flow_rate_gco(t) × 180_DPD(t-1)
            2. Apply liquidation factor (portfolio runoff from Current)
            3. Add new originations (Operational mode only; $0 in CECL mode)
            4. Compute Recovery = recovery_rate × GCO(t)
            5. Compute NCO = GCO - Recovery
            6. Compute ECL at month t:
               - Simple: PD(remaining) × EAD(outstanding) × LGD
               - DCF: NPV of expected cash flow shortfalls
            7. Compute ALLL ratio = ECL / total_outstanding
            8. Compute Flow-Through Rate (Current → GCO) = product of all flow rates
            9. Compute reserve build/release = ECL(t) - ECL(t-1) - NCO(t)

        Returns:
            ProjectionResult object containing:
            - monthly_balances: DataFrame (months × buckets)
            - monthly_gco: Series
            - monthly_recovery: Series
            - monthly_nco: Series
            - monthly_ecl: Series
            - monthly_alll_ratio: Series
            - monthly_flow_through_rate: Series (NEW)
            - monthly_reserve_change: Series (provision expense)
            - summary_by_year: DataFrame (annual aggregates)
        """
        pass

    def apply_macro_overlay(self, scenario_weights=None):
        """
        Apply macro scenario overlay to flow rates.

        APPLIES TO BOTH OPERATIONAL AND CECL MODES.

        Adjusts flow rates based on macroeconomic conditions:
        - Pre-FEG: use empirical flow rates as-is (no adjustment)
        - Central: apply baseline macro scenario to flow rates
        - Post-FEG: weighted average flow rates across all scenarios + qualitative adjustment

        If scenario_weights provided (e.g., {'baseline': 0.60,
        'mild_downturn': 0.25, 'stress': 0.15}), compute weighted
        average flow rates across scenarios.

        This mirrors the FEG framework:
        - Pre-FEG = pure model output (no macro overlay)
        - Central = baseline macro scenario applied
        - Post-FEG = weighted average across all scenarios + qualitative adjustment

        Key difference between modes:
        - In Operational mode, stressed rates extend flat for the full horizon
        - In CECL mode, stressed rates apply during Phase 1 (R&S period) then revert
        """
        pass

    def export_tracker(self, filepath='forecast_output.xlsx'):
        """
        Export projection in institutional receivables tracker format.

        Sheet 1: Summary (annual KPIs)
        Sheet 2: Monthly Balances by DPD Bucket (with Flow Rates below)
        Sheet 3: Monthly GCO, Recovery, NCO
        Sheet 4: Monthly ECL and ALLL Ratio
        Sheet 5: Flow Rates Used (baseline + scenario-adjusted)
        Sheet 6: Assumptions
        """
        pass


class ProjectionResult:
    """Container for projection outputs with plotting methods."""

    def plot_receivables_trajectory(self):
        """Dual-axis: total receivables (declining) + ECL reserve."""

    def plot_loss_trajectory(self):
        """Stacked area: GCO, Recovery, NCO over time."""

    def plot_alll_ratio(self):
        """ALLL ratio trajectory with LendingClub 10-K benchmark line."""

    def plot_delinquency_composition(self):
        """Stacked bar of DPD bucket composition over time."""

    def plot_flow_through_rate_trajectory(self):
        """Flow-through rate over time with early warning threshold."""

    def to_dataframe(self):
        """Full projection as a single wide DataFrame."""

    def summary_table(self):
        """Annual summary: Year, Outstanding, GCO, NCO, ECL, ALLL Ratio, FTR."""
```

**Key User Interactions:**
1. Select forecasting mode (Operational vs. CECL) — activates different UI panels
2. Select FEG scenario (Pre-FEG / Central / Post-FEG) — applies to both modes
3. User can use LendingClub dataset (auto-generates receivables snapshot) OR upload their own file
4. Upload assumptions from Excel OR adjust sliders in UI
5. Run button triggers projection and updates all charts
6. Download button exports full forecast as Excel (in institutional tracker format)
7. Scenario comparison: run multiple scenarios side-by-side

---

### Page 5: Macro Scenario Analysis — STRESS AT FLOW RATE LEVEL

**Purpose:** Integrate macroeconomic data and run scenario-weighted ECL projections with stress applied at the flow rate level (not final ECL multiplier).

**V3 Key Change: Flow-Rate-Level Stress**

Instead of applying stress as a multiplier on final ECL (which loses information about which delinquency transitions are affected), stress scenarios now adjust individual flow rates multiplicatively. This is methodologically superior because:

1. **Preserves non-linear delinquency dynamics:** A 15% stress on each flow rate produces ~75% increase in cumulative flow-through due to compounding
2. **More accurate:** Reflects how macro conditions actually affect transitions (higher unemployment → more Current → 30 DPD, more 30 DPD → 60 DPD, etc.)
3. **Audit trail:** Clear visibility into which flow rates were adjusted and by how much

**Implementation (engine/macro_overlay.py):**

```python
class MacroOverlay:
    """
    Apply macroeconomic scenario stress at the flow rate level.

    Instead of multiplying final ECL by a stress factor, stress scenarios
    adjust individual flow rates multiplicatively. This preserves non-linear
    delinquency dynamics and is methodologically superior.

    NOTE: Flow-rate-level stress applies to BOTH Operational and CECL modes.
    In Operational mode, stressed rates extend flat. In CECL mode, they apply
    during Phase 1 (R&S) then revert.
    """

    def __init__(self):
        self.scenarios = {
            'baseline': {},
            'mild_downturn': {},
            'stress': {}
        }
        self.scenario_weights = {
            'baseline': 0.75,
            'mild_downturn': 0.15,
            'stress': 0.10
        }

    def define_scenario(self, scenario_name, flow_rate_stresses):
        """
        Define a stress scenario as a set of flow rate multipliers.

        Args:
            scenario_name: 'baseline', 'mild_downturn', 'stress'
            flow_rate_stresses: dict like {
                '30_plus': 1.0,      # no change
                '60_plus': 1.10,     # 10% increase
                '90_plus': 1.15,     # 15% increase
                '120_plus': 1.12,    # 12% increase
                '150_plus': 1.08,    # 8% increase
                '180_plus': 1.05,    # 5% increase
                'gco': 1.02          # 2% increase
            }
        """
        self.scenarios[scenario_name] = flow_rate_stresses

    def apply_stress_to_flow_rates(self, baseline_flow_rates, scenario='mild_downturn'):
        """
        Apply scenario stress multipliers to baseline flow rates.

        Args:
            baseline_flow_rates: dict of baseline rates
            scenario: scenario name

        Returns:
            dict: stressed flow rates
        """
        stressed = {}
        for bucket, rate in baseline_flow_rates.items():
            if bucket in self.scenarios[scenario]:
                multiplier = self.scenarios[scenario][bucket]
                stressed[bucket] = rate * multiplier
            else:
                stressed[bucket] = rate
        return stressed

    def compute_flow_through_under_stress(self, baseline_rates, scenario):
        """
        Demonstrate compounding effect of flow rate stress.

        Example:
            Baseline: [0.028, 0.382, 0.701, 0.85, 0.90, 0.92, 0.95]
            FTR = 0.468%

            Apply 15% stress to each: [0.0322, 0.4393, 0.8062, 0.9775, 1.035, 1.058, 1.093]
            FTR_stressed = product = ~1.23%
            → ~163% increase in flow-through (compounding, not 15% flat increase)
        """
        stressed_rates = self.apply_stress_to_flow_rates(baseline_rates, scenario)
        ftr_baseline = self._compute_flow_through_rate(baseline_rates)
        ftr_stressed = self._compute_flow_through_rate(stressed_rates)
        return {
            'baseline_ftr': ftr_baseline,
            'stressed_ftr': ftr_stressed,
            'percentage_increase': (ftr_stressed / ftr_baseline - 1) * 100
        }

    def compute_scenario_weighted_flow_rates(self, baseline_flow_rates):
        """
        Compute weighted average flow rates across all scenarios + qualitative adjustment.
        Used for 'Post-FEG' mode in ECLProjector.

        Returns:
            dict: blended flow rates
        """
        weighted = {bucket: 0 for bucket in baseline_flow_rates.keys()}
        for scenario, weight in self.scenario_weights.items():
            stressed = self.apply_stress_to_flow_rates(baseline_flow_rates, scenario)
            for bucket in weighted.keys():
                weighted[bucket] += weight * stressed[bucket]
        return weighted

    def _compute_flow_through_rate(self, flow_rates):
        """Multiply all flow rates to get end-to-end transition rate."""
        ftr = 1.0
        for bucket in ['30_plus', '60_plus', '90_plus', '120_plus', '150_plus', '180_plus', 'gco']:
            if bucket in flow_rates:
                ftr *= flow_rates[bucket]
        return ftr
```

**Streamlit Page Implementation (pages/05_scenario_analysis.py):**

```
┌──────────────────────────────────────────────────────┐
│  Macro Scenario Analysis                             │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─ Scenario Definition ─────────────────────────────┐│
│  │ Baseline (75%)    [────────────────────] 0.75     ││
│  │ Mild Downturn (15%) [────────────────] 0.15       ││
│  │ Stress (10%)      [────────────────] 0.10         ││
│  │                                                  ││
│  │ [⚙️ Customize Weights]                            ││
│  └────────────────────────────────────────────────────┘│
│                                                      │
│  ┌─ Flow Rate Stress by Scenario ────────────────────┐│
│  │                                                  ││
│  │ BASELINE (No stress):                             ││
│  │ 30+ FR: 2.8%  60+ FR: 38.2%  90+ FR: 70.1% ...   ││
│  │                                                  ││
│  │ MILD DOWNTURN (+10% to flow rates):               ││
│  │ 30+ FR: 3.1%  60+ FR: 42.0%  90+ FR: 77.1% ...   ││
│  │                                                  ││
│  │ STRESS (+15% to flow rates):                      ││
│  │ 30+ FR: 3.2%  60+ FR: 44.0%  90+ FR: 80.6% ...   ││
│  │                                                  ││
│  │ → Compounding Effect Example:                     ││
│  │   Baseline FTR (Current→GCO): 0.468%              ││
│  │   Stress FTR: 1.23%                              ││
│  │   Increase: 163% (due to multiplicative effect)  ││
│  └────────────────────────────────────────────────────┘│
│                                                      │
│  ┌─ Side-by-Side Comparison ─────────────────────────┐│
│  │                                                  ││
│  │ Baseline │ Mild Downturn │ Stress                ││
│  │ ─────────┼───────────────┼──────────             ││
│  │ [Chart]  │ [Chart]       │ [Chart]              ││
│  │ FTR:0.47%│ FTR: 0.73%    │ FTR: 1.23%           ││
│  │ ECL Year1│ ECL Year1     │ ECL Year1            ││
│  └────────────────────────────────────────────────────┘│
│                                                      │
│  ┌─ FRED Data Integration ───────────────────────────┐│
│  │ Unemployment Rate:  [Pull FRED] Current: 4.2%     ││
│  │ GDP Growth Rate:    [Pull FRED] Current: 2.3%     ││
│  │ HPI Growth Rate:    [Pull FRED] Current: 1.8%     ││
│  └────────────────────────────────────────────────────┘│
│                                                      │
│  ┌─ Sensitivity Sliders ─────────────────────────────┐│
│  │ Unemployment (3% to 10%):  [●─────────────] 4.2% ││
│  │   → Watch ECL update in real-time                 ││
│  │ Recovery Rate (5% to 30%): [───────●─────] 17%   ││
│  │   → Watch ECL adjust                              ││
│  │ Flow Rate Stress Multiplier (0.85-1.25):          ││
│  │   [──────●──────────────] 1.15                    ││
│  └────────────────────────────────────────────────────┘│
│                                                      │
│  ┌─ Weighted ECL Across Scenarios ───────────────────┐│
│  │ Baseline ECL (Year 1):    $52.3M (weight: 75%)    ││
│  │ Downturn ECL (Year 1):    $73.8M (weight: 15%)    ││
│  │ Stress ECL (Year 1):      $95.4M (weight: 10%)    ││
│  │ ───────────────────────────────────────────────  ││
│  │ Probability-Weighted ECL: $59.7M                  ││
│  └────────────────────────────────────────────────────┘│
│                                                      │
│  ┌─ Mean Reversion Visualization ────────────────────┐│
│  │ (Shows macro variables reverting to long-run     ││
│  │  mean after 8-quarter explicit forecast horizon)  ││
│  └────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────┘
```

**Key Features:**
1. **FRED Data Integration:** Pull real-time macro data (unemployment, GDP, HPI)
2. **Scenario Definition:** Define multiple scenarios with probability weights
3. **Flow-Rate-Level Stress:** Adjust individual flow rates multiplicatively (not final ECL)
4. **Sensitivity Dashboard:**
   - Slider: Unemployment rate (3% to 10%) → watch ECL change in real-time
   - Slider: Recovery rate (5% to 30%) → watch LGD and ECL adjust
   - Slider: Flow rate stress multiplier (0.85 to 1.25) → watch compounding effect
5. **Stress Testing:** Apply historical stress scenarios (2008-2009 conditions) to current portfolio
6. **Weighted ECL:** Compute probability-weighted ECL across all scenarios
7. **Mean Reversion Visualization:** Show how macro variables revert to long-run mean after explicit forecast horizon (8 quarters)

---

### Page 6: Model Monitoring

**Purpose:** Track model performance over time with RAG status — exactly the quarterly monitoring report from my prior role.

**Prior Role Connection:** Behavioral scorecard monitoring with Gini, PSI, CSI, VDI metrics and RAG framework.

**Features:**

1. **RAG Dashboard:**

   | Metric | Value | Threshold | Status |
   |--------|-------|-----------|--------|
   | Gini | 62.3% | ≥ 60% | GREEN |
   | PSI | 0.08 | < 0.10 | GREEN |
   | AUC | 0.78 | ≥ 0.75 | GREEN |
   | KS | 34.2% | ≥ 30% | GREEN |

2. **Gini Over Time:** Line chart showing Gini by quarter with green/amber/red zones

3. **PSI Analysis:** Score distribution comparison (train vs. each test period)

4. **CSI/VDI by Feature:** Table showing characteristic stability for each variable

5. **Calibration Check:** Predicted PD vs. actual default rate by decile

6. **Out-of-Time Performance:** Model metrics on 2016, 2017, 2018 separately

7. **Benchmark Population Tab (NEW for V4):**
   - Load benchmark_population_2014.csv (200K records with FICO, delinquency bucket, performance outcome)
   - PSI computation: compare model's score distribution against 2014 benchmark population
   - External calibration: predicted PD vs. actual outcomes for 2014 cohort
   - Display as table + chart with RAG thresholds

---

### Page 7: AI Analyst (Claude-Powered Chatbot)

**Purpose:** An intelligent assistant that can analyze portfolio data, answer questions, generate reports, and perform on-the-fly analysis.

**V4 Updates to System Prompt:**

The AI analyst now understands:
- **Dataset Metadata:** 2,260,668 usable loans (after footer removal), 151 columns total, ~65-75 usable after cleaning
- **Default Rate:** 19.96% on terminal loans
- **Loan Status Values:** 9 unique values: Charged Off, Default, Fully Paid, Current, In Grace Period, Late (16-30 days), Late (31-120 days), Does not meet the credit policy. Status:Fully Paid, Does not meet the credit policy. Status:Charged Off
- **Benchmark Population:** benchmark_population_2014.csv available for validation queries
- **LGD Columns:** `recoveries` and `collection_recovery_fee` available for LGD analysis
- **Flow-Through Rate concept:** Current → GCO cumulative transition rate, early warning signal
- **Pre-FEG/Central/Post-FEG distinction:** Different assumptions for model output vs. macro-adjusted vs. weighted scenarios
- **Dual-mode forecasting:** Operational (PyCraft-style flat rates) vs. CECL (three-phase R&S approach)
- **Competing risks:** Default vs. prepayment dynamics in portfolio projections
- **Flow-rate-level stress:** Stress applied multiplicatively to individual transitions, not final ECL

**Implementation (components/chatbot.py):**

```python
import anthropic
import streamlit as st
import pandas as pd
import json

class PortfolioAnalystBot:
    """
    Claude-powered AI analyst embedded in the Streamlit app (V4 Enhanced).

    Capabilities:
    1. Analyze uploaded files (CSV, Excel, JSON)
    2. Answer questions about the portfolio using loaded data
    3. Run on-the-fly calculations and sensitivity analysis
    4. Generate executive summaries and memos
    5. Search the web for macro data and industry benchmarks
    6. Explain flow-through rates, FEG framework, dual-mode forecasting
    """

    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.conversation_history = []
        self.portfolio_context = None

    def set_portfolio_context(self, portfolio_data: dict):
        """
        Load portfolio data into context for the AI to reference.

        Args:
            portfolio_data: dict containing:
                - receivables_summary: current receivables by bucket
                - ecl_summary: ECL by grade and vintage
                - model_metrics: Gini, AUC, PSI, etc.
                - flow_rates: current flow rates by bucket transition
                - flow_through_rates: cumulative transition rates
                - vintage_curves: cumulative default rates by vintage
                - macro_data: current macro variables
                - dataset_metadata: usable rows, columns, default rate
        """
        self.portfolio_context = portfolio_data

    def _build_system_prompt(self):
        """Build the system prompt with portfolio context and V5 concepts."""
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

        LENDINGCLUB DATASET KNOWLEDGE (V5):
        - Total usable loans: 2,260,668 (after footer removal, terminal statuses)
        - Total columns in raw dataset: 151
        - Usable columns after cleaning: ~65-75 features
        - Expected default rate: 19.96% (on terminal loans)
        - Loan status values (9 unique):
          * Terminal: Charged Off, Default, Fully Paid
          * Non-terminal: Current, In Grace Period, Late (16-30 days), Late (31-120 days)
          * Policy non-conforming: Does not meet the credit policy. Status:Fully Paid/Charged Off
        - Available for LGD analysis: 'recoveries', 'collection_recovery_fee'
        - Benchmark population: benchmark_population_2014.csv (200K records, FICO + delinquency + outcome)

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
           - Operational Mode: 6-month rolling average flow rates extended flat (PyCraft)
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

        CURRENT PORTFOLIO DATA:
        {json.dumps(self.portfolio_context, indent=2, default=str)}

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
        - If asked to generate a report, format it professionally

        If the user uploads a file, analyze it in the context of the existing
        portfolio data. Compare metrics, identify anomalies, and provide insights.
        """

    def chat(self, user_message, uploaded_file=None):
        """
        Process a user message and return a response.

        Args:
            user_message: the user's question or request
            uploaded_file: optional uploaded file to analyze

        Returns:
            str: the AI's response
        """
        # If file uploaded, parse and add to context
        if uploaded_file:
            file_content = self._parse_uploaded_file(uploaded_file)
            user_message = f"[User uploaded file: {uploaded_file.name}]\n\nFile contents:\n{file_content}\n\nUser message: {user_message}"

        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        response = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=self._build_system_prompt(),
            messages=self.conversation_history
        )

        assistant_message = response.content[0].text
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def _parse_uploaded_file(self, file):
        """Parse uploaded CSV/Excel/JSON file into a string summary."""
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        elif file.name.endswith('.json'):
            return json.dumps(json.load(file), indent=2)
        else:
            return file.read().decode('utf-8')

        # Return summary + head for large files
        summary = f"Shape: {df.shape}\n"
        summary += f"Columns: {list(df.columns)}\n"
        summary += f"Data types:\n{df.dtypes.to_string()}\n\n"
        summary += f"Summary statistics:\n{df.describe().to_string()}\n\n"
        summary += f"First 20 rows:\n{df.head(20).to_string()}"
        return summary
```

**Sample AI Interactions:**

User: *"Why is our flow-through rate trending up?"*
AI: *"Your flow-through rate has increased from 0.42% three months ago to 0.51% today — a 21% increase. This is being driven primarily by the 90+ flow rate, which jumped from 71.2% to 78.3% (a 10% increase). This suggests either: (1) accounts in the 60+ DPD bucket are rolling forward more aggressively, indicating weaker borrower finances; or (2) your origination policy has shifted toward riskier grades. I recommend investigating the cohort composition of the 60+ bucket and comparing it to historical patterns."*

User: *"Compare Pre-FEG vs. Central ECL for Year 1."*
AI: *"Pre-FEG ECL (pure model, no macro): $52.3M. Central FEG ECL (baseline macro scenario): $57.8M. The $5.5M difference reflects the baseline macro overlay, which assumes stable unemployment and moderate loan demand. If unemployment rises to 6%, we'd expect the ECL to approach the Stress scenario ($95.4M in Year 1)."*

User: *"Write a quarterly loss forecast memo for the CFO"*
AI: *Generates a formatted memo with: executive summary, portfolio metrics, flow-through rate trend, vintage performance, ECL forecast under base/stress scenarios, FEG framework explanation, recommended actions*

---

## Core Engine Modules

### engine/flow_rate_engine.py

Handles computation and projection of flow rates (already in V2, minor updates for V3):

```python
def build_receivables_tracker(df, period_col='month', status_col='dpd_bucket',
                               balance_col='outstanding_balance', grade=None):
    """Build institutional-format receivables tracker."""

def compute_flow_rates(receivables_tracker):
    """Compute flow rates from tracker."""

def compute_flow_rate_trends(receivables_tracker, window=6):
    """Compute rolling average flow rates."""

def project_balances(current_balances, flow_rates, n_months,
                     liquidation_factor=0.0, new_origination=0.0):
    """Project balances forward."""
```

### engine/prepayment.py (NEW)

```python
class PrepaymentModel:
    """
    Model the competing risk of prepayment (portfolio liquidation factor).

    Prepayment dynamics:
    - CPR (Conditional Prepayment Rate) based on current rates, seasoning, vintage
    - Refinancing burnout: older vintages have lower refinancing propensity
    - Seasonality: more prepayments in spring/summer

    Used in ECLProjector to compute realistic liquidation factor by month/scenario.
    """

    def compute_cpr(self, current_rates, mortgage_rates, vintage_age):
        """Compute conditional prepayment rate."""

    def apply_seasonality(self, cpr, month):
        """Adjust CPR for seasonal variation."""

    def compute_monthly_liquidation_factor(self, **kwargs):
        """Convert CPR to monthly liquidation rate."""
```

### engine/flow_through_calculator.py (NEW)

```python
def compute_flow_through_rates(flow_rates_dict):
    """
    Compute cumulative flow-through rates from individual bucket flow rates.

    Flow-Through Rate (Current → 30 DPD) = FR₁
    Flow-Through Rate (Current → 60 DPD) = FR₁ × FR₂
    ...
    Flow-Through Rate (Current → GCO) = FR₁ × FR₂ × ... × FR_GCO
    """
    flow_through = {}
    cumulative_product = 1.0
    for i, (bucket, rate) in enumerate(flow_rates_dict.items()):
        cumulative_product *= rate
        flow_through[f"ftr_{bucket}"] = cumulative_product
    return flow_through

def compute_flow_through_rate_trend(receivables_tracker, window=6):
    """
    Track flow-through rate trend over time.
    Used on Portfolio Overview and Roll-Rate Analysis pages.
    """
```

### engine/macro_overlay.py (REDESIGNED FOR V3)

As detailed in Page 5 section above. Stress applied at flow rate level, not final ECL.

### engine/ecl_projector.py (REDESIGNED FOR V3)

As detailed in Page 4 section above. Dual-mode (Operational/CECL), FEG framework.

---

## Streamlit App Main Entry Point

```python
# streamlit_app.py

import streamlit as st

st.set_page_config(
    page_title="LendingClub Risk Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #0066cc;
    }
    .rag-green { color: #28a745; font-weight: bold; }
    .rag-amber { color: #ffc107; font-weight: bold; }
    .rag-red { color: #dc3545; font-weight: bold; }
    .ftr-warning {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 10px;
        border-radius: 5px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 LC Risk Analytics")
st.sidebar.markdown("---")
st.sidebar.markdown("**Portfolio Management & Loss Forecasting Platform**")
st.sidebar.markdown("*Inspired by PyCraft — V5*")
st.sidebar.markdown("---")

# Data loading (cached)
@st.cache_data
def load_portfolio_data():
    """Load processed LendingClub data and model outputs."""
    # Load from data/processed/ and data/results/
    pass

# Main page content
st.markdown("# LendingClub Risk Analytics Platform")
st.markdown("### Portfolio Management • Loss Forecasting • Model Monitoring • AI Analysis")
st.markdown("---")

# Quick links to pages
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info("📊 **Portfolio Overview**\n\nKPIs, composition, flow-through rate trend")
with col2:
    st.info("🔄 **Roll-Rate Analysis**\n\nFlow rates, receivables tracker, flow-through rates")
with col3:
    st.info("📈 **Vintage Performance**\n\nCumulative default curves by MOB")
with col4:
    st.info("🔮 **ECL Forecasting**\n\nDual-mode engine (Operational/CECL), FEG toggle")
```

---

## Technical Requirements

### Dependencies
```
streamlit>=1.28
plotly>=5.15
anthropic>=0.18
pandas>=2.0
numpy>=1.24
openpyxl>=3.1          # Excel export/import
fredapi>=0.5           # FRED macro data
scikit-learn>=1.3      # Model loading
xgboost>=2.0           # Model loading
```

### Environment Setup
```bash
# .streamlit/secrets.toml
ANTHROPIC_API_KEY = "your-api-key-here"
FRED_API_KEY = "your-fred-key-here"
```

### Running the App
```bash
cd lending-club-credit-risk
streamlit run app/streamlit_app.py
```

---

## Deployment Options

For interview demonstration:
1. **Local:** Run on laptop during interview (most reliable)
2. **Streamlit Cloud:** Free deployment at share.streamlit.io (requires public GitHub repo)
3. **Screen recording:** Pre-record a walkthrough video as backup

---

## Development Priority

Given the 3-week timeline, prioritize in this order:

1. **Must-have (Days 15-17):**
   - Portfolio Overview dashboard (with flow-through rate KPI)
   - Roll-Rate Analysis with flow rates and flow-through rates
   - Vintage Performance curves
   - ECL Forecasting engine (dual-mode with FEG toggle, basic UI)

2. **Should-have (Days 17-18):**
   - Macro Scenario Analysis with flow-rate-level stress
   - Model Monitoring with RAG status + benchmark population validation
   - Excel upload/export of assumptions
   - Excel export of forecast results

3. **Nice-to-have (Days 19-21):**
   - AI Analyst chatbot (Claude integration with V5 concepts)
   - Advanced sensitivity sliders
   - Sankey diagram for delinquency flow
   - Downloadable quarterly report generation
   - Prepayment model for competing risks

The AI chatbot is the most impressive feature but also the most time-intensive.
If time is tight, the core forecasting engine + dashboard + vintage analysis
already demonstrate everything a hiring manager needs to see. The chatbot
becomes the "wow factor" if you have time.

---

## Summary of V5 Changes from V4

| Change | Description | Impact |
|--------|-------------|--------|
| **Generic Framing** | Replaced all HSBC/institutional-specific references with portable framing | Interview-ready, no company baggage |
| **New Originations Visibility** | Hidden in CECL mode (hardcoded to $0); visible in Operational mode | Correctly reflects CECL mechanics |
| **FEG/Stress Applies to Both Modes** | Documented that FEG toggle + flow-rate stress apply to Operational and CECL | Clearer methodology, orthogonal design |
| **Recovery Rate LGD Connection** | Updated to reference LGD model output (83% LGD → 17% recovery) | Data-driven defaults, audit trail |
| **Sidebar Version Update** | Changed from "V4" to "V5" | Current version marking |

All V4 content retained and enhanced. V5 is data-driven, portable, and ready for interview.
