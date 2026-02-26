# CECL Credit Risk Analytics — LendingClub Portfolio

End-to-end **CECL-compliant** credit risk platform for a **$19.4B LendingClub consumer lending portfolio** (1.35M loans, 2007–2018). Built using the same institutional frameworks used in production ALLL/CECL workflows at major financial institutions — PD/LGD/EAD modeling, synthetic receivables tracking, forward default flow rates, macro stress testing, and probability-weighted ECL across Pre-FEG, Central, and Post-FEG views.

## Key Results

| Metric | Value |
|--------|-------|
| **Portfolio** | $19.39B total funded, $11.74B EAD, 1,345,350 loans, 7 credit grades |
| **PD Scorecard** (Logistic Regression) | AUC **0.6931**, Gini 38.6%, KS 28.1% — 13 WoE features |
| **PD ML Ensemble** (XGBoost) | AUC **0.7202** — 50 SHAP-selected features from 101 |
| **LGD Model** (Two-stage) | MAE **7.6%**, portfolio mean 89.0% |
| **EAD Model** (Random Forest) | MAPE **23.8%** |
| **Pre-FEG ECL** (baseline) | $2.39B — 20.35% ALLL ratio |
| **Central ECL** (macro-adjusted) | $1.15B — 9.81% ALLL ratio |
| **Post-FEG ECL** (probability-weighted) | **$3.49B** — 29.71% ALLL under macro stress |
| **FEG Formula** | 0.60×$1.15B + 0.25×$4.75B + 0.15×$10.72B = $3.49B |
| **Population Stability** | All PSI GREEN (0.0043, 0.0097, 0.0355) across test periods |

## Architecture

```mermaid
graph TD
    A["<b>DATA LAYER</b><br/>LendingClub Loans (1.35M) + FRED Macro (UNRATE, HPI, GDP, CPI)"] --> B["<b>PD Model</b><br/>Scorecard + XGBoost"]
    A --> C["<b>LGD Model</b><br/>Two-stage<br/>Classification + Regression"]
    A --> D["<b>EAD Model</b><br/>Random Forest"]
    A --> E["<b>Prepayment</b><br/>Competing Risks<br/>Kaplan-Meier CPR"]

    B --> F["<b>Synthetic Monthly Panel</b><br/>31.4M rows · Receivables Tracker · Flow Rate Engine"]
    C --> F
    D --> F
    E --> F

    F --> G["<b>Pre-FEG</b><br/>Baseline ECL<br/>$2.39B · 20.35%"]
    F --> H["<b>Central</b><br/>Macro-Adjusted ECL<br/>$1.15B · 9.81%"]
    F --> I["<b>Post-FEG</b><br/>Probability-Weighted<br/>$3.49B · 3 Scenarios"]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#16213e,stroke:#0f3460,color:#fff
    style G fill:#0f3460,stroke:#53a8b6,color:#fff
    style H fill:#0f3460,stroke:#53a8b6,color:#fff
    style I fill:#e94560,stroke:#e94560,color:#fff
```

## Project Structure

```
├── notebooks/
│   ├── 01_EDA_and_Data_Cleaning.ipynb        # Data profiling, cleaning, FRED macro merge
│   ├── 02_WOE_IV_Feature_Engineering.ipynb   # WoE/IV binning, feature selection
│   ├── 03_PD_Scorecard_with_Grade.ipynb      # Logistic regression PD scorecard
│   ├── 04_PD_Model_ML_Ensemble.ipynb         # XGBoost/LightGBM + SHAP analysis
│   ├── 055_Prepayment_Model.ipynb            # Kaplan-Meier CPR, competing risks
│   ├── 05_EAD_Model.ipynb                    # Exposure at Default
│   ├── 06_LGD_Model.ipynb                    # Loss Given Default (two-stage)
│   ├── 07_ECL_and_Flow_Rates.ipynb           # Receivables tracker, DCF-ECL engine
│   ├── 08_Model_Validation_and_Monitoring.ipynb  # PSI/CSI/Gini/KS, RAG framework
│   └── 09_Macro_Scenarios_and_Strategy.ipynb # 3-scenario stress testing, FEG views
│
├── src/                          # Modular Python package
│   ├── ecl_engine.py             # DCF-ECL computation with competing risks
│   ├── flow_rates.py             # Forward default flow rates from receivables tracker
│   ├── macro_scenarios.py        # Multi-quarter UNRATE paths, stress multipliers
│   ├── models.py                 # PD/LGD/EAD model wrappers
│   ├── scorecard.py              # WoE scorecard scoring engine
│   ├── validation.py             # Discrimination, calibration, stability metrics
│   └── woe_binning.py            # Weight of Evidence binning framework
│
├── data/
│   ├── results/                  # All outputs: visualizations, tables, metrics
│   ├── processed/                # Cleaned parquets, train/val/test splits
│   └── raw/                      # Original LendingClub CSV (not tracked)
│
├── streamlit_app/               # Interactive dashboard
│   ├── app.py                   # Landing page
│   └── pages/                   # 7 dashboard pages
│       ├── 1_Portfolio_Overview  # KPIs, geographic, grade analysis
│       ├── 2_Roll_Rate_Analysis  # Receivables tracker, Sankey, flow rates
│       ├── 3_Vintage_Performance # MOB curves, marginal PD, seasoning
│       ├── 4_ECL_Forecasting    # Operational + CECL mode projections
│       ├── 5_Scenario_Analysis  # 3-scenario stress testing, FEG views
│       ├── 6_Model_Monitoring   # RAG dashboard, PSI/CSI, calibration
│       └── 7_AI_Analyst         # Natural language portfolio Q&A
│
├── docs/                         # Project documentation and roadmap
├── config.py                     # Centralized paths and constants
└── requirements.txt
```

## Methodology

### PD Modeling — Dual Approach

**Logistic Regression Scorecard** — Behavioral monitoring model using WoE/IV feature engineering. Grade is included as an origination-time attribute; `int_rate` and `sub_grade` are excluded to avoid circularity. 13 features selected via IV >= 0.05 and pairwise correlation < 0.70. L2 regularization tuned via 5-fold stratified CV. Test AUC 0.693 with temporal split (train 2007–2015, test 2017–2018) — consistent with published benchmarks for origination-only features without leakage.

**XGBoost Ensemble** — Performance ceiling model including all 101 features (macro, borrower, loan, bureau). Optuna-tuned. SHAP-based feature selection reduces to 50 features with < 0.1% AUC loss. Test AUC 0.720. SHAP analysis reveals UNRATE, HPI, and DFF as retained macro drivers.

### LGD — Two-Stage Model

Stage 1 classifies P(any recovery) via logistic regression. Stage 2 predicts recovery rate conditional on recovery via gradient boosting. Combined: LGD = 1 − P(recovery) × E[recovery_rate | recovery]. Portfolio mean LGD of 89.0% aligns with unsecured consumer credit benchmarks.

### ECL — DCF with Competing Risks

Monthly cash flow projection where each loan faces three outcomes per period: stay current, default, or prepay. Cash flows discounted at effective interest rate. ECL = NPV(contractual) − NPV(expected). Prepayment rates from Kaplan-Meier survival analysis (CPR lookup by term × grade × vintage). Pre-FEG ECL of $2.39B (20.35% ALLL), Central ECL of $1.15B (9.81% ALLL), and Post-FEG probability-weighted ECL of $3.49B (29.71% ALLL) across three macro scenarios (60/25/15 weights).

### Forward Default Flow Rates

Synthetic monthly panel (31.4M rows) reconstructed from loan-level terminal outcomes. Receivables tracker in institutional format: monthly dollar balances across 7 DPD buckets (Current, 30+, 60+, 90+, 120+, 150+, 180+) with GCO, Recovery, and NCO. Flow rates are forward-only (curing is unobservable from this dataset). Flow Through Rate tracks cumulative delinquency progression as a product of all intermediate flow rates.

### Macro Stress Testing — 3 Scenarios

Multi-quarter forward UNRATE paths over 8-quarter projection horizon:

| Scenario | UNRATE Path | Weight |
|----------|-------------|--------|
| **Central** | Baseline UNRATE 4.0%, actual 2019 FRED path | 60% |
| **Mild Downturn** | Peak UNRATE 5.5%, then mean reversion | 25% |
| **Stress** | Peak UNRATE 10.0%, then mean reversion | 15% |

Time-varying stress multipliers applied at the flow rate level (not output-level), preserving non-linear compounding dynamics. Post-FEG ECL is the probability-weighted blend across all scenarios.

### Model Validation

Comprehensive monitoring framework: Gini, KS, AUC with bootstrap confidence intervals, Hosmer-Lemeshow calibration, PSI/CSI/VDI stability testing, and RAG status reporting. All population stability indices GREEN (PSI < 0.04) across 2017–2018 test periods.

## Data

**Source**: [LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) — `accepted_2007_to_2018Q4.csv` (2.26M rows, 151 columns). After filtering to terminal statuses (Fully Paid + Charged Off/Default): 1,345,350 loans.

**Macro variables** merged from FRED API by origination month: unemployment rate (UNRATE), Case-Shiller HPI, GDP growth, CPI, Federal Funds Rate, and consumer sentiment.

**Important**: The public LendingClub dataset provides loan-level terminal outcomes, not monthly payment tapes. The synthetic monthly panel is reconstructed from origination dates, last payment dates, and terminal statuses. In production, this framework would use observed monthly payment data directly.

## Setup

```bash
git clone https://github.com/harshvardhan-sekar/lendingclub-credit-risk.git
cd lendingclub-credit-risk

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download data
python download_data.py

# Run notebooks sequentially (01 through 09)
jupyter notebook
```

## License

MIT
