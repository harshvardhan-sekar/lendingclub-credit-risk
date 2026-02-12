# LendingClub Credit Risk Analytics Project — Roadmap V5

## Executive Summary

**Objective:** Build a production-grade Credit Risk Analytics project that goes beyond model-building to demonstrate portfolio strategy, loss forecasting, and operational risk management capabilities — the exact skills LendingClub's Credit Strategy, Loss Forecasting, and Model Development teams need.

**Timeline:** Flexible — quality over speed, target interview-ready by March/April 2026

**Key Shift from V1:** The project is no longer "build PD/EAD/LGD models." It's "build the kind of loss forecasting and portfolio management tool that LendingClub's risk team uses every day, powered by industry-standard models and enhanced with AI."

**Two Deliverables:**
1. **Modeling Notebooks** (Weeks 1-2): The analytical backbone — PD scorecard with macro covariates, ML models, EAD, LGD, ECL, vintage analysis, validation
2. **LendingClub Risk Analytics Platform** (Week 3): A Streamlit-based portfolio management and loss forecasting tool inspired by PyCraft, with dual-mode forecasting (Operational vs. CECL) and an embedded AI chatbot

---

## Changes from V2

The following major enhancements have been added to V3:

1. **FRED Macroeconomic Data Integration** — 6 key series (unemployment, HPI, GDP, CPI, Fed Funds, consumer sentiment) merged onto loans by origination month in Notebook 01
2. **Macro Features as PD Model Covariates** — PD models now include unemployment, GDP, HPI to capture economic regime dependence; essential for time-based validation to work correctly
3. **Competing Risks & Prepayment Model** — New section added to Week 2; prepayment rates computed from LendingClub data and fed into DCF-ECL calculation
4. **LGD Model Terminology Fix** — "Stage 1/Stage 2" renamed to "Step 1 (Classification Phase)/Step 2 (Regression Phase)" to avoid IFRS-9 confusion
5. **ECLProjector Class Redesign** — Flow rates now computed FROM receivables tracker data, not a constructor parameter; new `compute_forecast_flow_rates()` method
6. **Dual-Mode Forecasting Engine** — method='extend' (operational, 6-month rolling average extended flat) vs. method='cecl' (ASC 326 compliant, three-phase with macro adjustment, reversion, historical)
7. **Flow Through Rate Metric** — Explicit output: product of all intermediate flow rates (0.028 × 0.382 × ... = 0.468% final), displayed as KPI on Streamlit
8. **Pre-FEG / Central / Post-FEG ECL Toggle** — Streamlit radio buttons showing impact of macro overlay and scenario weighting on final ECL
9. **Stress Applied at Flow Rate Level** — Not ECL level; preserves multiplicative dynamics (15% stress per flow rate ≈ 75% cumulative effect)
10. **Liquidation Factor Design** — Operational mode (single portfolio-level slider) vs. CECL mode (differentiated by term and vintage from empirical curves)
11. **Assumption Input UI Design** — Streamlit sliders + "Upload/Export Assumptions" Excel buttons for operational workflow compatibility
12. **Conditional PD Model** — PD outputs now vary by borrower characteristics, macro scenario, and loan age/MOB for scenario-dependent ECL
13. **CLAUDE.md Updates** — Reflects all technical redesigns and new methods

---

## Changes from V3 (V4 Enhancements)

The following data-driven corrections and enhancements have been added to V4:

1. **Dataset Profiling Results** — Confirmed 2,260,668 usable rows after footer cleanup, 151 columns, date range 2007-2018 Q4
2. **Footer Row Cleanup** — 33 junk summary rows at end of CSV to be dropped during initial load
3. **Data Type Fixes** — `term` column has leading spaces; `emp_length` requires text parsing to numeric values
4. **Column Drop Lists Documented** — 14 completely empty sec_app_* columns, 15 near-empty hardship/settlement columns, plus non-feature columns (member_id, desc, url, etc.)
5. **Exact Missing Data Percentages** — Documented (e.g., mths_since_last_delinq 48.19%, mths_since_recent_revol_delinq 64.09%, etc.)
6. **Additional Data Files Identified** — benchmark_population_2014.csv (200K records) for external validation, rejected_2007_to_2018Q4.csv (27.6M) for selection bias discussion
7. **Credit Bureau Features Categorized** — ~30 features acknowledged for WOE/IV screening (account activity, balance/limit, utilization, time-based, delinquency, etc.)
8. **Enhanced LGD Formula** — Added `recoveries` and `collection_recovery_fee` to LGD calculation for more accuracy
9. **External Benchmark Validation** — benchmark_population_2014.csv usage documented for PSI computation and external calibration
10. **Known Data Quirks** — Comprehensive list added to CLAUDE.md for quick reference during development

---

## Why This Approach Wins

### What LendingClub Actually Hires For

From live job postings (Sr Credit Strategy Analyst, Director of Data Science & ML, Sr Model Risk Manager):

| What They Want | How This Project Demonstrates It |
|----------------|----------------------------------|
| "Develop, implement and handle credit risk strategies involving credit underwriting, pricing, and loan amount assignment" | Scorecard with credit policy cutoff analysis; pricing optimization by grade |
| "Design A/B tests to understand risk-return tradeoffs" | Strategy analysis: approval rate vs. expected loss tradeoffs |
| "Craft automated dashboards to track KPIs around portfolio performance" | Streamlit dashboard with real-time portfolio metrics |
| "Mine loan performance data and identify pockets of underperformance" | Vintage analysis, roll-rate monitoring, root cause analysis |
| "Measure credit expansion opportunities to optimize risk-adjusted revenue" | Grade-level profitability analysis, marginal expansion scenarios |
| "End-to-end development, deployment, and performance monitoring of ML models" | Full model lifecycle: development → validation → monitoring → deployment in tool |
| "CECL, DCF, PD/LGD methodologies" | DCF-based ECL mirroring LendingClub's 10-K, dual-mode (Operational/CECL), three-view ECL (Pre-FEG/Central/Post-FEG) |
| "Macroeconomic integration and scenario analysis" | FRED integration with 6 series, scenario weighting, macro-adjusted flow rates |

### What Separates This From Every Other Student Project

Most candidates build a PD model on LendingClub data with an ROC curve and call it a day. Your project will:
- Frame models as inputs to business decisions (not the end product)
- Include a working forecasting tool that mirrors what their team actually uses, with dual operational/regulatory modes
- Show you understand the full credit risk lifecycle: origination → monitoring → forecasting → reserving, with macroeconomic context
- Demonstrate you've done this before at my prior institution with real portfolios ($18B+ mortgage, $230M+ cards)
- Add an AI layer that no other candidate will have
- Include competing risks (prepayment) which most candidates overlook
- Demonstrate data profiling diligence and handling of real-world data quirks

---

## LendingClub Business Context (From 2024 10-K)

### Portfolio Composition
- **Primary product:** Unsecured personal loans — $3.1B HFI at amortized cost (Dec 2024)
- **Secondary:** Residential mortgages ($173M), Secured consumer ($230M), Commercial ($616M)
- **All personal loans are FIXED RATE and UNSECURED** — no collateral → high LGD
- Loan grades A through G with corresponding interest rates
- 3-year and 5-year terms

### CECL / ALLL Methodology
- **DCF Approach:** NPV of expected cash flow shortfalls for each loan pool
- **Key model inputs:** Probability and timing of defaults, loss rate, recovery exposure at default, prepayment timing/amount
- **Qualitative adjustments:** Based on macroeconomic unemployment forecast from external third-party economist + management judgment
- **ALLL ratio:** 5.7% (Dec 2024), down from 6.4% (Dec 2023)
- **Gross ALLL:** $285.7M = $285.7M gross allowance − $49.0M recovery asset value = $236.7M net ALLL
- **Recovery rate implied:** $49M / $286M ≈ 17% → **LGD ≈ 83%**
- **Net charge-off ratio:** 5.8% (2024), up from 4.9% (2023)

### Credit Quality Reporting
- Evaluates by **delinquency status** (Current, 30-59 DPD, 60-89 DPD, 90+ DPD)
- Reports by **origination vintage year** — this is how they track portfolio performance
- Charge-offs: $303.6M (2024), Recoveries: $54.5M (2024)
- Provision for credit losses: $175.4M (2024)

---

## Dataset

### Data Files and Profiling

**Primary Dataset:**
- File: accepted_2007_to_2018Q4.csv (1.6 GB)
- Total rows in file: 2,260,701 (includes 33 footer/summary rows to drop)
- Usable loan records: 2,260,668
- Total columns: 151
- Date range: 2007 to 2018 Q4
- Source: wordsforthewise/lending-club from Kaggle (https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Covers Great Recession, recovery, and growth — multiple economic cycles
- Rich feature set including bureau data, payment history, geographic data

### Loan Status Distribution (Full Dataset)

| Status | Count | Action |
|--------|-------|--------|
| Fully Paid | 1,076,751 | default=0 |
| Current | 878,317 | DROP (right-censored) |
| Charged Off | 268,559 | default=1 |
| Late (31-120 days) | 21,467 | DROP |
| In Grace Period | 8,436 | DROP |
| Late (16-30 days) | 4,349 | DROP |
| Does not meet credit policy: Fully Paid | 1,988 | DROP |
| Does not meet credit policy: Charged Off | 761 | DROP |
| Default | 40 | default=1 |

**Terminal loans for modeling:** 1,345,350 (268,599 defaults = 19.96% default rate)

### Additional Data Files

- **rejected_2007_to_2018Q4.csv**: 27.6M rejected applications (9 columns) — use for selection bias discussion. NOT used for modeling (no outcome variable). Interview talking point: "I acknowledged selection bias from modeling only approved loans. In production, reject inference techniques could address this, but the analytical focus here is on portfolio management and loss forecasting for the existing book."

- **benchmark_population_2014.csv**: 200,000 records with FICO scores, delinquency buckets (CURRENT, 30_DPD, 60_DPD), PERFORMANCE_OUTCOME (GOOD/BAD). Period: JUN-AUG 2014. Use for: PSI computation (compare model population to benchmark), external calibration check. Score your model on 2014 benchmark population and compare predicted PD distribution to actual outcomes. This mirrors a benchmark population validation approach from my prior role where we compared model outputs to known performance cohorts.

- **LCDataDictionary.xlsx**: Official variable definitions (3 sheets: LoanStats, browseNotes, RejectStats)

---

## Project Architecture

```
lending-club-credit-risk/
│
├── CLAUDE.md                          # Context file for Claude Code sessions
├── README.md                          # Project overview, results, and methodology
├── requirements.txt                   # Python dependencies (pinned versions)
├── config.py                          # Configuration constants and paths
├── .gitignore                         # data/, *.pkl, *.parquet, etc.
│
├── data/
│   ├── raw/                           # Original dataset (gitignored)
│   ├── processed/                     # Cleaned datasets, feature-engineered data
│   ├── models/                        # Pickled models, scorecard objects
│   └── results/                       # Metrics JSONs, ECL summaries, validation reports
│
├── notebooks/
│   ├── 01_EDA_and_Data_Cleaning.ipynb
│   ├── 02_WOE_IV_Feature_Engineering.ipynb
│   ├── 03_PD_Model_Scorecard.ipynb
│   ├── 04_PD_Model_ML_Ensemble.ipynb
│   ├── 05_EAD_Model.ipynb
│   ├── 06_LGD_Model.ipynb
│   ├── 07_ECL_Computation_and_Vintage_Analysis.ipynb
│   ├── 08_Model_Validation_and_Monitoring.ipynb
│   └── 09_Macro_Scenario_and_Strategy_Analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py             # Data cleaning, preprocessing pipeline
│   ├── woe_binning.py                 # WOE/IV computation engine
│   ├── scorecard.py                   # Scorecard development and scoring
│   ├── models.py                      # PD, EAD, LGD model classes
│   ├── ecl_engine.py                  # ECL computation (simple + DCF)
│   ├── flow_rates.py                  # Flow rate computation and receivables tracker
│   ├── ecl_projector.py               # Dual-mode forecasting engine (extend vs. cecl)
│   ├── macro_scenarios.py             # FRED integration, scenario weighting, macro adjustments
│   ├── validation.py                  # Gini, KS, PSI, CSI, VDI, backtesting
│   └── visualization.py              # Plotting utilities
│
├── app/
│   ├── streamlit_app.py               # Main Streamlit application
│   ├── pages/
│   │   ├── 01_portfolio_overview.py
│   │   ├── 02_roll_rate_analysis.py
│   │   ├── 03_vintage_performance.py
│   │   ├── 04_ecl_forecasting.py      # Dual-mode toggle here (Operational vs. CECL)
│   │   ├── 05_scenario_analysis.py
│   │   ├── 06_model_monitoring.py
│   │   └── 07_ai_analyst.py           # Claude-powered chatbot
│   ├── components/
│   │   ├── charts.py                  # Reusable chart components
│   │   ├── tables.py                  # Formatted table components
│   │   └── chatbot.py                 # AI chatbot interface
│   └── utils/
│       ├── data_loader.py             # Load and cache data for Streamlit
│       └── session_state.py           # Manage Streamlit session state
│
├── reports/
│   └── presentation.pdf               # Final presentation deck
│
└── tests/
    ├── test_woe.py
    ├── test_ecl.py
    └── test_flow_rates.py
```

---

## CLAUDE.md File (For Claude Code Context)

Save this at your project root. Claude Code reads it automatically:

```markdown
# LendingClub Credit Risk Analytics Project

## Project Context
Building a production-grade credit risk analytics project targeting a Credit Risk Analyst
role at LendingClub (Model Development / Loss Forecasting / Strategy teams).

## Key Technical Decisions

### Data & Target Definition
- **Target Variable:** default=1 for Charged Off/Default, default=0 for Fully Paid.
  DROP all right-censored statuses (Current, In Grace Period, Late).
- **Train/Test Split:** Time-based. Train: 2007-2015, Validation: 2016, Test: 2017-2018. No random splitting.
- **Macroeconomic Data:** Pull 6 FRED series by origination month (issue_d):
  - UNRATE (unemployment rate)
  - CSUSHPINSA (Case-Shiller Home Price Index)
  - A191RL1Q225SBEA (Real GDP growth, quarterly)
  - CPIAUCSL (CPI)
  - DFF (Federal Funds Rate)
  - UMCSENT (University of Michigan Consumer Sentiment)
  Merge onto loan data by origination month. Start with origination-time macro merge (mandatory);
  performance-period merge as optional enhancement. CRITICAL for time-based split to work correctly —
  without macro covariates, the model would overpredict defaults on 2016-2018 test data because it
  learned recession-era rates as structural truths.

### Known Data Quirks (Confirmed from Full-File Profiling — All 2,260,701 Rows)
- CSV has 33 footer/summary rows at end — drop rows where loan_amnt is null (leaves 2,260,668 usable)
- `term` has leading spaces: ' 36 months' — use .str.strip() then extract int
- `emp_length` is text: '10+ years', '< 1 year', etc. — parse to numeric (11 unique values)
- Dates are text format 'MMM-YYYY' (e.g., 'Dec-2015') — use format='%b-%Y'
- `int_rate` and `revol_util` are already float64 (no % stripping needed)
- `member_id` is 100% empty — drop immediately
- `desc` is 94.42% empty (~126K records have content, not ~40) — drop for PD modeling
- 13 sec_app_* columns + revol_bal_joint are ~95.22% empty (108K joint application records exist; not 100% empty as originally estimated from sampling)
- ~15 hardship/settlement columns are >97% empty — drop immediately
- `dti_joint` (94.66%), `annual_inc_joint` (94.66%), `verification_status_joint` (94.88%) — drop
- loan_status has 9 unique values including 'Does not meet credit policy' variants
- Terminal statuses for modeling: Fully Paid (1,076,751), Charged Off (268,559), Default (40)
- Default rate after filtering: ~19.96%
- `recoveries` and `collection_recovery_fee` exist — use for accurate LGD
- benchmark_population_2014.csv available for external validation (200K records, JUN-AUG 2014)

#### Confirmed Outlier Flags (from Full-File Numeric Ranges)
- `annual_inc`: max = $110,000,000 — extreme outlier, cap at 99th percentile
- `dti`: range [-1, 999] — negative DTI and 999 are data entry errors, cap/clean
- `revol_util`: max = 892.3 — values >100% exist (over-limit accounts), investigate and cap
- `revol_bal`: max = $2,904,836 — extreme outlier
- `tot_coll_amt`: max = $9,152,545 — extreme outlier
- `tot_cur_bal`: max = $9,971,659 — extreme outlier
- `bc_util`: max = 339.6 — over 100%, investigate
- `il_util`: max = 1,000 — extreme outlier, likely sentinel value
- `all_util`: max = 239 — over 100%
- `delinq_amnt`: max = $249,925 — extreme outlier
- `tot_hi_cred_lim`: max = $9,999,999 — possibly capped/sentinel value
- `total_rev_hi_lim`: max = $9,999,999 — possibly capped/sentinel value
- `last_fico_range_high/low`: min = 0 — invalid FICO, needs handling (set to NaN or investigate)
- `settlement_percentage`: max = 521.35 — above 100%, data quality issue
- `policy_code`: all values = 1.0 — constant, zero variance, drop
- `pymnt_plan`: all values = 'n' — constant, zero variance, drop

### PD Model
- **Scorecard:** Logistic regression with L2 (Ridge) regularization on WOE-transformed
  features. L2 preferred because all features are pre-selected with IV>0.1 — we want all
  features contributing with stable coefficients, not sparse elimination.
- **Macro Covariates:** PD model MUST include macro features (unemployment, GDP, HPI, etc.)
  alongside borrower characteristics. This is REQUIRED for the chronological out-of-time split
  to generalize across economic regimes. Basel requires full economic cycle coverage; macro
  features carry the cycle information. Without them: "FICO 680 defaults at 8%" (learned from
  recession). With them: "FICO 680 defaults at 8% WHEN unemployment is 10%, but at 3% WHEN
  unemployment is 4%". Interview framing: "included macroeconomic covariates so the model
  learns the conditional relationship between borrower risk and the economic environment."
- **Scorecard Format:** 600 base score, PDO=20 (points to double odds).
- **ML Models:** XGBoost and LightGBM on original features with SHAP analysis. Macro features
  also included as covariates.
- **Conditional PD:** Monthly conditional PDs vary by borrower characteristics, macroeconomic
  scenario, and loan age/MOB. These feed directly into DCF-ECL calculation. Under different
  macro scenarios, same borrower gets different PD paths → different ECL.

### EAD & Prepayment
- **EAD:** Use `out_prncp` (outstanding principal) for defaulted loans. NOT `revol_bal`.
  For term loans, EAD ≈ outstanding balance at time of default.
- **Prepayment Model:** Build empirical prepayment rates from LendingClub data using issue_d,
  last_pymnt_d, loan_status. Identify prepayments as Fully Paid loans with life significantly
  shorter than contractual term. Competing risks: three outcomes (default, prepay, maturity).
  Prepayment rates feed into DCF-ECL calculation.

### LGD Model
- **Two-Step Model:**
  - Step 1 (Classification Phase): Binary classifier for any recovery (recovery_flag = 1 if recoveries > 0)
  - Step 2 (Regression Phase): For loans with recovery > 0, predict recovery rate using Beta regression
  - LGD = 1 - (P(recovery) × E[recovery_rate | recovery > 0])
- **More accurate LGD formula:** LGD = 1 - ((recoveries - collection_recovery_fee) / EAD)
  - `recoveries`: post-charge-off cash recovered (100% populated for all loans)
  - `collection_recovery_fee`: fee paid to recovery agent (100% populated)
  - Net recovery = recoveries - collection_recovery_fee
  - This is more accurate than using total_rec_prncp alone, which includes pre-default principal payments
- **Cross-check formula:** LGD_simple = 1 - (total_rec_prncp / EAD)
- **Target LGD:** ≈ 83% based on LendingClub 10-K ($49M recovery asset / $286M gross ALLL).
- **IMPORTANT NAMING:** Use "Step 1" and "Step 2", NOT "Stage 1" and "Stage 2" to avoid
  confusion with IFRS-9 Stages 1/2/3 which are a completely separate concept.

### ECL & Forecasting

#### ECLProjector Class Design
- **Constructor:** Takes only `pd_model` and `lgd_model` (no flow_rates dict parameter)
- **New Method:** `compute_forecast_flow_rates(lookback_months=6)` computes 6-month rolling
  average from historical flow rates in receivables tracker
- **Process:** load_receivables() → compute_forecast_flow_rates() → apply_assumptions() → project()
- Flow rates NOT a constructor parameter; derived from data

#### Dual-Mode Forecasting Engine
Two methods for projection engine:

1. **method='extend' (PyCraft style, OPERATIONAL MODE):**
   - Take 6-month rolling average of historical flow rates
   - Extend flat across entire projection horizon (10 years)
   - Simple, operationally practical
   - Used for AOP (Annual Operating Plan) / FRP (Financial Resource Planning) / internal planning
   - Assumes flow rates stabilize at recent average

2. **method='cecl' (ASC 326 COMPLIANT, REGULATORY MODE):**
   - Phase 1 (R&S period, default 24 months): macro-adjusted flow rates
     Apply scenario-specific adjustments to each flow rate based on macro forecasts
   - Phase 2 (Reversion, default 12 months): straight-line transition from Phase 1 rates
     to long-run historical averages
   - Phase 3 (Remaining horizon): pure historical averages, NO adjustment for current/future conditions
   - Preserves ASC 326 requirement that model reflects expected conditions but reverts
     to historical baseline for extended horizon

#### Flow Through Rate
- **Definition:** Flow Through Rate (Current → GCO) = product of all intermediate flow rates
  e.g., 0.028 × 0.382 × 0.701 × 0.85 × 0.90 × 0.92 × 0.95 = 0.468%
- **Interpretation:** For every $100 in Current, approximately $0.47 will eventually charge off
- **Output:** Display on Streamlit dashboard as KPI metric
- **Cross-check:** Use as validation against PD model outputs
- **Monitoring:** Track trend over time — early warning signal if trending up

#### Three ECL Views (Pre-FEG / Central / Post-FEG)
- **Pre-FEG:** Pure model output — 6-month rolling avg flow rates, no macro overlay
- **Central (FEG):** Baseline macro scenario applied to flow rates
- **Post-FEG:** Weighted average across all scenarios (Baseline 60%, Mild Downturn 25%, Stress 15%)
  plus qualitative adjustments
- **Streamlit Toggle:** Radio buttons on ECL page; charts/tables/reserve numbers update accordingly
- **Shows Impact:** Demonstrates effect of macro overlay and scenario weighting on final ECL
- **IMPORTANT (NEW V5):** The Pre-FEG/Central/Post-FEG toggle is ORTHOGONAL to the forecasting mode (Operational vs. CECL). Both modes support all three FEG views. The difference: in Operational mode, stressed flow rates extend flat across the full horizon; in CECL mode, stressed rates only apply during Phase 1 (R&S period) and then revert during Phase 2/3. This is a great interview point: operational stress gives worst-case long-run trajectory; CECL stress is more conservative because it reverts.

#### Simple ECL Baseline
- ECL = PD × EAD × LGD (point-in-time, aggregated by segment)

#### DCF-Based ECL (mirrors LendingClub 10-K)
- For each loan pool (by grade), project monthly cash flows over remaining life
- Apply monthly marginal PD (conditional on macro scenario) to determine timing of defaults
- Apply prepayment assumptions and competing risks
- Discount expected cash flows at effective interest rate
- ECL = Contractual cash flows (NPV) - Expected cash flows (NPV)
- **IMPORTANT (NEW V5):** The `recovery_rate` assumption in `set_assumptions()` should DEFAULT to the portfolio-level output of the LGD model (if LGD model estimates ~83% average LGD → default recovery_rate = 17%). The slider allows override for sensitivity testing. This connects the LGD model (loan-level) to the flow-rate projection engine (portfolio-level).

### Stress Scenarios
- **Stress Application Level:** Adjust individual flow rates (e.g., increase each by 15%),
  NOT applied as a multiplier on final ECL output.
- **Why:** Compounding through the waterfall is multiplicative. A 15% stress on each flow rate
  produces ~75% increase in cumulative flow-through (because 1.15^7 ≈ 2.66), vs. only 15% increase
  if applied to ECL output. Also changes loss timing curve shape (losses accelerate and concentrate
  under stress). This preserves non-linear dynamics of delinquency behavior.
- **IMPORTANT (NEW V5):** Stress scenarios are independent of forecasting mode. In Operational mode, stressed flow rates extend flat across the full horizon for worst-case long-run trajectory. In CECL mode, stressed flow rates apply during Phase 1 only and then revert, providing more conservative but realistic reserving under extended time horizons.

### Liquidation Factor Design
Two modes:

1. **Operational Mode (default):** Single portfolio-level liquidation factor, set via Streamlit slider.
   Simple, matches PyCraft. Example: "3.2% monthly liquidation" applied flat.
   - **IMPORTANT (NEW V5):** In Operational mode, both Liquidation Factor and New Originations inputs are shown to the user.

2. **CECL Mode:** Differentiated by loan term (36 vs 60 months) and vintage age. Empirical
   paydown curves computed from LendingClub data. Interview framing: "PyCraft used portfolio-level
   because it was operational planning. I also built CECL-aligned version where prepayment rates
   vary by term and vintage."
   - **IMPORTANT (NEW V5):** In CECL mode, only Liquidation Factor is shown; New Originations is hidden (hardcoded to $0) because CECL reserves are calculated on the existing portfolio only — new loans get their own Day 1 CECL assessment at origination.

### Assumption Input UI Design
- **Streamlit Frontend:** PRIMARY interface with sliders, inputs, dropdowns for interactive exploration
- **"Upload Assumptions" Button:** Accepts Excel in institutional format for operational workflow compatibility
- **"Export Assumptions" Button:** Downloads current settings as Excel for audit trail/documentation
- **Best of Both Worlds:** Interactive for exploration, Excel-compatible for operations

## Validation Metrics
- AUC, Gini, KS, PSI (<0.1 green, 0.1-0.25 amber, >0.25 red),
  CSI, VDI, Hosmer-Lemeshow, calibration plots, out-of-time validation.

## External Benchmark Validation (NEW V4)
- **File:** benchmark_population_2014.csv (200,000 records)
- **Contains:** FICO scores, delinquency buckets (CURRENT, 30_DPD, 60_DPD), PERFORMANCE_OUTCOME (GOOD/BAD)
- **Period:** JUN-AUG 2014
- **Use for:** PSI computation (compare model population to benchmark), external calibration check
- **Approach:** Score your model on 2014 benchmark population and compare predicted PD distribution to actual outcomes
- **Interview framing:** "This mirrors a benchmark population validation approach from my prior role where we compared model outputs to known performance cohorts"

## Coding Standards
- All notebooks should save outputs to data/processed/ or data/results/
- Use src/ modules for reusable logic — notebooks should call functions, not define them
- Every notebook starts with a markdown cell explaining purpose and prior role experience connection
- Use consistent random_state=42 everywhere
- Pin all package versions in requirements.txt
- Type hints on all function signatures
- Docstrings on all public functions

## Prior Role Experience Connections (reference in notebook markdown)
- WOE/IV → Credit card behavioral scorecard monitoring Q4'22
- Scorecard + RAG → Behavioral scorecard RAG status framework
- Macro Integration → FEG scenarios with GDP/HPI/unemployment weights
- Validation (Gini/PSI/CSI/VDI) → Quarterly model monitoring
- Vintage Analysis → Sherwood PD curves by mortgage product × MOB
- Roll-Rate / Flow-Rate Analysis → Loss forecasting receivables tracker (flow rates computed as simple bucket ratios, not account-level transition matrices)
- Prepayment Model → Empirical prepayment curves from historical data
- ECL/CECL → ALLL tracker, Pre-FEG/Central/Post-FEG computation
- Forecasting Tool → PyCraft (Django-based loss forecasting tool) with dual operational/regulatory modes
- Liquidation Factors → Portfolio-level (operational) vs. differentiated by term/vintage (CECL)
- External Benchmark Validation → Benchmark population approach for PSI and calibration

## Dependencies
pandas, numpy, scikit-learn, xgboost, lightgbm, optbinning, shap,
matplotlib, seaborn, plotly, scipy, statsmodels, fredapi, optuna,
streamlit, anthropic, jupyter, lifelines>=0.28 (for survival analysis/competing risks)
```

---

## Detailed 3-Week Roadmap

### WEEK 1: Data Foundation + PD Models (Days 1-7)

#### Days 1-2: Data Acquisition, Cleaning & EDA

**Notebook: `01_EDA_and_Data_Cleaning.ipynb`**

**Prior Role Connection:** This mirrors the data preparation phase of both the behavioral scorecard monitoring (pulling data from SAS, segmenting by delinquency bucket) and the loss forecasting receivables tracker (pulling from CWOD, aggregating by portfolio and vintage).

1. **Load the full dataset** (~2.26M records, 151 features)
   - File: accepted_2007_to_2018Q4.csv (1.6 GB)
   - Total rows in file: 2,260,701 (includes 33 footer/summary rows)

2. **Footer Row Removal (NEW V4):**
   - After loading CSV, drop rows where `loan_amnt` is null or non-numeric
   - The raw file has 33 summary/footer rows at the end that are not loan records
   - This leaves exactly 2,260,668 usable loan records

3. **Define target variable:**
   - `default = 1` if `loan_status` in ['Charged Off', 'Default']
   - `default = 0` if `loan_status` in ['Fully Paid']
   - **DROP** all other statuses ('Current', 'In Grace Period', 'Late (16-30)', 'Late (31-120)', 'Does not meet credit policy: Fully Paid', 'Does not meet credit policy: Charged Off') — right-censored observations that bias PD downward
   - Terminal loans for modeling: 1,345,350 (268,599 defaults = 19.96% default rate)

4. **Immediate Column Drops (NEW V4 - from profiling):**

   a) **14 completely empty sec_app_* columns (100% null):**
      - sec_app_fico_range_low, sec_app_fico_range_high, sec_app_earliest_cr_line, sec_app_inq_last_6mths, sec_app_mort_acc, sec_app_open_acc, sec_app_revol_util, sec_app_open_act_il, sec_app_num_rev_accts, sec_app_chargeoff_within_12_mths, sec_app_collections_12_mths_ex_med, sec_app_mths_since_last_major_derog, revol_bal_joint, member_id

   b) **15 near-empty hardship/settlement columns (>97% null):**
      - hardship_type, hardship_reason, hardship_status, deferral_term, hardship_amount, hardship_start_date, hardship_end_date, payment_plan_start_date, hardship_length, hardship_dpd, hardship_loan_status, hardship_payoff_balance_amount, hardship_last_payment_amount, orig_projected_additional_accrued_interest, debt_settlement_flag_date, settlement_status, settlement_date, settlement_amount, settlement_percentage, settlement_term

   c) **Non-feature columns:**
      - id, url, desc (94.42% empty), pymnt_plan (all 'n'), policy_code (all 1)

5. **Data Type Fixes (NEW V4 - exact details):**

   a) **`term` column:** Strip leading spaces → ' 36 months' becomes '36 months'; then extract integer (36 or 60)

   b) **`emp_length` column:** Parse text to numeric
      - '10+ years' → 10
      - '< 1 year' → 0
      - '2 years' → 2, etc.
      - NaN stays as NaN (create emp_length_unknown flag)

   c) **Date columns:** Parse from 'MMM-YYYY' text to datetime
      - `issue_d`, `earliest_cr_line`, `last_pymnt_d`
      - Use: `pd.to_datetime(col, format='%b-%Y')`

   d) **`int_rate` and `revol_util`:** Already float64 — NO % stripping needed (confirmed from profiling)

6. **Missingness-based drops (>40% missing) (NEW V4 - exact percentages):**
   - mths_since_last_delinq (48.19% missing — but DON'T drop: create no_delinq_history flag)
   - mths_since_recent_revol_delinq (64.09%)
   - mths_since_last_major_derog (70.63%)
   - mths_since_last_record (82.20%)
   - il_util (47.28%), mths_since_rcnt_il (40.25%)
   - dti_joint (94.66%), annual_inc_joint (94.66%), verification_status_joint (94.88%)
   - NOTE: open_act_il, open_il_12m, open_il_24m, total_bal_il, open_acc_6m, open_rv_12m,
     open_rv_24m, max_bal_bc, all_util, inq_fi, total_cu_tl, inq_last_12m are ~38.31% missing
     (NOT 78-81% as estimated from 100K sample). These have 1.39M populated records — keep
     with missing flag + imputation, send through WOE/IV for feature selection.

7. **Handle missing values with domain logic:**
   - Tier 2 (70-85% missing): create binary flag, then drop original column
   - Tier 3 (38-68% missing): create binary flag AND keep with median imputation for WOE/IV
   - Tier 4 (<10%): standard imputation
   - `mths_since_last_delinq` missing → encode as "no delinquency" flag (not median fill)
   - `emp_length` missing → create "unknown" category
   - `annual_inc` missing → investigate and drop if very few

8. **Feature categorization (NEW V4 - ~30 credit bureau features acknowledged):**
   - Borrower: `annual_inc`, `emp_length`, `home_ownership`, `verification_status`
   - Credit history: `fico_range_low/high`, `earliest_cr_line`, `open_acc`, `total_acc`, `revol_util`, `revol_bal`, `pub_rec`, `delinq_2yrs`, `inq_last_6mths`, `mths_since_last_delinq`
   - Account activity: `num_actv_bc_tl`, `num_actv_rev_tl`, `num_bc_sats`, `num_bc_tl`, `num_il_tl`, `num_op_rev_tl`, `num_rev_accts`, `num_rev_tl_bal_gt_0`, `num_sats`, `num_tl_op_past_12m`, `acc_open_past_24mths`
   - Balance/limit: `tot_cur_bal`, `tot_hi_cred_lim`, `total_bal_ex_mort`, `total_bc_limit`, `total_il_high_credit_limit`, `avg_cur_bal`, `bc_open_to_buy`, `max_bal_bc`
   - Utilization: `bc_util`, `all_util`, `percent_bc_gt_75`
   - Time-based: `mo_sin_old_il_acct`, `mo_sin_old_rev_tl_op`, `mo_sin_rcnt_rev_tl_op`, `mo_sin_rcnt_tl`, `mths_since_recent_bc`, `mths_since_recent_inq`
   - Delinquency: `num_accts_ever_120_pd`, `num_tl_30dpd`, `num_tl_90g_dpd_24m`, `pct_tl_nvr_dlq`
   - Other: `mort_acc`, `pub_rec_bankruptcies`, `tax_liens`, `inq_last_12m`, `total_cu_tl`
   - Loan: `loan_amnt`, `term`, `int_rate`, `grade`, `sub_grade`, `purpose`, `installment`
   - Geographic: `addr_state`, `zip_code`

9. **Merge macroeconomic data from FRED API (CRITICAL FOR TIME-BASED SPLIT):**
   - Pull 6 FRED series by origination month (issue_d):
     - **UNRATE**: Unemployment rate (monthly)
     - **CSUSHPINSA**: Case-Shiller Home Price Index (monthly, not seasonally adjusted)
     - **A191RL1Q225SBEA**: Real GDP growth rate (quarterly; map to monthly by carrying forward)
     - **CPIAUCSL**: Consumer Price Index (monthly)
     - **DFF**: Effective Federal Funds Rate (monthly)
     - **UMCSENT**: University of Michigan Consumer Sentiment Index (monthly)
   - Merge onto loan data by origination month using `issue_d`
   - Handle quarterly (GDP) data by carrying forward values within each quarter
   - Start with origination-time macro merge (mandatory); performance-period merge as optional enhancement
   - **WHY CRITICAL:** Without macro covariates, the model would overpredict defaults on 2016-2018 test data because it learned recession-era rates as structural truths. Time-based split requires macro context.
   - Add these macro features to output parquet files for use in subsequent notebooks

10. **EDA deliverables:**
    - Default rate by grade/sub-grade (should be monotonically increasing A→G)
    - Default rate by origination vintage year (mirrors LendingClub 10-K vintage reporting)
    - Default rate by term (36 vs 60 months)
    - Distribution of key features by default status
    - Correlation matrix (including macro variables)
    - Geographic default rate heatmap by state
    - Portfolio composition over time (volume by grade, by purpose)
    - Macro variable trends over time (unemployment, HPI, GDP) — context for defaults

11. **Create time-based split:**
    - Train: 2007-2015 (issue_d)
    - Validation: 2016
    - Test: 2017-2018

12. **Save processed data:** `data/processed/loans_cleaned.parquet` (includes macro features)

**Output files:** `loans_cleaned.parquet` (with macro features), EDA summary statistics JSON, FRED data cache

---

#### Days 3-4: WOE/IV Analysis & Feature Engineering

**Notebook: `02_WOE_IV_Feature_Engineering.ipynb`**

**Prior Role Connection:** Directly mirrors Project #1 — behavioral scorecard monitoring where you computed WOE, IV, CSI, VDI for credit card portfolio variables (VantageScore/FICO bins, utilization ratio, DTI, inquiries, open tradelines, months on book, delinquencies).

1. **Build WOE binning engine (`src/woe_binning.py`):**
   - Use `optbinning` library for optimal binning (decision tree-based)
   - For continuous variables: find optimal bins that maximize IV
   - For categorical variables: group small categories
   - Compute per bin: event count, non-event count, event rate, WoE, IV
   - WoE = ln(Distribution of Events / Distribution of Non-Events)
   - IV = Σ (Dist_Events - Dist_NonEvents) × WoE

2. **IV-based feature selection:**
   - IV < 0.02: Not predictive → drop
   - IV 0.02-0.1: Weak → consider dropping
   - IV 0.1-0.3: Medium → include
   - IV 0.3-0.5: Strong → include
   - IV > 0.5: Suspicious → investigate for data leakage (e.g., `int_rate` and `grade` will have very high IV because they're assigned based on credit risk — decide whether to include)
   - **IMPORTANT:** Include macro variables (UNRATE, HPI, GDP, etc.) in WOE binning. These will have high IV because they correlate strongly with defaults, and that's exactly what we want for scenario-conditional PD models.

3. **Feature engineering:**
   - `credit_history_years` = years since `earliest_cr_line`
   - `fico_avg` = (fico_range_low + fico_range_high) / 2
   - `loan_to_income` = loan_amnt / annual_inc
   - `installment_to_income` = installment / (annual_inc / 12)
   - `total_credit_utilization` = revol_bal / total_rev_hi_lim
   - `delinq_flag` = 1 if delinq_2yrs > 0
   - `recent_inquiry_flag` = 1 if inq_last_6mths > 2
   - `high_dti_flag` = 1 if dti > 30
   - `no_delinq_history` = 1 if mths_since_last_delinq is missing

4. **Validate monotonicity:** Bad rate should increase/decrease monotonically across WoE bins. Flag any rank-ordering breaks (just as you flagged in the RAG framework at my prior institution).

5. **Generate WOE/IV summary table** for all features — this becomes a key deliverable.

**Output files:** `woe_binning_results.pkl`, `iv_summary.csv`, `loans_woe_transformed.parquet` (includes macro features WOE-transformed)

---

#### Days 5-7: PD Models — Scorecard + ML Ensemble

**Notebook: `03_PD_Model_Scorecard.ipynb`**

**Prior Role Connection:** This is the culmination of your behavioral scorecard experience — but now you're building the scorecard from scratch rather than monitoring an existing one.

1. **Logistic regression scorecard with WOE features:**
   - Input: WOE-transformed features (only those with IV > 0.1), **including macro features**
   - L2 regularization (Ridge) — rationale: features are pre-selected and economically justified; we want stable coefficients keeping all features, not sparse elimination
   - Use sklearn's LogisticRegression with penalty='l2', tune C via cross-validation
   - Verify all coefficients are negative (higher WoE = lower risk = lower PD)
   - **MACRO INTEGRATION:** Include unemployment, HPI, GDP as covariates. Their coefficients will show how PD changes with economic regime.

2. **Convert to scorecard points:**
   - Base score: 600, PDO (Points to Double Odds): 20
   - Score = Offset + Factor × Σ(βi × WoEi)
   - Factor = PDO / ln(2)
   - Offset = Base_Score - Factor × ln(Base_Odds)
   - Generate scorecard table: Feature → Bin → WoE → Points
   - **Include macro feature point contributions** so you can show "in recession, same borrower scores 15 points lower"

3. **Credit policy analysis (THE STRATEGY LAYER):**
   - Plot score distribution for good vs. bad accounts
   - For different score cutoffs, compute: approval rate, expected default rate, expected loss rate
   - Build an **approval rate vs. expected loss tradeoff curve**
   - Identify the optimal cutoff that maximizes risk-adjusted return
   - This is what LendingClub's Credit Strategy team does daily

4. **Grade mapping:**
   - Map score ranges to LendingClub grades A-G
   - Compare your model's grade assignment with actual grades in data
   - Analyze mis-grades (accounts your model would grade differently)

5. **Scenario-conditional PD:**
   - Generate PD estimates under different macro scenarios (Baseline, Mild Downturn, Stress)
   - Show same borrower's PD under different unemployment levels
   - Frame: "the scorecard is agnostic to macro conditions; it scores 620 regardless. But when we input macro features into the PD model, the same borrower has 3% PD in Baseline but 5.2% in Stress scenario."

6. **RAG status framework:**
   - Define thresholds: Gini ≥ 60% (Green), 50-60% (Amber), <50% (Red)
   - Compute Gini on train, validation, and test separately
   - Report RAG status per time period — mirrors your quarterly monitoring at my prior institution

**Metrics targets:** AUC ≥ 0.75, Gini ≥ 55%, KS ≥ 30%

**Output files:** `pd_logreg_model.pkl`, `scorecard_table.csv`, `pd_scorecard_metrics.json`, `conditional_pd_by_scenario.csv`

---

**Notebook: `04_PD_Model_ML_Ensemble.ipynb`**

1. **XGBoost model:**
   - Input: original features (not WOE-transformed) — XGBoost handles non-linear relationships natively
   - **Include macro features as original columns** (UNRATE, HPI, GDP, etc.)
   - Hyperparameter tuning via Optuna (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_alpha, reg_lambda)
   - SHAP analysis: global feature importance + individual prediction explanations
   - Show macro variables in top SHAP features

2. **LightGBM model:**
   - Same feature set as XGBoost (including macro features)
   - Compare training speed and performance

3. **Model comparison table:**

   | Metric | LogReg Scorecard | XGBoost | LightGBM |
   |--------|-----------------|---------|----------|
   | AUC (Train) | | | |
   | AUC (Test) | | | |
   | Gini (Test) | | | |
   | KS Statistic | | | |
   | Overfit Gap | | | |

4. **Scenario sensitivity analysis:**
   - Generate PD predictions under Baseline, Mild Downturn, Stress macro scenarios using XGBoost
   - Show how predictions shift as unemployment/GDP change
   - Compare to scorecard's scenario-conditional output

5. **Discussion:** Why would LendingClub use the scorecard for production credit decisions despite lower AUC? (Interpretability, regulatory requirements, monotonicity constraints, model governance — connect to your prior institution experience with model review by Independent Model Review Team and OCC submission.)

**Metrics targets:** XGBoost AUC ≥ 0.80, KS ≥ 35%

**Output files:** `pd_xgboost_model.pkl`, `pd_lgbm_model.pkl`, `shap_values.pkl`, `model_comparison.json`, `xgb_scenario_predictions.csv`

---

### WEEK 2: EAD/LGD/ECL + Validation + Strategy (Days 8-14)

#### Days 8-9: EAD and LGD Models

**Notebook: `05_EAD_Model.ipynb`**

**Prior Role Connection:** At my prior institution, EAD was assumed to be 1 (100% of outstanding balance) for mortgages because they're fully drawn term loans. Similarly, LendingClub's personal loans are fully drawn at origination — but the outstanding balance declines with amortization.

1. **For term loans (LendingClub's case):**
   - EAD = outstanding principal at time of default
   - Use `out_prncp` from the dataset for defaulted loans
   - Compute Credit Conversion Factor: CCF = out_prncp / funded_amnt
   - Build Random Forest / Gradient Boosting regressor to predict EAD given loan characteristics
   - Compare to analytical amortization formula (can compute expected balance at any month given rate, term, payment)

2. **Key insight:** For fully-drawn term loans, EAD is much simpler than for revolving credit (credit cards) where you need to model undrawn commitments. This should be stated explicitly — it shows you understand the difference.

**Metrics targets:** MAE/MAPE < 15%, R² > 0.70

---

**Notebook: `06_LGD_Model.ipynb`**

**Prior Role Connection:** LGD model at my prior institution used Basel III assumptions. At LendingClub, recoveries are tracked explicitly. The 10-K shows $49M recovery asset value against $286M gross ALLL → ~17% recovery rate → LGD ≈ 83%.

1. **Two-step LGD model (IMPORTANT NAMING CHANGE V3):**
   - **Step 1 (Classification Phase):** Logistic regression — did any recovery occur? (recovery_flag = 1 if recoveries > 0)
   - **Step 2 (Regression Phase):** For loans with recovery > 0, predict recovery rate using Beta regression
   - LGD = 1 - (P(recovery) × E[recovery_rate | recovery > 0])

2. **More accurate LGD formula (NEW V4):**
   - **Primary formula:** LGD = 1 - ((recoveries - collection_recovery_fee) / EAD)
     - `recoveries`: post-charge-off cash recovered (100% populated for all loans)
     - `collection_recovery_fee`: fee paid to recovery agent (100% populated)
     - Net recovery = recoveries - collection_recovery_fee
     - This is more accurate than using total_rec_prncp alone, which includes pre-default principal payments
   - **Cross-check formula:** LGD_simple = 1 - (total_rec_prncp / EAD)

3. **Validate against 10-K:** Your portfolio-level average LGD should be approximately 83% (± some variance since your dataset is 2007-2018 and 10-K reports 2024).

4. **LGD by grade:** Higher-risk grades should show higher LGD (lower recovery). Validate this.

**Metrics targets:** Binary step AUC > 0.65, Overall LGD MAE < 0.10

---

#### Days 10-11: ECL Computation, Vintage Analysis, Prepayment Model & Roll Rates

**Notebook: `07_ECL_Computation_and_Vintage_Analysis.ipynb`**

**Prior Role Connection:** This is the core of your Loss Forecasting team experience. The receivables tracker, ALLL computation, vintage analysis, and roll-rate matrices are exactly what you built at my prior institution for $18B+ in mortgage portfolios. NEW IN V3: Prepayment model and competing risks framework.

1. **Prepayment Model (NEW in V3 - add before ECL section):**
   - **Goal:** Model competing risks — loans can default, prepay, or reach maturity
   - **Data:** Use `issue_d` (origination), `last_pymnt_d` (last payment), `loan_status`, `term` (36 or 60 months)
   - **Prepayment Identification:** Fully Paid loans with actual life < contractual term
     - Example: 60-month term loan that was Fully Paid in month 48 = prepayment
   - **Empirical Prepayment Rates:** Compute by term (36 vs 60 months) and vintage year
   - **Competing Risks Framework:**
     - Three outcomes: Default (before maturity), Prepay (early full repayment), Maturity (reach end of term and pay off)
     - Quarterly prepayment rates feed into DCF-ECL as alternative cash flow scenario
   - **Integration into ECL:** For each loan, survival analysis considers both default and prepayment hazards
   - **Interview Framing:** "LendingClub has significant prepayment risk; ignoring it would overstate expected loss. I built a competing risks model to separately estimate default vs prepay rates."

2. **Simple ECL baseline:**
   - ECL = PD × EAD × LGD (point-in-time, aggregated by segment)
   - Compute by grade (A-G), by vintage year, by purpose
   - Portfolio-level ALLL ratio = total ECL / total outstanding balance
   - Compare to LendingClub 10-K ALLL ratio (5.7%)

3. **DCF-based ECL (mirrors LendingClub 10-K):**
   - For each loan pool (by grade), project monthly cash flows over remaining life
   - Apply monthly marginal PD (conditional on macro scenario) to determine timing of defaults
   - Apply prepayment assumptions (from competing risks model)
   - Discount expected cash flows at effective interest rate
   - ECL = Contractual cash flows (NPV) - Expected cash flows (NPV)
   - This is the approach LendingClub explicitly uses per their 10-K

4. **Vintage analysis:**
   - Cumulative default rate by origination year, plotted against MOB
   - This directly mirrors the Sherwood PD curves you built at my prior institution (Product Type × MOB grid)
   - Identify which vintages are performing better/worse than expected
   - Compute smoothed marginal PD curves (6-month rolling average, as you did at my prior institution)
   - **Macro Context:** Overlay macro variables (unemployment, HPI) at origination to explain vintage differences

5. **Flow-rate analysis (NOT account-level transition matrices):**
   - **Important distinction:** The industry often conflates "roll rates" and "flow rates." True roll rates track individual accounts across buckets (requires account-level longitudinal data). Flow rates simply compare consecutive bucket balances in consecutive months. Even the OCC's Comptroller's Handbook acknowledges that most banks use the flow rate approach in practice.
   - Build the **Receivables Tracker** in institutional format: monthly dollar balances by DPD bucket (Current, 30+, 60+, 90+, 120+, 150+, 180+) with account counts, GCO, Recovery, NCO
   - Compute **flow rates** as simple ratios displayed below the dollar receivables:
     - 30+ Flow Rate = 30 DPD balance (this month) / Current balance (last month)
     - 60+ Flow Rate = 60 DPD balance (this month) / 30 DPD balance (last month)
     - 90+ Flow Rate = 90 DPD balance (this month) / 60 DPD balance (last month)
     - ...continuing through each bucket to GCO
   - **Flow Through Rate (NEW V3 KPI):** Product of all intermediate flow rates
     - Example: 0.028 × 0.382 × 0.701 × 0.85 × 0.90 × 0.92 × 0.95 = 0.468%
     - Interpretation: For every $100 in Current, approximately $0.47 will eventually charge off
     - Use as cross-check against PD model outputs
     - Track trend over time — early warning signal if trending up
   - Segment by grade and by vintage
   - Track flow rate trends over time — identify acceleration patterns (when flow rates spike)
   - These flow rates feed directly into the Streamlit forecasting tool's projection engine
   - **Optional enhancement:** Since we have account-level data in LendingClub, we CAN also compute a simplified transition matrix as a "model development view" — but position this as analytical context, not the primary forecasting mechanism

6. **ALLL tracker:**
   - Monthly ECL reserve level
   - Reserve build vs. release (ΔECL = provision expense)
   - NCO coverage ratio = ALLL / annualized NCO
   - Connect to prior role ALLL tracker work

7. **Three ECL views (from FEG framework):**
   - **Pre-FEG:** Pure model output with no macro overlay — uses historical average PD/flow rates
   - **Central:** Model output with base-case macro scenario applied — applies Baseline scenario adjustments
   - **Post-FEG:** Weighted average across all scenarios (Baseline 60%, Mild Downturn 25%, Stress 15%) plus qualitative adjustments

**Output files:** `ecl_by_grade.csv`, `ecl_by_vintage.csv`, `receivables_tracker.csv`, `flow_rates.csv`, `vintage_curves.csv`, `prepayment_rates_by_term.csv`, `flow_through_rate.json`

---

#### Days 12-14: Model Validation, Monitoring & Strategy Analysis

**Notebook: `08_Model_Validation_and_Monitoring.ipynb`**

**Prior Role Connection:** Directly mirrors your Q4'22 behavioral scorecard monitoring with Gini, PSI, CSI, VDI metrics and RAG framework.

1. **Discrimination metrics:**
   - AUC with 95% confidence intervals (bootstrap)
   - Gini coefficient = 2 × AUC - 1
   - KS statistic and KS plot (max separation between cumulative good/bad distributions)
   - CAP curve (Cumulative Accuracy Profile)
   - Gini over time (by quarter) — track stability

2. **Calibration metrics:**
   - Hosmer-Lemeshow test (by decile)
   - Calibration plot: predicted PD vs. actual default rate by score band
   - Brier score
   - Expected vs. actual defaults by grade
   - **Scenario Calibration:** Compare actual defaults in different macro regimes to predicted conditional PDs

3. **Stability metrics (the prior role hallmark):**
   - **PSI (Population Stability Index):** Compare score distribution: train vs. each test year
     - Green: PSI < 0.10 (stable)
     - Amber: 0.10 ≤ PSI < 0.25 (moderate drift)
     - Red: PSI ≥ 0.25 (significant drift — action required)
   - **CSI (Characteristic Stability Index):** Per-feature distribution shift
   - **VDI (Variable Deviation Index):** Per-variable drift measurement
   - RAG status table for all metrics — this is the exact format from your prior institution quarterly reports
   - **Macro PSI:** Apply PSI to macro features separately to show economic drift

4. **Out-of-time validation:**
   - Train on 2007-2015, validate on 2016, test on 2017 and 2018 separately
   - Track Gini/AUC degradation over time
   - This mirrors the benchmark population approach from my prior role (June-Aug 2014 benchmark)
   - **Scenario Validation:** For each vintage, compare predicted PD (with origination macro) to actual default rate

5. **External Benchmark Validation (NEW V4):**
   - Load benchmark_population_2014.csv (200,000 records)
   - Score population on your developed models
   - Compute PSI: Compare score distribution in benchmark to training population
   - Validate predicted PD distribution against actual PERFORMANCE_OUTCOME (GOOD/BAD)
   - Use as external calibration check
   - Document in notebook: "This mirrors a benchmark validation approach from my prior role where we compared model scores and outcomes to a known cohort"

6. **Backtesting:**
   - Predicted ECL vs. actual realized losses by vintage
   - By grade: are we over/under-reserving for any segment?
   - Prepayment backtesting: predicted vs. actual prepayment rates by term

---

**Notebook: `09_Macro_Scenario_and_Strategy_Analysis.ipynb`**

**Prior Role Connection:** FEG macro scenarios with GDP/HPI/unemployment weighted across Baseline/Upside/Downside/Downside 2 (75/5/15/5 weights). Mean reversion for extending beyond explicit forecast horizon.

1. **Macroeconomic data integration (FRED data from Notebook 01):**
   - Already pulled: UNRATE, CSUSHPINSA, A191RL1Q225SBEA, CPIAUCSL, DFF, UMCSENT
   - Map historical macro conditions to vintage performance
   - Build regression: default_rate ~ f(unemployment, GDP, HPI)
   - Estimate PD elasticity: "for each 1% increase in unemployment, PD increases by X bps"

2. **Scenario definition (mirroring FEG framework):**
   - **Baseline (60%):** Current trajectory from FRED forecasts (or assume rates stabilize at recent levels)
   - **Mild Downturn (25%):** Unemployment +1.5pp, GDP -1%
   - **Stress (15%):** Unemployment +3pp, GDP -3% (recession scenario)
   - Weighted average ECL = Σ(weight_i × ECL_i)

3. **Flow Rate Stress (NEW V3 - CRITICAL):**
   - **Application Level:** Adjust individual flow rates (e.g., increase each by 15%), NOT applied as a multiplier on final ECL output
   - **Why:** Compounding through the waterfall is multiplicative. A 15% stress on each flow rate produces ~75% increase in cumulative flow-through (because 1.15^7 ≈ 2.66), vs. only 15% increase if applied to ECL output. Also changes loss timing curve shape (losses accelerate and concentrate under stress). This preserves non-linear dynamics of delinquency behavior.
   - **Implementation:** For each scenario, adjust each flow rate by scenario-specific factor:
     - Baseline: 1.0× (no adjustment)
     - Mild Downturn: 1.10× (10% increase in each flow rate)
     - Stress: 1.20× (20% increase in each flow rate)

4. **Mean reversion for extended horizon:**
   - Explicit macro forecast: 8 quarters (2 years)
   - Beyond 8 quarters: mean-revert to long-run averages over remaining life
   - This explains the 8 quarters vs. 160 quarters distinction:
     - 8 quarters = explicit econometric forecast
     - 160 quarters = 40 years remaining life for mortgages, with mean-reverted macro inputs after Q8

5. **Sensitivity analysis:**
   - Impact of ±1% unemployment on portfolio ECL
   - Impact of ±100bp interest rates on prepayment and timing of losses
   - Impact of tightening/loosening credit policy (moving scorecard cutoff)

6. **STRATEGY ANALYSIS (the differentiator):**
   - **Credit policy optimization:** For each possible scorecard cutoff, compute approval rate, expected loss rate, expected revenue (interest income), and risk-adjusted return
   - **Grade-level profitability:** Interest income by grade minus expected loss — which grades are profitable?
   - **Credit expansion analysis:** "If LendingClub loosened the G cutoff by 10 points, what's the incremental loss vs. incremental revenue?"
   - **Vintage comparison:** "Why is 2017 vintage underperforming 2016 at the same MOB? What changed in origination strategy? What was unemployment at origination?"
   - These are the exact questions from the Sr Credit Strategy Analyst job posting

**Output files:** `macro_scenarios.json`, `strategy_analysis.csv`, `ecl_by_scenario.csv`, `flow_rate_stress_scenarios.csv`

---

### WEEK 3: Streamlit Platform + Polish (Days 15-21)

#### Days 15-18: Build the Streamlit Risk Analytics Platform

This is the PyCraft-equivalent tool. The high-level pages are:

1. **Portfolio Overview Dashboard** — KPIs, composition, default rates by grade/vintage/state

2. **Roll-Rate Analysis** — Receivables tracker with flow rates, delinquency flow visualization

3. **Vintage Performance** — Cumulative default curves by vintage × MOB (Sherwood-style)

4. **ECL Forecasting Engine** — The PyCraft core with dual-mode toggle:
   - **Dual-Mode Forecasting (NEW V3):**
     - Radio button: "Operational Forecast (Extend)" vs. "CECL Reserve Estimation (CECL)"
     - **Operational Mode (extend):**
       - Uses 6-month rolling average flow rates extended flat across 10-year projection
       - Simple, operationally practical — what PyCraft does
       - Shows "What would GCO/NCO be if current trends continue?"
     - **CECL Mode (cecl):**
       - Phase 1 (24 months, default): Macro-adjusted flow rates based on scenario
       - Phase 2 (12 months, default): Straight-line transition to historical averages
       - Phase 3 (remaining): Pure historical averages, no macro overlay
       - Shows "What's the regulatory-compliant ECL under this scenario?"
   - Tables: Monthly projections of Current, 30+, 60+, ..., 180+, GCO, NCO, ECL
   - Charts: Waterfall of balances, GCO/NCO timeline, ECL progression
   - **Flow Through Rate Display (NEW V3):** Show as KPI metric and trend chart
   - **Assumption Inputs (NEW V3):**
     - Streamlit sliders for: liquidation factor (%), prepayment rate (%), discount rate (%)
     - "Upload Assumptions" button: accepts Excel in institutional format
     - "Export Assumptions" button: downloads current settings
     - Dropdown to select macro scenario (Baseline, Mild Downturn, Stress)

5. **Scenario Analysis** — FRED integration, scenario weights, sensitivity analysis
   - Show Pre-FEG/Central/Post-FEG toggle (NEW V3)
   - Charts: ECL under each scenario, sensitivity tornado (impact on ECL)
   - Macro variable forecasts from FRED

6. **Model Monitoring** — Gini/PSI/CSI/VDI tracking with RAG status

7. **AI Analyst** — Claude-powered chatbot that can analyze uploaded files, answer portfolio questions, generate reports

#### Days 19-21: Polish, GitHub, Interview Prep

1. **Clean all notebooks:**
   - Add markdown explanations connecting each section to prior role experience
   - Document macro feature integration and why it's critical
   - Explain competing risks (prepayment) integration
   - Clarify LGD terminology change (Step 1/Step 2)

2. **README.md:** Frame around portfolio strategy and loss forecasting, not just modeling
   - Highlight macro integration
   - Explain dual-mode forecasting
   - Document prepayment model
   - Include screenshots of Streamlit pages

3. **CLAUDE.md:** Already done in this V5 roadmap — embed it in project root

4. **GitHub repository:** Clean commit history, proper .gitignore, requirements.txt with pinned versions

5. **Interview prep:** Walk through every model decision and connect to prior role experience:
   - "Why macro features in PD model?" → Time-based split, economic regimes
   - "Why prepayment model?" → Competing risks, LendingClub has significant prepayment
   - "Why dual-mode forecasting?" → Operational (PyCraft) vs. regulatory (CECL) needs
   - "Why stress at flow rate level?" → Multiplicative dynamics
   - "Flow Through Rate?" → Cross-check against PD, early warning

---

## Target Metrics Summary

| Component | Metric | Target |
|-----------|--------|--------|
| PD Scorecard (LogReg) | Gini | ≥ 55% |
| PD Scorecard (LogReg) | AUC | ≥ 0.75 |
| PD XGBoost | AUC | ≥ 0.80 |
| PD XGBoost | KS | ≥ 35% |
| EAD Model | MAPE | < 15% |
| LGD Model | MAE | < 0.10 |
| LGD Portfolio Average | Value | ≈ 0.83 |
| ECL / ALLL Ratio | Value | ≈ 5-7% (benchmarked to 10-K) |
| PSI (Score Stability) | Value | < 0.10 (Green) |
| Prepayment Model | MAPE | < 20% |
| Flow Through Rate | Trend | Stable or declining |

---

## Prior Role Experience Integration Map

| Project Component | Prior Role Experience | How to Discuss in Interview |
|-------------------|----------------|----------------------------|
| Macro Feature Integration | FEG scenarios with unemployment, GDP, HPI; PD model includes macro covariates | "I built models that learned conditional PD relationships — the same borrower has different risk in different economic regimes. FRED data was essential for time-based validation." |
| PD Scorecard + Macro | Behavioral scorecard with economic cycle context | "The scorecard is the policy tool, but macro features in the PD model capture regime shifts. This is how LendingClub goes beyond static risk assessment." |
| Prepayment Model | Empirical curves from historical payment data | "LendingClub has significant prepayment — ignoring it would overstate ECL. I built a competing risks model to separately estimate default vs. prepay hazards." |
| WOE/IV Feature Engineering | Q4'22 Credit Card Behavioral Scorecard — computed WoE, IV for VantageScore/FICO bins, utilization, DTI, inquiries, open tradelines | "I monitored these exact metrics quarterly for a $230M+ credit card portfolio" |
| Scorecard + RAG Framework | Behavioral scorecard with Gini ≥60% threshold, RAG status reporting to stakeholders | "I defined and tracked RAG status for model performance, escalating Amber/Red flags" |
| Model Validation (Gini/PSI/CSI/VDI) | Quarterly model monitoring with performance windows (6mo) and stability windows (3mo) | "I used 6-month performance windows and 3-month stability windows to assess model drift" |
| Vintage Analysis | Sherwood Lifetime Loss — marginal/cumulative PD curves by mortgage product (Fixed 15/30, ARM 3/5/7/10) × MOB | "I built PD curves by product type and MOB for mortgage portfolios, smoothed with 6-month rolling averages, and related them to macro conditions at origination" |
| Flow-Rate / Roll-Rate Analysis | Loss Forecasting receivables tracker — Current→30→60→90→120→150→180 DPD for $18B+ in portfolios. Flow rates computed as simple bucket-to-bucket ratios (e.g., 60+ flow rate = 60 DPD this month / 30 DPD last month), displayed below dollar receivables in tracker | "I tracked monthly receivables across 7 delinquency buckets and computed flow rates as simple ratios between consecutive buckets — this is the standard operational approach for portfolio-level loss forecasting" |
| Flow Through Rate | Loss forecasting output metric: product of flow rates showing ultimate GCO rate | "I tracked Flow Through Rate as an early warning signal — if it trends up, origination quality or economic conditions are deteriorating" |
| Dual-Mode Forecasting | PyCraft (operational extend method) vs. CECL regulatory mode | "I used PyCraft for AOP/FRP planning with simple rolling average extensions. For CECL, I built a three-phase approach with macro adjustment, reversion, and historical baseline — these serve different stakeholder needs." |
| Macro Scenarios | FEG scenarios with Baseline 75%, Upside 5%, Downside 15%, Downside2 5%. Mean reversion for 160 quarters | "I computed weighted macro forecasts across 4 scenarios and extended via mean reversion for the full remaining loan life" |
| ECL / CECL Framework | ALLL tracker, Pre-FEG/Central/Post-FEG ECL computation | "I maintained the monthly ALLL tracker and understood the three ECL views used for regulatory reporting — Pre-FEG (pure model), Central (base-case macro), Post-FEG (scenario-weighted + qualitative)" |
| Liquidation Factors | Portfolio-level (operational) vs. differentiated by term/vintage (CECL) | "PyCraft used portfolio-level because it was operational planning. For CECL, I built empirical prepayment curves differentiated by term and vintage — this better reflects LendingClub's actual data." |
| Forecasting Tool | PyCraft — Django-based tool taking receivables input, applying liquidation factors, projecting 10-year GCO/NCO using flow rates | "I used a proprietary loss forecasting tool for annual financial resource planning across all portfolios" |
| Portfolio Monitoring | Monthly receivables for HMC Premier ($14B), Credit Cards ($230M), Private Banking ($3B), etc. | "I tracked receivables across 10+ portfolio segments, reporting across reservable, reportable, and operational views" |
| External Benchmark Validation | Benchmark population approach for PSI and calibration | "This mirrors a benchmark validation approach from my prior role where we validated models against a known June-Aug 2014 benchmark population to ensure out-of-sample generalization" |

---

## Getting Started Checklist

- [ ] Review LendingClub_Complete_Data_Dictionary.md for variable reference (NEW V4)
- [ ] Create project directory structure
- [ ] Initialize git repository
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Save CLAUDE.md at project root
- [ ] Download wordsforthewise dataset from Kaggle
- [ ] Place dataset in `data/raw/`
- [ ] Verify dataset integrity: 2,260,668 usable rows after footer removal (NEW V4)
- [ ] Confirm benchmark_population_2014.csv is in data/raw/ for validation (NEW V4)
- [ ] Set up FRED API key (register at https://fred.stlouisfed.org/)
- [ ] Configure FRED access in config.py
- [ ] Begin with Notebook `01_EDA_and_Data_Cleaning.ipynb`

**Required packages:**
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0
optbinning>=0.19
shap>=0.43
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15
scipy>=1.11
statsmodels>=0.14
fredapi>=0.5
optuna>=3.3
streamlit>=1.28
anthropic>=0.18
jupyter>=1.0
pyarrow>=13.0
lifelines>=0.28
```

---

## V3 Implementation Notes

### Critical Dependencies for Macro Integration

1. **FRED API Access:** Requires `fredapi` library and API key (free from St. Louis Fed)
2. **Macro Data Alignment:** Issue_d month must align with FRED monthly/quarterly data. For quarterly macro (GDP), carry forward within quarter.
3. **Train/Test Split Integrity:** Time-based split (2007-2015 train, 2016 validation, 2017-2018 test) ensures models trained on recession/recovery are validated on post-crisis period — this is why macro features are critical.
4. **PD Model Coefficients:** Negative coefficients on macro features (higher unemployment → higher PD) should be intuitive and stable across train/validation/test.

### Competing Risks Implementation

1. **Prepayment Identification:** Not all "Fully Paid" loans are prepayments. Must compare actual life to contractual term.
2. **Data Requirements:** issue_d, last_pymnt_d, loan_status, term. Ensure dates are clean.
3. **Competing Risks Estimator:** Consider using lifelines library for survival analysis with competing risks.

### Dual-Mode Forecasting Architecture

1. **Extend Method:** Simple, no macro adjustment needed
   - Load 6-month rolling avg flow rates
   - Extend flat across projection horizon
   - Fast, operationally practical

2. **CECL Method:** Requires macro scenario application
   - Phase 1: Adjust each flow rate by scenario factor (1.0 Baseline, 1.10 Mild Downturn, 1.20 Stress)
   - Phase 2: Linear blend from Phase 1 rates to historical averages
   - Phase 3: Historical averages only
   - More complex, but regulatory compliant

### Streamlit UI/UX Priorities

1. **Dual-mode toggle** on ECL page — radio button, instant recalculation
2. **Assumption sliders** for liquidation, prepayment, discount rate
3. **Flow Through Rate** as KPI card with trend sparkline
4. **Pre-FEG/Central/Post-FEG** radio buttons showing scenario impact
5. **Download buttons** for assumptions Excel and results tables

---

## V4 Implementation Notes (NEW)

### Data Profiling and Quality

1. **Footer Row Cleanup:** The accepted_2007_to_2018Q4.csv contains 33 summary rows at the end. Drop these by identifying rows where loan_amnt is null after initial load. This leaves exactly 2,260,668 usable records from the reported 2,260,701.

2. **Column Drop Strategy (Notebook 01) — CORRECTED FROM FULL-FILE PROFILING:**

   **Tier 1: Drop Immediately (>93% missing or non-feature) — 34 columns:**
   - member_id (100% empty)
   - 13 sec_app_* + revol_bal_joint (95.22% empty — joint application only)
   - desc (94.42% empty), url (non-feature), id (non-feature, 89% empty)
   - pymnt_plan (constant 'n'), policy_code (constant 1)
   - dti_joint (94.66%), annual_inc_joint (94.66%), verification_status_joint (94.88%)
   - 15 hardship/settlement columns (>97% empty): hardship_type, hardship_reason,
     hardship_status, deferral_term, hardship_amount, hardship_start_date,
     hardship_end_date, payment_plan_start_date, hardship_length, hardship_dpd,
     hardship_loan_status, hardship_payoff_balance_amount, hardship_last_payment_amount,
     orig_projected_additional_accrued_interest, debt_settlement_flag_date,
     settlement_status, settlement_date, settlement_amount, settlement_percentage,
     settlement_term

   **Tier 2: Drop After EDA Review (70-85% missing) — 3 columns:**
   - mths_since_last_record (84.11% missing)
   - mths_since_recent_bc_dlq (77.01% missing)
   - mths_since_last_major_derog (74.31% missing)
   - NOTE: Create binary flags (e.g., has_public_record, has_major_derog) BEFORE dropping

   **Tier 3: KEEP with Missing Flag + Imputation (38-68% missing) — 14 columns:**
   - mths_since_recent_revol_delinq (67.25%) — create no_revol_delinq flag
   - next_pymnt_d (59.51%) — informational only, not a PD feature
   - mths_since_last_delinq (51.25%) — create no_delinq_history flag
   - il_util (47.28%) — installment utilization, keep for WOE/IV screening
   - mths_since_rcnt_il (40.25%) — months since recent installment, keep
   - open_acc_6m, open_act_il, open_il_12m, open_il_24m, total_bal_il,
     open_rv_12m, open_rv_24m, max_bal_bc, all_util, inq_fi, total_cu_tl,
     inq_last_12m (all ~38.31% missing) — installment/credit union features,
     keep ALL for WOE/IV screening with missing flag + median imputation
   - IMPORTANT: These were incorrectly listed as ">70% missing" in V4 based on
     100K sample. Full-file scan confirms they're only 38% missing (1.39M of 2.26M
     records populated). Dropping them loses usable signal.

   **Tier 4: Let WOE/IV Decide (1-38% missing) — all remaining:**
   - All Tier 3 and Tier 4 columns go through WOE/IV analysis in Notebook 02
   - Drop if IV < 0.02 (no predictive power)
   - Keep if IV > 0.02

3. **Data Type Parsing (Notebook 01):**
   - `term`: Use `.str.strip()` before extracting numeric value
   - `emp_length`: Build mapping from text values ('10+ years' → 10, '< 1 year' → 0, '2 years' → 2, etc.)
   - `issue_d`, `earliest_cr_line`, `last_pymnt_d`: Use `pd.to_datetime(col, format='%b-%Y')`
   - `int_rate`, `revol_util`: Already float64 — no % stripping required (confirmed from profiling)

4. **Missingness Strategy (Corrected from Full-File Scan):**
   - mths_since_last_delinq: 51.25% missing — create no_delinq_history flag, fill with median
   - mths_since_recent_revol_delinq: 67.25% — create no_revol_delinq flag, fill with median
   - Installment-specific features (open_act_il, open_il_12m, etc.): 38.31% missing
     (CORRECTED from earlier estimate of 78-81%; the 100K sample had disproportionate
     missingness in early records). Keep ALL with missing flags + median imputation.
   - mths_since_recent_inq: 13.07% missing — impute with median
   - emp_length: 5.64% missing — create emp_length_unknown flag, fill with median
   - Binary flag pattern: for any feature with >10% missing, always create a
     has_[feature] or no_[feature] binary flag BEFORE imputing, since missingness
     itself is often predictive in credit data

5. **Comprehensive Profiling Checklist (for Claude Code in Notebook 01):**

   This checklist ensures Claude Code performs thorough data validation beyond basic EDA.
   Results should be saved to `data/results/full_profiling_report.json` and key findings
   documented in Notebook 01 markdown cells.

   **A. Outlier Detection (all numeric columns):**
   - Compute IQR (Q1, Q3, IQR = Q3-Q1) for every numeric column
   - Flag values outside [Q1 - 3×IQR, Q3 + 3×IQR] as extreme outliers
   - Compute Z-scores; flag |Z| > 5 as extreme
   - PRIORITY columns with known outlier issues:
     * annual_inc: cap at 99th percentile (max = $110M is clearly erroneous)
     * dti: remove or cap values < 0 or > 100 (range is [-1, 999])
     * revol_util: investigate values > 150% (max = 892.3); decide cap threshold
     * revol_bal: investigate values > $500K (max = $2.9M)
     * tot_coll_amt, tot_cur_bal: cap at 99.5th percentile (max ~$9-10M)
     * bc_util, il_util, all_util: investigate values > 100%
     * last_fico_range_low/high: set values = 0 to NaN (invalid FICO)
     * settlement_percentage: investigate values > 100% (max = 521)
   - Document outlier treatment decisions (cap, winsorize, remove, or flag) in markdown
   - Save outlier summary: {column: {n_outliers, pct_outliers, min, max, p1, p99, treatment}}

   **B. Distribution Analysis (all numeric columns):**
   - Compute skewness and kurtosis for every numeric column
   - Classify: Normal (|skew|<0.5), Moderate Skew (0.5-1), High Skew (>1)
   - For highly skewed features (|skew|>1): note if log-transform might be needed
   - Generate histogram (20 bins) for top 30 features by IV (after WOE/IV in Notebook 02)
   - Plot KDE overlays for default=0 vs default=1 for key features
   - Assess if any features need transformation before logistic regression
   - Save distribution summary: {column: {mean, median, std, skew, kurtosis, shape_class}}

   **C. Correlation Analysis:**
   - Compute Pearson correlation matrix for ALL numeric features (after dropping Tier 1)
   - Flag all pairs with |correlation| > 0.80 (high multicollinearity risk)
   - Known expected high correlations to verify:
     * loan_amnt ↔ funded_amnt ↔ funded_amnt_inv ↔ installment (near-perfect)
     * fico_range_low ↔ fico_range_high (near-perfect, 4-point offset)
     * last_fico_range_low ↔ last_fico_range_high (near-perfect)
     * total_pymnt ↔ total_pymnt_inv (near-perfect)
     * out_prncp ↔ out_prncp_inv (near-perfect)
     * open_acc ↔ num_sats (likely >0.9)
     * tot_cur_bal ↔ tot_hi_cred_lim (likely >0.7)
   - For each highly correlated pair: recommend which to keep (domain knowledge)
   - Generate correlation heatmap (top 30 features, annotated)
   - Compute VIF (Variance Inflation Factor) for final feature set going to logistic regression
   - Save correlation findings: {pair: correlation, recommendation}

   **D. Full-File Validation (sanity checks):**
   - Confirm total rows = 2,260,668 after footer removal
   - Confirm terminal loan count: Fully Paid + Charged Off + Default ≈ 1,345,350
   - Confirm default rate ≈ 19.96%
   - Confirm grade monotonicity: default_rate(A) < B < C < D < E < F < G
   - Confirm term values are exactly {36, 60} after parsing
   - Confirm emp_length values are in {0, 1, 2, ..., 10, NaN} after parsing
   - Confirm int_rate range: [5.31, 30.99] with mean ≈ 13.09
   - Confirm fico_range_low range: [610, 845] with mean ≈ 698.59
   - Confirm loan_amnt range: [500, 40000] with mean ≈ $15,047
   - Verify no duplicate loan records (check on id or loan_amnt + issue_d + int_rate combination)
   - Cross-check: total_pymnt = total_rec_prncp + total_rec_int + total_rec_late_fee + recoveries (approximately)
   - Document any anomalies found

   **E. Deep Categorical Value Counts:**
   - For all categorical features with <50 unique values: full value_counts with percentages
   - For high-cardinality categoricals (emp_title, title, zip_code, addr_state): top 20 + unique count
   - Verify loan_status distribution matches expected: 9 values
   - Verify home_ownership: expected {RENT, MORTGAGE, OWN, OTHER, NONE, ANY}
   - Verify purpose: expected 14 categories (debt_consolidation dominant at ~47%)
   - Verify application_type: expected {Individual, Joint App}; ~95% Individual
   - Document any unexpected values (e.g., 'NONE' in home_ownership)
   - Save value counts: {column: {value: count, ...}}

### External Data Integration

1. **benchmark_population_2014.csv:**
   - 200,000 records from JUN-AUG 2014
   - Contains: FICO, delinquency bucket, PERFORMANCE_OUTCOME
   - Use in Notebook 08 for external validation
   - Compute PSI against training population
   - Backtest predicted PD against actual outcomes

2. **rejected_2007_to_2018Q4.csv:**
   - 27.6M records (9 columns only)
   - Used for selection bias discussion in Notebook 01
   - NOT used for modeling
   - Interview talking point: acknowledge survivorship bias but frame as intentional focus on portfolio management

### LGD Enhancement

1. **Two formulas for documentation:**
   - **Primary (NEW V4):** LGD = 1 - ((recoveries - collection_recovery_fee) / EAD) — more accurate
   - **Cross-check:** LGD = 1 - (total_rec_prncp / EAD) — for validation

2. **Why the upgrade:** Collection fee was previously included in recoveries numerator. Subtracting it gives net recovery to lender.

### Interview Framing for V4 Enhancements

1. **Data diligence:** "I profiled the full 2.26M loan dataset and documented data quirks — footer rows, leading spaces in term, text parsing for emp_length — this kind of diligence separates polished projects from student work."

2. **Feature engineering:** "I identified ~30 credit bureau variables for WOE/IV screening, categorized them (account activity, balance/limit, utilization, time-based, delinquency), and acknowledged that optbinning would handle complex interactions."

3. **External validation:** "The benchmark_population_2014.csv allowed me to validate my model on a separate cohort from a specific time period, mimicking the external validation approach I used at my prior institution."

4. **LGD accuracy:** "By separating recoveries from collection fees, I improved LGD formula accuracy. The portfolio-level LGD of ~83% matched the 10-K, validating my approach."

---

## V5 Enhancements (NEW February 2026)

### Changes Applied:
1. **Removed all HSBC-specific references** — replaced with generic framing emphasizing prior role experience
2. **Added contextual note on New Originations visibility** in Liquidation Factor Design section (Operational vs. CECL modes)
3. **Added clarifying note on FEG/stress orthogonality** in Three ECL Views and Stress sections
4. **Added recovery rate connection to LGD model** in ECL & Forecasting section
5. **Full-file profiling corrections** — Column drop strategy restructured into 4 tiers based on confirmed missingness from all 2,260,701 rows. 6 columns previously marked for immediate drop (38% missing) moved to "keep with imputation" tier. Outlier flags added for 14 numeric columns. Comprehensive profiling checklist embedded for Claude Code.

### Key Updates:
- All "HSBC" references converted to "my prior institution," "prior role," or removed entirely where appropriate
- Tool/framework names (PyCraft, Sherwood, FEG, OCC, Fed) preserved as-is
- New Originations input visibility now context-dependent: shown in Operational mode, hidden in CECL mode
- FEG views now explicitly documented as independent of forecasting mode
- Stress scenarios clarified for both Operational and CECL modes
- Recovery rate in DCL-ECL now linked to LGD model output for consistency

---

## Interview Talking Points Summary

1. **"Why macro features?"** — "Time-based split across economic regimes requires macroeconomic context. Without it, the model would overpredict defaults in boom periods because it learned recession rates as structural truths."

2. **"Why prepayment model?"** — "LendingClub has significant prepayment; ignoring it overstates ECL. I built a competing risks framework to separately estimate default vs. prepay hazards."

3. **"Scorecard vs. ML models?"** — "Scorecard is the policy tool — interpretable, monotonic, regulatory-friendly. ML models have better AUC and enable scenario analysis. Both serve different stakeholder needs."

4. **"Flow Through Rate?"** — "It's the product of all flow rates, showing the ultimate charge-off rate. I track it as an early warning signal — if it trends up, something's wrong with origination or the economy."

5. **"Dual-mode forecasting?"** — "Operational mode (extend) is what PyCraft does for planning — simple, rolling averages. CECL mode is three-phase with macro adjustment, reversion, and historical baseline. They serve different regulatory vs. operational needs."

6. **"Why stress at flow rate level?"** — "The waterfall is multiplicative. A 15% stress on each flow rate produces ~75% cumulative effect (because 1.15^7 ≈ 2.66), which correctly reflects how delinquency accelerates under stress."

7. **"Liquidation factors?"** — "Operational mode uses portfolio-level for simplicity. CECL mode uses empirical curves by term and vintage from LendingClub data — this better captures real prepayment dynamics."

8. **"Prior institution connection?"** — "I built this exact framework at my prior institution — FEG scenarios, loss forecasting receivables tracker, vintage analysis with macro context. LendingClub's approach is conceptually the same but with a smaller portfolio and richer individual loan data."

9. **"Data profiling?"** — "I documented all data quirks — footer rows, text parsing, missingness patterns, empty columns. This kind of diligence ensures robust pipelines and catches production issues early."

10. **"External validation?"** — "I used the benchmark_population_2014.csv to validate model generalization on a separate cohort, computing PSI and backtesting predicted PDs against actual outcomes. This mimics how we validated models at my prior institution."

---

End of V5 Roadmap
