# LendingClub Credit Risk Analytics Project — Roadmap V6

## Executive Summary

**Objective:** Build a production-grade Credit Risk Analytics project that goes beyond model-building to demonstrate portfolio strategy, loss forecasting, and operational risk management capabilities — the exact skills LendingClub's Credit Strategy, Loss Forecasting, and Model Development teams need.

**Timeline:** Flexible — quality over speed, target interview-ready by March/April 2026

**Key Shift from V1:** The project is no longer "build PD/EAD/LGD models." It's "build the kind of loss forecasting and portfolio management tool that LendingClub's risk team uses every day, powered by industry-standard models and enhanced with AI."

**Two Deliverables:**
1. **Modeling Notebooks** (Weeks 1-2): The analytical backbone — PD behavioral scorecard with grade, ML models with macro, EAD, LGD, ECL, vintage analysis, validation
2. **LendingClub Risk Analytics Platform** (Week 3): A Streamlit-based portfolio management and loss forecasting tool inspired by PyCraft, with dual-mode forecasting (Operational vs. CECL) and an embedded AI chatbot

---

## Changes from V2

The following major enhancements have been added to V3:

1. **FRED Macroeconomic Data Integration** — 6 key series (unemployment, HPI, GDP, CPI, Fed Funds, consumer sentiment) merged onto loans by origination month in Notebook 01
2. **Macro Features as PD Model Covariates** — ML PD models now include unemployment, GDP, HPI to capture economic regime dependence; essential for time-based validation to work correctly
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

## Changes from V5 (V5.1 + Data Gap Assessment Updates)

### V5.1 PD Scorecard Amendments:
1. **Grade included in WOE binning** — It's assigned at origination and legitimate for behavioral portfolio monitoring
2. **int_rate and sub_grade excluded** — Redundant with grade, cause collinearity
3. **Macro features excluded from logistic regression** — Confound with LC's growth trajectory (UNRATE sign inverts); reserved for ML models
4. **Disciplined feature selection** — IV ≥ 0.05, pairwise |corr| < 0.70, target 10-15 features
5. **All-negative WOE coefficients enforced** — Remove and refit if any positive coefficient appears
6. **Updated targets** — AUC ≥ 0.75, Gini ≥ 50%, KS ≥ 35%
7. **Updated RAG thresholds** — Green (Gini ≥ 55%), Amber (45-55%), Red (< 45%)

### Data Gap Assessment Fixes:
1. **Session 5.5:** Replaced month-level logistic regression with survival/CPR lookup approach
2. **Sessions 6-7:** Added synthetic monthly panel construction for flow rate computation
3. **Sessions 6-7:** Renamed flow rate analysis to "Forward Default Flow Rate Analysis"
4. **Sessions 6-7:** Added explicit data limitation documentation
5. **Sessions 8-9:** Added notes about synthetic flow rate base
6. **Sessions 10-11:** Added Streamlit data limitation disclaimers
7. **Session 12:** Added Data Limitations section to README

---

## Why This Approach Wins

### What LendingClub Actually Hires For

From live job postings (Sr Credit Strategy Analyst, Director of Data Science & ML, Sr Model Risk Manager):

| What They Want | How This Project Demonstrates It |
|----------------|----------------------------------|
| "Develop, implement and handle credit risk strategies involving credit underwriting, pricing, and loan amount assignment" | Behavioral scorecard with credit policy cutoff analysis; pricing optimization by grade |
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
- Be transparent about data limitations and demonstrate how to work around them analytically

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
- **IMPORTANT DATA LIMITATION:** Loan-level terminal outcomes only. No monthly payment history (PMTHIST file no longer publicly available). This requires synthetic monthly panel reconstruction for flow rate computation.

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
  performance-period merge as optional enhancement. Macro features are used in:
  - ML PD models (Notebook 04) — as raw covariates alongside borrower features
  - Stress testing and scenario analysis (Notebooks 08, 09) — macro overlay on flow rates
  - ECL computation (Notebook 07) — conditional PD adjustment
  Macro features are NOT used in the logistic regression scorecard (Notebook 03).
  In linear models, macro features confound with LC's growth trajectory across the
  time-based split (e.g., UNRATE falls 2010→2015 while LC scales up, causing inverted
  coefficient signs). Tree-based ML models handle this non-linearity correctly.

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
- **CRITICAL DATA LIMITATION:** Loan-level terminal outcomes only. No monthly payment history (PMTHIST file no longer publicly available).

### PD Models
- Behavioral PD Scorecard — Logistic Regression (Notebook 03):
  - Purpose: Behavioral scorecard for portfolio monitoring. These are loans already on books.
    Grade is a known, observed attribute (not a model output).
  - Features: WOE-transformed borrower characteristics INCLUDING grade
    - Exclude int_rate (mechanically set by grade — near-perfect collinearity)
    - Exclude sub_grade (finer version of grade — same collinearity issue)
    - Exclude macro features (confound in linear models — reserved for ML and stress testing)
  - Feature selection discipline:
    - IV ≥ 0.05 threshold (not 0.02 — eliminates noise features)
    - Pairwise |correlation| < 0.70 among selected WOE features
    - Binary flags included only if IV ≥ 0.02
    - Target: 10-15 final features (not 60)
  - Coefficient sign rule: ALL WOE coefficients must be negative
    (higher WOE = more good borrowers = lower default log-odds).
    If any coefficient is positive after fitting, remove that feature and refit.
  - L2 regularization (Ridge), hyperparameter tuned via 5-fold stratified CV
  - Output: probability of default + scorecard points
  - Target: AUC ≥ 0.75, Gini ≥ 50%, KS ≥ 35%
  - RAG status: Green (Gini ≥ 55%), Amber (45-55%), Red (< 45%)

- ML PD Models — XGBoost / LightGBM (Notebook 04):
  - Purpose: Performance ceiling model with all available information
  - Features: all borrower features + grade + int_rate + sub_grade + macro features
    Tree models handle collinearity natively through the tree structure
  - XGBoost and LightGBM with Optuna tuning
  - Macro features (UNRATE, CSUSHPINSA, etc.) are included as raw covariates
  - SHAP analysis shows macro + borrower feature importance
  - This is where macro features belong — tree models handle non-linear interactions
    with LC's growth trajectory correctly
  - Target: AUC ≥ 0.80, Gini ≥ 60%

### EAD & Prepayment
- **EAD:** Use `out_prncp` (outstanding principal) for defaulted loans. NOT `revol_bal`.
  For term loans, EAD ≈ outstanding balance at time of default.
- **Prepayment Model:** Build empirical prepayment rates from LendingClub data using
  issue_d, last_pymnt_d, loan_status, and term. Identify prepayments as Fully Paid loans
  with actual life significantly shorter than contractual term. Use survival analysis or
  empirical CPR lookup table (term × grade × vintage). Competing risks: three outcomes
  (default, prepay, maturity). Prepayment rates feed into DCF-ECL calculation.

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

#### Synthetic Monthly Panel Construction
- Loan-level data only provides terminal outcomes. To compute flow rates, reconstruct
  approximate monthly DPD status from terminal outcomes:
  - Fully Paid loans: assume Current every month until payoff. Last known balance = payoff
  - Charged Off loans: assume Current until last_pymnt_d, then 30→60→90→120 DPD progression
    to charge-off, back-calculating delinquency onset from last payment date
  - Assign scheduled amortization balances for Current months; freeze balance for delinquent
  - This produces ~50-60M rows (2.26M loans × ~24 avg months), requires chunked processing
  - Output: data/processed/synthetic_monthly_panel.parquet
  - Limitations: Curing (delinquent → current recovery) is unobservable. Flow rates are
    forward-only (Current → 30+ → 60+ → ... → GCO). Balances for delinquent months are
    approximate. In production with monthly payment tapes, curing and two-way transitions
    would be observable.

#### Forward Default Flow Rate Analysis
- Compute flow rates from synthetic monthly receivables:
  - Current → 30+ DPD
  - 30+ → 60+ DPD
  - ... continuing through each bucket to Gross Charge-Off
  - Flow rates are one-directional only. No curing rates or two-way transitions (curing unobservable)
- Flow Through Rate: product of all intermediate flow rates
  - Example: 0.028 × 0.382 × 0.701 × 0.85 × 0.90 × 0.92 × 0.95 = 0.468%
  - Interpretation: For every $100 in Current, approximately $0.47 will eventually charge off
  - Use as cross-check against PD model outputs and early warning signal
- Segment by grade and vintage
- All flow rates derived from synthetic monthly panel reconstruction

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
   - Flow rates are synthetically derived; dollar amounts are approximate

2. **method='cecl' (ASC 326 COMPLIANT, REGULATORY MODE):**
   - Phase 1 (R&S period, default 24 months): macro-adjusted flow rates
     Apply scenario-specific adjustments to each flow rate based on macro forecasts
   - Phase 2 (Reversion, default 12 months): straight-line transition from Phase 1 rates
     to long-run historical averages
   - Phase 3 (Remaining horizon): pure historical averages, NO adjustment for current/future conditions
   - Preserves ASC 326 requirement that model reflects expected conditions but reverts
     to historical baseline for extended horizon
   - Flow rates are synthetically derived; dollar amounts are approximate
   - CECL-compliant implementation would require observed monthly DPD data

#### Three ECL Views (Pre-FEG / Central / Post-FEG)
- **Pre-FEG:** Pure model output — 6-month rolling avg flow rates, no macro overlay
- **Central (FEG):** Baseline macro scenario applied to flow rates
- **Post-FEG:** Weighted average across all scenarios (Baseline 60%, Mild Downturn 25%, Stress 15%)
  plus qualitative adjustments
- **Streamlit Toggle:** Radio buttons on ECL page; charts/tables/reserve numbers update accordingly
- **Shows Impact:** Demonstrates effect of macro overlay and scenario weighting on final ECL
- The Pre-FEG/Central/Post-FEG toggle is ORTHOGONAL to the forecasting mode (Operational vs. CECL).
  Both modes support all three FEG views. The difference: in Operational mode, stressed flow rates
  extend flat across the full horizon; in CECL mode, stressed rates only apply during Phase 1
  (R&S period) and then revert during Phase 2/3. This is a great interview point: operational stress
  gives worst-case long-run trajectory; CECL stress is more conservative because it reverts.

#### Simple ECL Baseline
- ECL = PD × EAD × LGD (point-in-time, aggregated by segment)

#### DCF-Based ECL (mirrors LendingClub 10-K)
- For each loan pool (by grade), project monthly cash flows over remaining life
- Apply monthly marginal PD (conditional on macro scenario) to determine timing of defaults
- Apply prepayment assumptions and competing risks
- Discount expected cash flows at effective interest rate
- ECL = Contractual cash flows (NPV) - Expected cash flows (NPV)
- The `recovery_rate` assumption in `set_assumptions()` should DEFAULT to the portfolio-level output
  of the LGD model (if LGD model estimates ~83% average LGD → default recovery_rate = 17%).
  The slider allows override for sensitivity testing. This connects the LGD model (loan-level)
  to the flow-rate projection engine (portfolio-level).

### Stress Scenarios
- **Stress Application Level:** Adjust individual flow rates (e.g., increase each by 15%),
  NOT applied as a multiplier on final ECL output.
- **Why:** Compounding through the waterfall is multiplicative. A 15% stress on each flow rate
  produces ~75% increase in cumulative flow-through (because 1.15^7 ≈ 2.66), vs. only 15% increase
  if applied to ECL output. Also changes loss timing curve shape (losses accelerate and concentrate
  under stress). This preserves non-linear dynamics of delinquency behavior.
- **Stress Scenarios:** Stress is independent of forecasting mode. In Operational mode, stressed flow
  rates extend flat across the full horizon for worst-case long-run trajectory. In CECL mode, stressed
  flow rates apply during Phase 1 only and then revert, providing more conservative but realistic
  reserving under extended time horizons.
- **Note on Synthetic Base:** Flow rate stress is applied to synthetically derived rates.
  The compounding math is correct. Dollar amounts are approximate.

### Liquidation Factor Design
Two modes:

1. **Operational Mode (default):** Single portfolio-level liquidation factor, set via Streamlit slider.
   Simple, matches PyCraft. Example: "3.2% monthly liquidation" applied flat.
   - In Operational mode, both Liquidation Factor and New Originations inputs are shown to the user.

2. **CECL Mode:** Differentiated by loan term (36 vs 60 months) and vintage age. Empirical
   paydown curves computed from LendingClub data. Interview framing: "PyCraft used portfolio-level
   because it was operational planning. I also built CECL-aligned version where prepayment rates
   vary by term and vintage."
   - In CECL mode, only Liquidation Factor is shown; New Originations is hidden (hardcoded to $0)
     because CECL reserves are calculated on the existing portfolio only — new loans get their own
     Day 1 CECL assessment at origination.

### Assumption Input UI Design
- **Streamlit Frontend:** PRIMARY interface with sliders, inputs, dropdowns for interactive exploration
- **"Upload Assumptions" Button:** Accepts Excel in institutional format for operational workflow compatibility
- **"Export Assumptions" Button:** Downloads current settings as Excel for audit trail/documentation
- **Best of Both Worlds:** Interactive for exploration, Excel-compatible for operations

## Validation Metrics
- AUC, Gini, KS, PSI (<0.1 green, 0.1-0.25 amber, >0.25 red),
  CSI, VDI, Hosmer-Lemeshow, calibration plots, out-of-time validation.
- PD model metrics (AUC, Gini, KS, PSI) are fully based on real observed data.
- ECL backtesting uses synthetically derived flow rates.

## External Benchmark Validation
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
- Macro Integration → FEG scenarios with GDP/HPI/unemployment weights (ML models and stress testing only)
- Validation (Gini/PSI/CSI/VDI) → Quarterly model monitoring
- Vintage Analysis → Sherwood PD curves by mortgage product × MOB
- Prepayment Model → Empirical curves from historical data; competing risks framework
- Synthetic Monthly Panel → Reconstructed monthly DPD status from terminal outcomes for flow rate computation
- Roll-Rate / Flow-Rate Analysis → Forward default flow rates from synthetic receivables tracker
  (one-directional only; curing unobservable)
- Flow Through Rate → Cross-check against PD, early warning signal
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

2. **Footer Row Removal:**
   - After loading CSV, drop rows where `loan_amnt` is null or non-numeric
   - The raw file has 33 summary/footer rows at the end that are not loan records
   - This leaves exactly 2,260,668 usable loan records

3. **Define target variable:**
   - `default = 1` if `loan_status` in ['Charged Off', 'Default']
   - `default = 0` if `loan_status` in ['Fully Paid']
   - **DROP** all other statuses ('Current', 'In Grace Period', 'Late (16-30)', 'Late (31-120)', 'Does not meet credit policy: Fully Paid', 'Does not meet credit policy: Charged Off') — right-censored observations that bias PD downward
   - Terminal loans for modeling: 1,345,350 (268,599 defaults = 19.96% default rate)

4. **Immediate Column Drops:**

   a) **14 completely empty sec_app_* columns (100% null):**
      - sec_app_fico_range_low, sec_app_fico_range_high, sec_app_earliest_cr_line, sec_app_inq_last_6mths, sec_app_mort_acc, sec_app_open_acc, sec_app_revol_util, sec_app_open_act_il, sec_app_num_rev_accts, sec_app_chargeoff_within_12_mths, sec_app_collections_12_mths_ex_med, sec_app_mths_since_last_major_derog, revol_bal_joint, member_id

   b) **15 near-empty hardship/settlement columns (>97% null):**
      - hardship_type, hardship_reason, hardship_status, deferral_term, hardship_amount, hardship_start_date, hardship_end_date, payment_plan_start_date, hardship_length, hardship_dpd, hardship_loan_status, hardship_payoff_balance_amount, hardship_last_payment_amount, orig_projected_additional_accrued_interest, debt_settlement_flag_date, settlement_status, settlement_date, settlement_amount, settlement_percentage, settlement_term

   c) **Non-feature columns:**
      - id, url, desc (94.42% empty), pymnt_plan (all 'n'), policy_code (all 1)

5. **Data Type Fixes:**

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

6. **Missingness-based drops (>40% missing):**
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

8. **Feature categorization (~30 credit bureau features acknowledged):**
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

9. **Merge macroeconomic data from FRED API:**
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
   - IV > 0.5: Suspicious or strong → investigate for data leakage or legitimacy (e.g., `grade` will have very high IV because it's assigned based on credit risk — INCLUDE in behavioral scorecard context)
   - **IMPORTANT:** Include grade in WOE binning. Grade is a behavioral feature for portfolio monitoring.

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

4. **Validate monotonicity:** Bad rate should increase/decrease monotonically across WoE bins. Flag any rank-ordering breaks.

5. **Generate WOE/IV summary table** for all features — this becomes a key deliverable.

**Output files:** `woe_binning_results.pkl`, `iv_summary.csv`, `loans_woe_transformed.parquet` (includes macro features WOE-transformed)

---

#### Days 5-7: PD Models — Behavioral Scorecard + ML Ensemble

**Notebook: `03_PD_Model_Scorecard.ipynb`** — Behavioral PD Scorecard

**Prior Role Connection:** This is the culmination of your behavioral scorecard experience — but now you're building the scorecard from scratch rather than monitoring an existing one. This is a behavioral scorecard: the loan is already on the books, so grade is a known attribute.

1. **Logistic regression scorecard with WOE features:**
   - Input: WOE-transformed features with IV ≥ 0.05, **including grade**
   - **EXCLUDE** int_rate and sub_grade (collinear with grade)
   - **EXCLUDE** macro features (confound in linear models — reserved for ML and stress testing)
   - Feature selection: IV ≥ 0.05, pairwise |correlation| < 0.70, target 10-15 final features
   - Binary flags only if IV ≥ 0.02
   - L2 regularization (Ridge) — rationale: features are pre-selected and economically justified; we want stable coefficients keeping all features, not sparse elimination
   - Use sklearn's LogisticRegression with penalty='l2', tune C via cross-validation
   - **All WOE coefficients must be negative** — if any is positive, remove that feature and refit

2. **Convert to scorecard points:**
   - Base score: 600, PDO (Points to Double Odds): 20
   - Score = Offset + Factor × Σ(βi × WoEi)
   - Factor = PDO / ln(2)
   - Offset = Base_Score - Factor × ln(Base_Odds)
   - Generate scorecard table: Feature → Bin → WoE → Points
   - Grade should have the largest point spread (strongest discriminator)

3. **Credit policy analysis (THE STRATEGY LAYER):**
   - Plot score distribution for good vs. bad accounts
   - For different score cutoffs, compute: approval rate, expected default rate, expected loss rate
   - Build an **approval rate vs. expected loss tradeoff curve**
   - Identify the optimal cutoff that maximizes: approval_rate × (1 - default_rate)
   - The optimal cutoff should NOT be the minimum score (100% approval = no discrimination)
   - This is what LendingClub's Credit Strategy team does daily

4. **Grade mapping:**
   - Map score ranges to LendingClub grades A-G
   - Compare your model's grade assignment with actual grades in data
   - Analyze mis-grades (accounts your model would grade differently)

5. **RAG status framework:**
   - Define thresholds: Gini ≥ 55% (Green), 45-55% (Amber), < 45% (Red)
   - Compute Gini on train, validation, and test separately
   - Report RAG status per time period — mirrors your quarterly monitoring at my prior institution

**Metrics targets:** AUC ≥ 0.75, Gini ≥ 50%, KS ≥ 35%

**Output files:** `pd_logreg_model.pkl`, `scorecard_table.csv`, `pd_scorecard_metrics.json`, `credit_policy_analysis.csv`

---

**Notebook: `04_PD_Model_ML_Ensemble.ipynb`** — Performance Ceiling Models

**Context Note:** This is where macro features, int_rate, and sub_grade are included alongside grade for maximum predictive power. The ML models serve as the performance ceiling — tree-based models handle the non-linear interactions between macro features and LC's growth trajectory that caused problems in the logistic regression scorecard.

1. **XGBoost model:**
   - Input: original features (not WOE-transformed) — XGBoost handles non-linear relationships natively
   - **Include grade, int_rate, sub_grade** — collinearity handled by tree structure
   - **Include all 6 macro features** — tree models capture non-linear interactions correctly
   - Hyperparameter tuning via Optuna
   - SHAP analysis: global feature importance + individual prediction explanations
   - Show macro variables in top SHAP features

2. **LightGBM model:**
   - Same feature set as XGBoost (including grade, int_rate, sub_grade, macro features)
   - Compare training speed and performance

3. **Model comparison table:**

   | Metric | LogReg Scorecard | XGBoost | LightGBM |
   |--------|-----------------|---------|----------|
   | AUC (Train) | | | |
   | AUC (Test) | | | |
   | Gini (Test) | | | |
   | KS Statistic | | | |
   | Overfit Gap | | | |
   | N Features | 10-15 | All | All |

4. **Discussion:** Why would LendingClub use the scorecard for production credit decisions despite lower AUC? (Interpretability, regulatory requirements, monotonicity constraints, model governance — connect to your prior institution experience.)

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

1. **Two-step LGD model:**
   - **Step 1 (Classification Phase):** Logistic regression — did any recovery occur? (recovery_flag = 1 if recoveries > 0)
   - **Step 2 (Regression Phase):** For loans with recovery > 0, predict recovery rate using Beta regression
   - LGD = 1 - (P(recovery) × E[recovery_rate | recovery > 0])

2. **More accurate LGD formula:**
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

#### Days 10-11: ECL Computation, Vintage Analysis, Prepayment Model & Flow Rates

**Notebook: `07_ECL_Computation_and_Vintage_Analysis.ipynb`**

**Prior Role Connection:** This is the core of your Loss Forecasting team experience. The receivables tracker, ALLL computation, vintage analysis, and flow-rate matrices are exactly what you built at my prior institution for $18B+ in mortgage portfolios. Competing risks framework for prepayment is integrated.

**CRITICAL DATA LIMITATION:** The LendingClub dataset provides loan-level terminal outcomes, not monthly payment history. This requires synthetic monthly panel construction.

1. **Synthetic Monthly Panel Construction (New Component):**
   - Before computing flow rates, reconstruct approximate monthly DPD status from terminal outcomes
   - For each loan, create one row per month from `issue_d` to:
     - `last_pymnt_d` for Fully Paid loans
     - Estimated charge-off date for Charged Off loans
   - Back-calculate: Fully Paid = current every month. Charged Off = current until delinquency onset (last_pymnt_d + 1 month), then 30→60→90→120 DPD progression to charge-off
   - Assign scheduled amortization balances
   - Output: ~50-60M rows, needs chunked processing
   - Output: data/processed/synthetic_monthly_panel.parquet
   - Document assumptions: "Performing loans assumed current until payoff. Defaulted loans back-calculated from last payment date. Curing is unobservable. Intermediate delinquencies for eventually-performing loans are invisible."

2. **Prepayment Model:**
   - **Goal:** Model competing risks — loans can default, prepay, or reach maturity
   - **Data:** Use `issue_d` (origination), `last_pymnt_d` (last payment), `loan_status`, `term` (36 or 60 months)
   - **Prepayment Identification:** Fully Paid loans with actual life < contractual term
     - Example: 60-month term loan that was Fully Paid in month 48 = prepayment
   - **Empirical Prepayment Rates:** Compute by term (36 vs 60 months) and vintage year using CPR lookup tables
   - **Competing Risks Framework:**
     - Three outcomes: Default (before maturity), Prepay (early full repayment), Maturity (reach end of term and pay off)
     - Quarterly prepayment rates feed into DCF-ECL as alternative cash flow scenario
   - **Integration into ECL:** For each loan, survival analysis considers both default and prepayment hazards
   - **Interview Framing:** "LendingClub has significant prepayment risk; ignoring it would overstate expected loss. I built a competing risks model to separately estimate default vs prepay rates."

3. **Simple ECL baseline:**
   - ECL = PD × EAD × LGD (point-in-time, aggregated by segment)
   - Compute by grade (A-G), by vintage year, by purpose
   - Portfolio-level ALLL ratio = total ECL / total outstanding balance
   - Compare to LendingClub 10-K ALLL ratio (5.7%)

4. **DCF-based ECL (mirrors LendingClub 10-K):**
   - For each loan pool (by grade), project monthly cash flows over remaining life
   - Apply monthly marginal PD (conditional on macro scenario) to determine timing of defaults
   - Apply prepayment assumptions (from competing risks model)
   - Discount expected cash flows at effective interest rate
   - ECL = Contractual cash flows (NPV) - Expected cash flows (NPV)
   - This is the approach LendingClub explicitly uses per their 10-K

5. **Vintage analysis:**
   - Cumulative default rate by origination year, plotted against MOB
   - This directly mirrors the Sherwood PD curves you built at my prior institution (Product Type × MOB grid)
   - Identify which vintages are performing better/worse than expected
   - Compute smoothed marginal PD curves (6-month rolling average, as you did at my prior institution)
   - **Macro Context:** Overlay macro variables (unemployment, HPI) at origination to explain vintage differences

6. **Forward Default Flow Rate Analysis (Synthetic Reconstruction):**
   - Build the **Receivables Tracker** in institutional format: monthly dollar balances by DPD bucket (Current, 30+, 60+, 90+, 120+, 150+, 180+) with account counts, GCO, Recovery, NCO
   - Compute **flow rates** as simple ratios (one-directional only):
     - 30+ Flow Rate = 30 DPD balance (this month) / Current balance (last month)
     - 60+ Flow Rate = 60 DPD balance (this month) / 30 DPD balance (last month)
     - 90+ Flow Rate = 90 DPD balance (this month) / 60 DPD balance (this month)
     - ...continuing through each bucket to GCO
   - **Flow Through Rate:** Product of all intermediate flow rates
     - Example: 0.028 × 0.382 × 0.701 × 0.85 × 0.90 × 0.92 × 0.95 = 0.468%
     - Interpretation: For every $100 in Current, approximately $0.47 will eventually charge off
     - Use as cross-check against PD model outputs
     - Track trend over time — early warning signal if trending up
   - Flow rates are forward-only: Current → 30+ → 60+ → ... → GCO
   - **CRITICAL:** No curing rates or two-way transition matrices (curing is unobservable in loan-level data)
   - Segment by grade and by vintage
   - Track flow rate trends over time — identify acceleration patterns
   - These flow rates feed directly into the Streamlit forecasting tool's projection engine
   - **Data Limitation Documentation:** "Flow rates derived from synthetic monthly panel reconstruction. Curing is unobservable. Flow rates represent forward transitions only. Balances are approximate for delinquent months. In a production environment with monthly payment tapes, curing rates and two-way transition matrices would be observable."

7. **ALLL tracker:**
   - Monthly ECL reserve level
   - Reserve build vs. release (ΔECL = provision expense)
   - NCO coverage ratio = ALLL / annualized NCO
   - Connect to prior role ALLL tracker work

8. **Three ECL views (from FEG framework):**
   - **Pre-FEG:** Pure model output with no macro overlay — uses historical average PD/flow rates
   - **Central:** Model output with base-case macro scenario applied — applies Baseline scenario adjustments
   - **Post-FEG:** Weighted average across all scenarios (Baseline 60%, Mild Downturn 25%, Stress 15%) plus qualitative adjustments

**Output files:** `ecl_by_grade.csv`, `ecl_by_vintage.csv`, `receivables_tracker.csv`, `flow_rates.csv`, `vintage_curves.csv`, `prepayment_rates_by_term.csv`, `flow_through_rate.json`, `synthetic_monthly_panel.parquet`

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

5. **External Benchmark Validation:**
   - Load benchmark_population_2014.csv (200,000 records)
   - Score population on your developed models
   - Compute PSI: Compare score distribution in benchmark to training population
   - Validate predicted PD distribution against actual PERFORMANCE_OUTCOME (GOOD/BAD)
   - Use as external calibration check
   - Document in notebook: "This mirrors a benchmark validation approach from my prior role where we compared model scores and outcomes to a known cohort"

6. **Backtesting:**
   - Reframe as: Predicted cumulative default rate vs actual cumulative default rate by vintage (not flow-rate-based ECL vs actual losses)
   - By grade: are we over/under-reserving for any segment?
   - Prepayment backtesting: predicted vs. actual prepayment rates by term
   - Note: PD model metrics (AUC, Gini, KS, PSI) are fully based on real observed data. ECL backtesting uses synthetically derived flow rates.

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

3. **Flow Rate Stress:**
   - **Application Level:** Adjust individual flow rates (e.g., increase each by 15%), NOT applied as a multiplier on final ECL output
   - **Why:** Compounding through the waterfall is multiplicative. A 15% stress on each flow rate produces ~75% increase in cumulative flow-through (because 1.15^7 ≈ 2.66), vs. only 15% increase if applied to ECL output. Also changes loss timing curve shape (losses accelerate and concentrate under stress). This preserves non-linear dynamics of delinquency behavior.
   - **Implementation:** For each scenario, adjust each flow rate by scenario-specific factor:
     - Baseline: 1.0× (no adjustment)
     - Mild Downturn: 1.10× (10% increase in each flow rate)
     - Stress: 1.20× (20% increase in each flow rate)
   - **Note on Synthetic Base:** Flow rate stress is applied to synthetically derived rates. The compounding math is correct. Dollar amounts are approximate.

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
   - Add sidebar disclaimer: "Flow rates derived from synthetic monthly panel reconstruction. Production implementation would use observed monthly payment data."

3. **Vintage Performance** — Cumulative default curves by vintage × MOB (Sherwood-style)

4. **ECL Forecasting Engine** — The PyCraft core with dual-mode toggle:
   - **Dual-Mode Forecasting:**
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
   - **Flow Through Rate Display:** Show as KPI metric and trend chart
   - **Assumption Inputs:**
     - Streamlit sliders for: liquidation factor (%), prepayment rate (%), discount rate (%)
     - "Upload Assumptions" button: accepts Excel in institutional format
     - "Export Assumptions" button: downloads current settings
     - Dropdown to select macro scenario (Baseline, Mild Downturn, Stress)
   - Add sidebar disclaimer: "Flow rates derived from synthetic monthly panel reconstruction. Production implementation would use observed monthly payment data."

5. **Scenario Analysis** — FRED integration, scenario weights, sensitivity analysis
   - Show Pre-FEG/Central/Post-FEG toggle
   - Charts: ECL under each scenario, sensitivity tornado (impact on ECL)
   - Macro variable forecasts from FRED
   - Add sidebar disclaimer: "Flow rates derived from synthetic monthly panel reconstruction. Production implementation would use observed monthly payment data."

6. **Model Monitoring** — Gini/PSI/CSI/VDI tracking with RAG status

7. **AI Analyst** — Claude-powered chatbot that can analyze uploaded files, answer portfolio questions, generate reports

#### Days 19-21: Polish, GitHub, Interview Prep

1. **Clean all notebooks:**
   - Add markdown explanations connecting each section to prior role experience
   - Document macro feature integration and why it's used in ML/stress but not scorecard
   - Explain competing risks (prepayment) integration
   - Clarify LGD terminology (Step 1/Step 2)
   - Document synthetic panel construction and limitations

2. **README.md:** Frame around portfolio strategy and loss forecasting, not just modeling
   - Highlight macro integration in ML and stress testing
   - Explain dual-mode forecasting
   - Document prepayment model
   - **Add Data Limitations section:**
     - The LendingClub public dataset has loan-level terminal outcomes, no monthly payment history
     - Monthly DPD status is synthetically reconstructed
     - Flow rates are forward-only (curing unobservable)
     - PD, EAD, LGD models use real observed data
     - Framework is production-ready; only input granularity differs
   - Include screenshots of Streamlit pages

3. **CLAUDE.md:** Already done in this V6 roadmap — embed it in project root

4. **GitHub repository:** Clean commit history, proper .gitignore, requirements.txt with pinned versions

5. **Interview prep:** Walk through every model decision and connect to prior role experience:
   - "Why grade in scorecard?" → Behavioral scorecard, loan already on books, grade is known attribute
   - "Why macro in ML but not scorecard?" → Linear model confounding with LC growth trajectory; tree models handle non-linearity correctly
   - "Why prepayment model?" → Competing risks, LendingClub has significant prepayment
   - "Why dual-mode forecasting?" → Operational (PyCraft) vs. regulatory (CECL) needs
   - "Why stress at flow rate level?" → Multiplicative dynamics
   - "Flow Through Rate?" → Cross-check against PD, early warning
   - "Why synthetic panel?" → Answered in talking points below

---

## Target Metrics Summary

| Component | Metric | Target |
|-----------|--------|--------|
| PD Behavioral Scorecard | Gini | ≥ 50% |
| PD Behavioral Scorecard | AUC | ≥ 0.75 |
| PD Behavioral Scorecard | KS | ≥ 35% |
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
| Macro Feature Integration | FEG scenarios with unemployment, GDP, HPI in ML and stress testing | "I built models that learned conditional PD relationships — the same borrower has different risk in different economic regimes. Macro features are essential for scenario testing and stress analysis, handled properly in tree models." |
| PD Behavioral Scorecard | Behavioral scorecard with grade as strongest predictor. Macro reserved for ML. | "The behavioral scorecard monitors loans already on books — grade is an observed attribute. For economic cycle effects, we use the ML PD model and stress testing framework rather than forcing macro features into a linear model." |
| Prepayment Model | Empirical curves from historical data; competing risks framework | "LendingClub has significant prepayment — ignoring it would overstate ECL. I built a competing risks model to separately estimate default vs. prepay hazards." |
| Synthetic Monthly Panel | Reconstructed monthly DPD status from terminal outcomes for flow rate computation | "The public LendingClub dataset provides loan-level terminal outcomes, not monthly payment tapes. I reconstructed approximate monthly DPD status using the amortization schedule and charge-off timing. This captures the forward default cascade reliably, but cannot observe curing events." |
| WOE/IV Feature Engineering | Q4'22 Credit Card Behavioral Scorecard — computed WoE, IV for VantageScore/FICO bins, utilization, DTI, inquiries, open tradelines | "I monitored these exact metrics quarterly for a $230M+ credit card portfolio" |
| Scorecard + RAG Framework | Behavioral scorecard with Gini ≥55% threshold, RAG status reporting to stakeholders | "I defined and tracked RAG status for model performance, escalating Amber/Red flags" |
| Model Validation (Gini/PSI/CSI/VDI) | Quarterly model monitoring with performance windows (6mo) and stability windows (3mo) | "I used 6-month performance windows and 3-month stability windows to assess model drift" |
| Vintage Analysis | Sherwood Lifetime Loss — marginal/cumulative PD curves by mortgage product × MOB | "I built PD curves by product type and MOB for mortgage portfolios, smoothed with 6-month rolling averages, and related them to macro conditions at origination" |
| Forward Default Flow Rates | Loss Forecasting receivables tracker — Current→30→60→90→120→150→180 DPD for $18B+ in portfolios. Flow rates computed as simple bucket-to-bucket ratios. | "I tracked monthly receivables across 7 delinquency buckets and computed flow rates as simple ratios between consecutive buckets. For LendingClub, I reconstructed monthly status from terminal data since monthly tapes aren't available." |
| Flow Through Rate | Loss forecasting output metric: product of flow rates showing ultimate GCO rate | "I tracked Flow Through Rate as an early warning signal — if it trends up, origination quality or economic conditions are deteriorating" |
| Dual-Mode Forecasting | PyCraft (operational extend method) vs. CECL regulatory mode | "I used PyCraft for AOP/FRP planning with simple rolling average extensions. For CECL, I built a three-phase approach with macro adjustment, reversion, and historical baseline." |
| Macro Scenarios | FEG scenarios with Baseline 75%, Upside 5%, Downside 15%, Downside2 5%. Mean reversion. | "I computed weighted macro forecasts across 4 scenarios and extended via mean reversion for the full remaining loan life" |
| ECL / CECL Framework | ALLL tracker, Pre-FEG/Central/Post-FEG ECL computation | "I maintained the monthly ALLL tracker and understood the three ECL views used for regulatory reporting — Pre-FEG, Central, Post-FEG" |
| Liquidation Factors | Portfolio-level (operational) vs. differentiated by term/vintage (CECL) | "PyCraft used portfolio-level for simplicity. For CECL, I built empirical prepayment curves differentiated by term and vintage." |
| Forecasting Tool | PyCraft — Django-based tool taking receivables input, applying liquidation factors, projecting 10-year GCO/NCO | "I used a proprietary loss forecasting tool for annual financial resource planning" |
| Portfolio Monitoring | Monthly receivables for multiple portfolio segments | "I tracked receivables across 10+ portfolio segments, reporting across reservable, reportable, and operational views" |
| External Benchmark Validation | Benchmark population approach for PSI and calibration | "I validated models against a known June-Aug 2014 benchmark population to ensure out-of-sample generalization" |

---

## Getting Started Checklist

- [ ] Review LendingClub_Complete_Data_Dictionary.md for variable reference
- [ ] Create project directory structure
- [ ] Initialize git repository
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Save CLAUDE.md at project root
- [ ] Download wordsforthewise dataset from Kaggle
- [ ] Place dataset in `data/raw/`
- [ ] Verify dataset integrity: 2,260,668 usable rows after footer removal
- [ ] Confirm benchmark_population_2014.csv is in data/raw/ for validation
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

## Interview Talking Points Summary

1. **"Why grade in the scorecard?"** — "This is a behavioral scorecard for portfolio monitoring — the loans are already on LendingClub's books. Grade is a known, observed attribute assigned at origination, making it the single strongest predictor. If this were an origination scorecard (deciding whether to approve a new applicant), we'd exclude grade because we'd be building the tool that assigns it."

2. **"Why macro features in ML but not scorecard?"** — "In a logistic regression with our time-based split (2007-2015 train, 2017-2018 test), macro features confound with LendingClub's growth trajectory. UNRATE falls 2010→2015 while LC scales dramatically, causing the model to learn an inverted relationship. Tree-based models (XGBoost, LightGBM) handle non-linear interactions correctly, so macro features belong there and in stress testing."

3. **"Why not use Freddie Mac data?"** — "I chose LendingClub because the role is at LendingClub and personal loan credit risk is directly relevant. The trade-off is that the public dataset lacks monthly payment history, which I worked around with synthetic reconstruction. This analytical exercise actually demonstrates deeper understanding than simply loading a pre-made file."

4. **"Why synthetic monthly panel?"** — "The public LendingClub dataset provides loan-level terminal outcomes, not monthly payment tapes. I reconstructed approximate monthly DPD status using the amortization schedule and charge-off timing. This captures the forward default cascade — Current through Charge-off — reliably, but cannot observe curing events. I scoped the flow rate analysis to forward transitions accordingly."

5. **"Why prepayment model?"** — "LendingClub has significant prepayment; ignoring it overstates ECL. I built a competing risks framework to separately estimate default vs. prepay hazards from terminal data."

6. **"Scorecard vs. ML models?"** — "Scorecard is the policy tool — interpretable, monotonic, regulatory-friendly, 10-15 features. ML models have better AUC and enable scenario analysis. Both serve different stakeholder needs."

7. **"Flow Through Rate?"** — "It's the product of all flow rates, showing the ultimate charge-off rate. I track it as an early warning signal — if it trends up, something's wrong with origination or the economy."

8. **"Dual-mode forecasting?"** — "Operational mode (extend) is what PyCraft does for planning — simple, rolling averages extended flat. CECL mode is three-phase with macro adjustment, reversion, and historical baseline. They serve different regulatory vs. operational needs."

9. **"Why stress at flow rate level?"** — "The waterfall is multiplicative. A 15% stress on each flow rate produces ~75% cumulative effect (because 1.15^7 ≈ 2.66), which correctly reflects how delinquency accelerates under stress."

10. **"How accurate are the flow rates?"** — "The PD, LGD, and EAD models use real observed data. The flow-rate-based ECL uses synthetically reconstructed monthly status, making dollar amounts approximate. The framework is identical to production implementation — only input granularity differs."

---

## V6 Enhancements (Data Gap Assessment + V5.1 Amendments)

### V5.1 PD Scorecard Amendments:
1. Grade included in WOE binning and PD scorecard (portfolio monitoring context)
2. int_rate and sub_grade excluded from scorecard (collinearity with grade)
3. Macro features excluded from logistic regression (confounding with LC growth trajectory — UNRATE sign inverts)
4. Disciplined feature selection: IV ≥ 0.05, |corr| < 0.70, 10-15 features
5. All WOE coefficients must be negative; remove and refit if positive
6. Updated targets: AUC ≥ 0.75, Gini ≥ 50%, KS ≥ 35%
7. Updated RAG: Green (Gini ≥ 55%), Amber (45-55%), Red (< 45%)

### Data Gap Assessment Fixes:
1. Session 5.5: Replaced month-level logistic regression with survival/CPR approach
2. Session 6: Added synthetic monthly panel construction for flow rate computation
3. Session 6: Renamed flow rate analysis to "Forward Default Flow Rate Analysis"
4. Session 6: Added explicit data limitation documentation
5. Sessions 7-8: Added notes about synthetic flow rate base
6. Sessions 9-10: Added Streamlit data limitation disclaimers
7. Session 11: Added Data Limitations section to README

### Key Principle:
The LendingClub public dataset (accepted_2007_to_2018Q4.csv) provides one row per loan with origination features and terminal outcomes. There is NO monthly payment history (the PMTHIST file is no longer publicly available). Sessions 0-5 (PD, ML, EAD, LGD) use real observed data. Sessions 5.5-10 use a mix of real data and synthetically reconstructed monthly status. All limitations are documented honestly.

---

End of V6 Roadmap
