# Claude Code — Session-by-Session Prompting Guide
## LendingClub Credit Risk Analytics Project
## Version 6 (V6) — February 2026

### Changes from V5:
This version incorporates:
- **V5.1 amendments**: PD scorecard now includes grade as a behavioral feature, excludes macro features from logistic regression (reserved for ML and stress testing), and applies disciplined feature selection (IV ≥ 0.05, |correlation| < 0.70, target 10-15 features)
- **Data gap fixes**: Session 5.5 and 6 reframed around synthetic monthly panel reconstruction instead of unavailable monthly payment history
- **Flow rate revision**: Forward-only flow rate analysis with honest documentation of limitations (curing unobservable, intermediate delinquencies invisible)
- **Updated interview framing**: Guidance for discussing synthetic reconstruction as an analytical exercise demonstrating technical depth
- All other V5 content preserved: known data quirks, column categories, exact file paths, FRED integration, quality gates

---

## General Principles for Claude Code

### 1. The CLAUDE.md File Is Your Foundation
The CLAUDE.md file at your project root is automatically read by Claude Code at the start of every session. This is the single most important file for maintaining consistency across sessions. It should contain your key technical decisions, coding standards, and project context.

### 2. Scope Each Session Tightly
Claude Code works best when you give it one clear task per session. Don't say "build the entire PD modeling pipeline." Instead: "Build Notebook 03: PD Scorecard using logistic regression with WOE-transformed features. Here's what Notebook 02 produced [point to files]."

### 3. The Input-Process-Output Pattern
Every prompt should follow this structure:
- **INPUT:** What files/data does Claude Code have to work with?
- **PROCESS:** What should it build/do?
- **OUTPUT:** What files should it produce and where should they go?
- **QUALITY:** What standards must the output meet?
- **SANITY CHECKS:** What numerical ranges should we expect?

### 4. Point to Files, Don't Paste Content
Instead of copying your roadmap text into the prompt, say: "Read the roadmap at `docs/LendingClub_Credit_Risk_Project_Roadmap_V5.md`, specifically the section on Days 3-4." Claude Code can read files directly — this saves context space.

### 5. Let Claude Code Run and Iterate
Claude Code's strength is that it can write code, execute it, see errors, and fix them autonomously. Don't micro-manage each line. Give it the goal and let it work. Intervene only when it's going in the wrong direction.

### 6. Review Outputs Before Moving On
After each session, review the generated files. Check that:
- Output files exist in the expected locations
- Notebooks have markdown explanations (not just code)
- Key metrics are printed and match expectations
- The src/ module functions have docstrings and type hints

---

## Session 0: Project Setup

### Prompt:
```
I'm starting a new credit risk analytics project for LendingClub.

Read the project roadmap at docs/LendingClub_Credit_Risk_Project_Roadmap_V5.md —
specifically the "Project Architecture" and "CLAUDE.md File" sections.

Please do the following:

1. Create the complete project directory structure as specified in the roadmap
2. Create the CLAUDE.md file at the project root with the following content:

---
# CLAUDE.md — LendingClub Credit Risk Analytics

## Project Overview
Portfolio management and loss forecasting tool for LendingClub loans.
Uses institutional format receivables tracking, flow rate analysis, and DCF-ECL methodology.

## Key Technical Decisions

### Data Preparation
- Target variable: binary (0=non-default, 1=default)
  - Default: status in ['Charged Off', 'Default']
  - Non-default: status in ['Fully Paid']
  - Drop: all other statuses (right-censored, in-transit, etc.)
- Time-based split (no random shuffle):
  - Train: 2007-2015
  - Validation: 2016
  - Test: 2017-2018
- Rationale: Credit models must validate out-of-time. This split tests generalization
  across economic regimes (2007-08 crisis, recovery, expansion).

### Known Data Quirks (from full-file profiling — all 2,260,701 rows scanned)
- CSV has 33 footer/summary rows at end — drop rows where loan_amnt is null
- `term` has leading spaces: ' 36 months' — use .str.strip() then extract int
- `emp_length` is text: '10+ years', '< 1 year', etc. — parse to numeric (5.64% missing)
- Dates are text format 'MMM-YYYY' (e.g., 'Dec-2015') — use format='%b-%Y'
- `int_rate` and `revol_util` are already float64 (no % stripping)
- 14 sec_app_* columns are 95.22% empty (joint apps only, ~108K records) — drop
- ~15 hardship/settlement columns are >97% empty — drop immediately
- member_id is 100% empty — drop
- desc is 94.42% empty (~126K records have content) — drop
- loan_status has 9 unique values including 'Does not meet credit policy' variants
- Terminal statuses for modeling: Fully Paid (1,076,751), Charged Off (268,559), Default (40)
- Default rate after filtering: ~19.96%
- `recoveries` and `collection_recovery_fee` exist — use for accurate LGD
- benchmark_population_2014.csv available for external validation
- OUTLIER FLAGS: annual_inc max=$110M (cap p99), dti range [-1,999] (clean),
  revol_util max=892.3 (cap 150%), last_fico min=0 (invalid, set NaN)

### Column Categories (from dataset profiling)

#### Columns to Drop Immediately (never use):
# 14 completely empty (100% null):
sec_app_fico_range_low, sec_app_fico_range_high, sec_app_earliest_cr_line,
sec_app_inq_last_6mths, sec_app_mort_acc, sec_app_open_acc, sec_app_revol_util,
sec_app_open_act_il, sec_app_num_rev_accts, sec_app_chargeoff_within_12_mths,
sec_app_collections_12_mths_ex_med, sec_app_mths_since_last_major_derog,
revol_bal_joint, member_id

# Non-feature columns:
id, url, desc, pymnt_plan, policy_code

# Near-empty hardship/settlement (>97% null):
hardship_type, hardship_reason, hardship_status, deferral_term, hardship_amount,
hardship_start_date, hardship_end_date, payment_plan_start_date, hardship_length,
hardship_dpd, hardship_loan_status, hardship_payoff_balance_amount,
hardship_last_payment_amount, orig_projected_additional_accrued_interest,
debt_settlement_flag_date, settlement_status, settlement_date, settlement_amount,
settlement_percentage, settlement_term

# High-missingness (>70% null — create binary flag then drop original):
mths_since_last_record (84.11%), mths_since_last_major_derog (74.31%),
mths_since_recent_bc_dlq (77.01%),
dti_joint (94.66%), annual_inc_joint (94.66%), verification_status_joint (94.88%)

# KEEP with missing flag + imputation (38-68% missing, feed to WOE/IV):
# mths_since_recent_revol_delinq (67.25%), mths_since_last_delinq (51.25%),
# il_util (47.28%), mths_since_rcnt_il (40.25%),
# open_act_il, open_il_12m, open_il_24m, total_bal_il, open_acc_6m,
# open_rv_12m, open_rv_24m, max_bal_bc, all_util, inq_fi, total_cu_tl,
# inq_last_12m (all ~38.31% — 1.39M records populated, DO NOT drop)

#### Leakage Variables (use for LGD/EAD only, NOT PD):
out_prncp, out_prncp_inv, total_pymnt, total_pymnt_inv, total_rec_prncp,
total_rec_int, total_rec_late_fee, recoveries, collection_recovery_fee,
last_pymnt_amnt, last_pymnt_d, last_fico_range_high, last_fico_range_low,
next_pymnt_d, last_credit_pull_d, hardship_flag, debt_settlement_flag

#### PD Model Features (use for PD scorecard and ML models):
# Borrower: annual_inc, emp_length, home_ownership, verification_status, dti
# Credit: fico_range_low, fico_range_high, earliest_cr_line, open_acc, total_acc,
#   revol_util, revol_bal, pub_rec, delinq_2yrs, inq_last_6mths, mths_since_last_delinq
# Loan: loan_amnt, term, int_rate, grade, sub_grade, purpose, installment, funded_amnt
# Geographic: addr_state, zip_code
# Bureau (extended): acc_open_past_24mths, avg_cur_bal, bc_open_to_buy, bc_util,
#   mo_sin_old_rev_tl_op, mo_sin_rcnt_rev_tl_op, mo_sin_rcnt_tl, mort_acc,
#   num_actv_bc_tl, num_actv_rev_tl, num_bc_sats, num_bc_tl, num_il_tl,
#   num_op_rev_tl, num_rev_accts, num_rev_tl_bal_gt_0, num_sats,
#   num_tl_op_past_12m, pct_tl_nvr_dlq, percent_bc_gt_75,
#   pub_rec_bankruptcies, tax_liens, tot_cur_bal, tot_hi_cred_lim,
#   total_bal_ex_mort, total_bc_limit, total_il_high_credit_limit,
#   total_rev_hi_lim, num_accts_ever_120_pd, num_tl_30dpd, num_tl_90g_dpd_24m,
#   inq_last_12m, total_cu_tl, chargeoff_within_12_mths, collections_12_mths_ex_med,
#   acc_now_delinq, delinq_amnt, mths_since_recent_bc, mths_since_recent_inq, max_bal_bc
# Macro (from FRED merge): UNRATE, CSUSHPINSA, A191RL1Q225SBEA, CPIAUCSL, DFF, UMCSENT

### Feature Engineering
- WOE-IV approach for scorecard (Notebook 02)
- Macro features from FRED API merged by origination month (Notebook 01):
  - UNRATE (unemployment rate)
  - CSUSHPINSA (Case-Shiller HPI, home prices)
  - A191RL1Q225SBEA (GDP growth rate)
  - CPIAUCSL (Consumer Price Index)
  - DFF (Federal Funds Rate)
  - UMCSENT (University of Michigan Consumer Sentiment)
- Macro features are included in all parquet files (train, val, test) and are used in:
  - ML PD models (Notebook 04) — as raw covariates alongside borrower features
  - Stress testing and scenario analysis (Notebooks 08, 09) — macro overlay on flow rates
  - ECL computation (Notebook 07) — conditional PD adjustment
- Macro features are NOT used in the logistic regression scorecard (Notebook 03).
  In linear models, macro features confound with LC's growth trajectory across the
  time-based split (e.g., UNRATE falls 2010→2015 while LC scales up, causing inverted
  coefficient signs). Tree-based ML models handle this non-linearity correctly.
- WOE binning candidates include grade (behavioral feature — loan already on books).
  Exclude int_rate and sub_grade (mechanically determined by grade, near-perfect collinearity).

### PD Models

- **Behavioral PD Scorecard — Logistic Regression (Notebook 03):**
  - Purpose: Behavioral scorecard for portfolio monitoring. These are loans already
    on the books — grade is a known, observed attribute (not a model output).
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
  - Target (V5.1 CORRECTED): AUC 0.68-0.72, Gini 36-44%, KS 26-32%

- **ML PD Models — XGBoost / LightGBM (Notebook 04):**
  - Purpose: Performance ceiling model with all available information
  - Features: all borrower features + grade + int_rate + sub_grade + macro features
  - XGBoost and LightGBM with Optuna tuning
  - SHAP analysis shows macro + borrower feature importance
  - This is where macro features belong — tree models handle non-linear
    interactions with LC's growth trajectory correctly
  - Target: AUC 0.71-0.73 (realistic ceiling for tree-based models on origination-only data with temporal split)
    - SHAP-based feature selection: 101 → 50 features with <0.001 AUC loss
    - Macro features retained by SHAP: CSUSHPINSA, DFF, UNRATE (3 of 6)
    - Performance ceiling consistent with proper methodology (no leakage features)

### Basel Framework Integration
- Basel requires models to be validated across economic regimes (full cycle)
- Macro features (FRED merge) provide cycle information and are used in:
  - ML PD models (Notebook 04) as direct covariates
  - Stress testing scenarios (Notebook 09) as flow rate multipliers
  - ECL computation (Notebook 07) for conditional PD adjustment
- The logistic regression scorecard captures cycle effects indirectly through
  grade (which embeds LC's risk assessment at origination, reflecting macro
  conditions) and through out-of-time validation demonstrating stability
- Flow rates computed monthly to capture delinquency migration patterns

### LGD Model Structure
- Two-stage approach:
  - **Step 1 (Classification)**: LogisticRegression predicts P(any recovery)
  - **Step 2 (Regression)**: GradientBoostingRegressor predicts recovery_rate | recovery
  - Combined LGD = 1 - P(recovery) × E[recovery_rate | recovery]
  - NOTE: "Step 1" and "Step 2" are model construction stages, NOT IFRS-9 stages
- LGD formula (V4 UPDATE): LGD = 1 - ((recoveries - collection_recovery_fee) / out_prncp)
  - Use defaulted loans only
  - Portfolio avg should be ~0.83 (from LendingClub 10-K)
  - Cross-check with: LGD_simple = 1 - (total_rec_prncp / out_prncp)
- Default recovery_rate should be seeded from the LGD model's portfolio-level output
  - LGD ≈ 83% → recovery_rate default = 0.17

### EAD Model
- Filter to defaulted loans
- Target: out_prncp (outstanding principal at default)
- Compute CCF = out_prncp / funded_amnt per loan
- Random Forest regressor
- Target MAPE < 15%
- Note: EAD ≈ 1 for fully-drawn term loans (simplifying assumption)

### Prepayment Model
- Competing risk alongside default
- Identify prepaid loans: "Fully Paid" status where actual_life < 0.8 × contractual_term
- Model approach: Kaplan-Meier survival curves by segment or empirical CPR lookup by term × grade × vintage
- Output: empirical prepayment rates by term (36 vs 60), vintage, grade
- Used in DCF-ECL (Notebook 07) and Streamlit forecasting
- Prepayment rates feed liquidation curves for operational and CECL forecasts
- NOTE: The LendingClub public dataset provides loan-level terminal outcomes, not monthly payment history.
  We use survival analysis on time-to-event data rather than month-level hazard modeling.
  The empirical CPR lookup table is the standard industry approach.

### ECL Computation
- Simple ECL: PD × EAD × LGD (by grade, vintage, purpose)
- DCF-ECL (LendingClub 10-K methodology):
  - Monthly cash flow projection with competing risks:
    - P(stay current) × payment
    - P(default) × recovery value
    - P(prepay) × remaining balance
  - Discount at effective interest rate
  - Incorporate prepayment rates from Notebook 5.5
  - ECL = Contractual CF (NPV) - Expected CF (NPV)

### Flow Rates and Receivables Tracking
- Use institutional format: Receivables Tracker with dollar balances by DPD bucket
- Flow rates = simple bucket ratios (NOT account-level transition matrices)
- Example: 30+ Flow Rate = 30 DPD(t) / Current(t-1)
- **FORWARD-ONLY FLOW RATES**: Current → 30+ → 60+ → ... → GCO
  - Curing is unobservable; intermediate delinquencies for performing loans are invisible
  - Flow rates represent worsening transitions only
  - Balances for delinquent months are approximate (scheduled, not actual with penalty interest)
  - **Synthetic Monthly Panel**: Monthly DPD status is back-calculated from loan-level
    terminal outcomes using amortization schedules and charge-off timing
- Segment by grade and vintage
- Track trends and identify acceleration patterns
- **Flow Through Rate (NEW)**: Diagonal multiplication of all intermediate rates
  - FTR = (Current→30+) × (30+→60+) × ... × (150+→180+) × (180+→GCO)
  - Tracks cumulative delinquency progression
  - Cross-check against PD model
  - **Note**: Based on synthetically reconstructed monthly status. Production implementation
    with observed payment data would enable two-way transitions and curing rates.

### ECL Views and Adjustment Methods
- **Pre-FEG**: pure model output, no adjustments
  - Rolling 6-month average flow rates
  - No macro overlay
- **Central**: baseline macro overlay applied to flow rates
  - Macro-adjusted flow rates per scenario
- **Post-FEG**: weighted across scenarios + qualitative adjustments
  - Compute in Notebook 09 (stress scenarios)
  - Multi-scenario average with expert judgment layer
- **NOTE**: Flow rates derived from synthetic monthly panel reconstruction.
  Production implementation would use observed monthly DPD data.

### Stress Testing (Macro Scenarios)
- Stress at flow rate level, NOT final ECL
- Multiplicative stress on individual flow rates preserves non-linear dynamics
- Example: 15% stress per flow rate → ~75% increase in cumulative flow-through
- Three competing risks per month: current → delinquent, default, prepay
- Show flow rate stress vs output-level stress comparison
- FEG toggle (Pre-FEG/Central/Post-FEG) applies to BOTH Operational and CECL modes
- Stress at flow rate level applies to both modes
- In Operational mode, stressed rates extend flat; in CECL mode, stressed rates apply during Phase 1 only then revert

### Streamlit Forecasting Engine (ECLProjector)
- **Redesigned in V3**:
  - __init__ takes ONLY pd_model, lgd_model (NOT flow_rates)
  - New method: compute_forecast_flow_rates(lookback_months=6, method='extend'|'cecl',
    rs_period_months=24, reversion_method='straight_line')
  - Flow rates loaded from receivables tracker, computed on-demand
  - Supports dual-mode forecasting:
    - **'extend' mode** (operational): rolling average extrapolation
    - **'cecl' mode** (regulatory): R&S period + reversion to historical avg
- New module: app/engine/prepayment.py for prepayment rate computation
- Output: monthly balances, GCO, recovery, NCO, ECL, ALLL ratio
- Export to Excel in institutional format receivables tracker format
- New Originations input is only active in Operational mode; hidden ($0) in CECL mode
  - Reason: CECL reserves are for existing portfolio only
- **Data limitation note**: Flow rates derived from synthetic monthly panel reconstruction.
  Production implementation would use observed monthly payment data.

### Streamlit UI Enhancements
- ECL page includes mode selector: "Operational Forecast" vs "CECL Reserve Estimation"
- FEG toggle: Pre-FEG | Central | Post-FEG radio buttons
- Flow Through Rate KPI on portfolio dashboard + ECL page
- Dual liquidation factor UI: simple slider (operational) vs term-level (CECL)
- Upload/Export Assumptions buttons on ECL page
- Scenario page: side-by-side flow rate stress comparison visualization
- Data limitation disclaimer in sidebar: "Flow rates derived from synthetic monthly panel reconstruction.
  Production implementation would use observed monthly payment data."

### Model Validation (Notebook 08)
- Discrimination: AUC, Gini, KS, CAP curve, bootstrap CI
- Calibration: Hosmer-Lemeshow, decile calibration, Brier score
- Stability: PSI, CSI, VDI for each test period
- **PD Scorecard RAG (V5.1 CORRECTED):** Green (Gini ≥ 42%), Amber (36-42%), Red (< 36%)
- **ML Model RAG:** Green (Gini ≥ 46%), Amber (42-46%), Red (< 42%)
- Out-of-time performance by vintage year (2016, 2017, 2018)
- Backtesting: predicted cumulative default rate vs actual cumulative default rate by vintage
  (Note: Uses flow rates derived from synthetic monthly panel reconstruction)
- External validation: PSI against benchmark_population_2014.csv

### External Benchmark Validation (V4 NEW)
- Load benchmark_population_2014.csv from data/raw/
  - 200,000 records from JUN-AUG 2014
  - Contains: FICO score, delinquency bucket, PERFORMANCE_OUTCOME (GOOD/BAD)
- Use for:
  1. PSI computation: Compare your PD model's score distribution vs benchmark FICO
  2. External calibration: Score benchmark population, compare predicted vs actual
  3. Display as: PSI table with RAG status + calibration chart
  4. Interview framing: "I validated my model against LendingClub's benchmark population,
     mirroring institutional approaches to validation"

### Coding Standards
- All src/ functions: docstrings + type hints
- Notebooks: markdown explanation cells before code cells
- No warnings in output; handle deprecations explicitly
- Use pd.eval, chunking for large datasets to manage memory
- Git commit AND push after each session with clear messages (never commit data/raw/ or .parquet files)

### Prior Role Connection
- PD scorecard: behavioral scorecard monitoring + RAG framework.
  Grade is included as a behavioral feature (loan already on books).
  Macro features reserved for ML models and stress testing.
- WOE/IV binning: mirrors VantageScore/FICO binning, utilization, DTI analysis.
  Grade is WOE-binned alongside borrower features (expected IV > 0.5).
- Receivables tracking: institutional receivables tracker format with dollar balances
- Flow rate analysis: operational KPI (replaces account-level roll rates)
- Vintage analysis: curve analysis + MOB analysis
- ECL: DCF methodology from institutional IMR framework
- Macro scenario: stress testing with 3-scenario weighting
- Model validation: quarterly monitoring RAG report
- External benchmark: mirroring institutional approaches to external validation
- PyCraft integration: system tools used at prior institution
- Sherwood curves: vintage analysis and MOB curves

---

3. Create a requirements.txt with all the pinned package versions listed in the roadmap
4. Create a config.py with the following constants:
   - DATA_RAW_PATH, DATA_PROCESSED_PATH, DATA_MODELS_PATH, DATA_RESULTS_PATH
   - RANDOM_STATE = 42
   - TARGET_COL = 'default'
   - DEFAULT_STATUSES = ['Charged Off', 'Default']
   - NON_DEFAULT_STATUSES = ['Fully Paid']
   - DROP_STATUSES = ['Current', 'In Grace Period', 'Late (16-30 days)',
     'Late (31-120 days)', 'Does not meet the credit policy. Status:Fully Paid',
     'Does not meet the credit policy. Status:Charged Off']
   - TRAIN_END_YEAR = 2015
   - VAL_YEAR = 2016
   - TEST_START_YEAR = 2017
   - SCORECARD_BASE = 600
   - SCORECARD_PDO = 20
   - PSI_GREEN, PSI_AMBER, PSI_RED thresholds
   - GRADE_ORDER = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
   - Add FRED series list: FRED_SERIES = ['UNRATE', 'CSUSHPINSA', 'A191RL1Q225SBEA',
     'CPIAUCSL', 'DFF', 'UMCSENT']
5. Create empty __init__.py files in src/ and app/ directories
6. Create a .gitignore with these entries:
   data/raw/
   data/processed/*.parquet
   *.pkl
   *.joblib
   .env
   __pycache__/
   .ipynb_checkpoints/
   *.csv
   !data/raw/benchmark_population_2014.csv
   .DS_Store
   venv/
   .venv/

7. Initialize a git repository and make the first commit:
   git init
   git add .
   git commit -m "Initial project scaffolding: directory structure, CLAUDE.md, config.py"

8. Create the GitHub remote and push:
   gh repo create lendingclub-credit-risk --private --source=. --push
   NOTE: The repo starts PRIVATE. Flip to public when interview-ready.
   VERIFY: Run 'git ls-files' and confirm NO .csv or .parquet files are tracked.
   If any data files slipped through, fix .gitignore and amend before pushing.

9. Create a download_data.py script in the project root with:
   - A function that downloads the LendingClub dataset from Kaggle using the kaggle API:
     kaggle datasets download -d wordsforthewise/lending-club -p data/raw/ --unzip
   - A fallback message if kaggle API is not configured, with manual download URL:
     https://www.kaggle.com/datasets/wordsforthewise/lending-club
   - Instructions to also download benchmark_population_2014.csv if not present
   - Print expected file sizes after download:
     accepted_2007_to_2018Q4.csv: ~1.6 GB, 2,260,701 rows
     rejected_2007_to_2018Q4.csv: ~1.7 GB, 27.6M rows
     benchmark_population_2014.csv: ~7.7 MB, 200,000 rows

10. Add a "Data Setup" section to the README.md with:
    - Project title and one-line description
    - Prerequisites (Python 3.10+, packages)
    - Data download instructions (both kaggle CLI and manual)
    - Expected directory structure after setup
    - NOTE: "The raw data files are not included in this repository due to size
      constraints. Follow the instructions below to download them."

11. Create the initial virtual environment and install requirements

12. Make a second commit and push:
    git add download_data.py README.md
    git commit -m "Add data download script and README with setup instructions"
    git push

Don't create any notebooks yet — just the infrastructure.
```

### What to verify after:
- Directory structure matches the roadmap
- CLAUDE.md exists at root with all Known Data Quirks and Column Categories
- config.py has all constants including FRED_SERIES
- Virtual environment is created and packages install without errors
- Git is initialized with proper .gitignore
- GitHub remote is created and first push succeeded
- Run `git ls-files` — NO .csv, .parquet, .pkl, or .env files should appear
- download_data.py exists and prints instructions when run without kaggle API
- README.md has a Data Setup section with download instructions

---

## Session 1: Notebook 01 — EDA and Data Cleaning (with FRED Macro Integration)

### Pre-requisite:
Download the wordsforthewise dataset from Kaggle and place it in `data/raw/`. The file is typically called `accepted_2007_to_2018Q4.csv` (or similar).

### Prompt:
```
Read the roadmap at docs/LendingClub_Credit_Risk_Project_Roadmap_V5.md and
the CLAUDE.md file at the project root for Known Data Quirks and Column Categories.

Build Notebook 01: EDA and Data Cleaning WITH FRED Macroeconomic Data Integration.

INPUT:
- Raw dataset at data/raw/accepted_2007_to_2018Q4.csv (~2.26M records, 151 features)
- Configuration from config.py (including FRED_SERIES and column drop lists)

PROCESS — the notebook should do the following in order:

STEP 1: Load data
- File: data/raw/accepted_2007_to_2018Q4.csv
- CRITICAL: After loading, drop rows where loan_amnt is null or non-numeric.
  The raw CSV has 33 footer/summary rows that are not loan records.
  After cleanup, you should have exactly 2,260,668 rows.
- Print shape, dtypes, first 5 rows

STEP 2: Drop Tier 1 columns (confirmed >93% missing or non-feature)
Drop these columns immediately:
- 100% empty: member_id
- Non-feature: id, url, pymnt_plan (constant 'n'), policy_code (constant 1)
- 94-95% empty (joint app / near-empty): sec_app_fico_range_low, sec_app_fico_range_high,
  sec_app_earliest_cr_line, sec_app_inq_last_6mths, sec_app_mort_acc, sec_app_open_acc,
  sec_app_revol_util, sec_app_open_act_il, sec_app_num_rev_accts,
  sec_app_chargeoff_within_12_mths, sec_app_collections_12_mths_ex_med,
  sec_app_mths_since_last_major_derog, revol_bal_joint,
  desc (94.42% empty), dti_joint (94.66%), annual_inc_joint (94.66%),
  verification_status_joint (94.88%)
- Hardship/settlement (>97% null): hardship_type, hardship_reason, hardship_status,
  deferral_term, hardship_amount, hardship_start_date, hardship_end_date,
  payment_plan_start_date, hardship_length, hardship_dpd, hardship_loan_status,
  hardship_payoff_balance_amount, hardship_last_payment_amount,
  orig_projected_additional_accrued_interest, debt_settlement_flag_date,
  settlement_status, settlement_date, settlement_amount, settlement_percentage,
  settlement_term

DO NOT drop these columns (they were incorrectly listed in V4 as >70% missing):
- open_act_il, open_il_12m, open_il_24m, total_bal_il, open_acc_6m,
  open_rv_12m, open_rv_24m, max_bal_bc, all_util, inq_fi, total_cu_tl,
  inq_last_12m (all ~38.31% missing — keep with missing flag + imputation)
- il_util (47.28% missing — keep with missing flag)
- mths_since_rcnt_il (40.25% missing — keep with missing flag)
- mths_since_recent_revol_delinq (67.25% — keep with missing flag)
- mths_since_last_delinq (51.25% — keep with no_delinq_history flag)

Print number of columns before/after.

STEP 3: Parse data types
- term: df['term'] = df['term'].str.strip().str.extract('(\d+)').astype(int)
  Values should be exactly {36, 60}
- emp_length: Map text to numeric:
  '< 1 year' → 0, '1 year' → 1, ..., '9 years' → 9, '10+ years' → 10
  NaN stays NaN → create emp_length_unknown flag
- issue_d: pd.to_datetime(df['issue_d'], format='%b-%Y')
  Sample values: 'Dec-2015', 'Nov-2015' etc.
- earliest_cr_line: pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
- last_pymnt_d: pd.to_datetime(df['last_pymnt_d'], format='%b-%Y')
  NOTE: 0.07% missing — these are likely current loans with no payments yet
- int_rate: already float64, no conversion needed (values like 13.99, 11.99)
- revol_util: already float64, no conversion needed (values like 29.7, 56.2)
Print sample of parsed values for each field.

STEP 4: Filter to terminal loan statuses and create target
- default=1: loan_status in ['Charged Off', 'Default']
- default=0: loan_status in ['Fully Paid']
- DROP: ['Current', 'In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)',
         'Does not meet the credit policy. Status:Fully Paid',
         'Does not meet the credit policy. Status:Charged Off']
- SANITY CHECK: After filtering, you should have ~1,345,350 terminal loans
  with a default rate of ~19.96% (268,599 defaults / 1,345,350 total)
- Print counts and percentages for each status before/after filtering

STEP 5: Merge FRED macroeconomic data
- Create issue_month column from issue_d for merge key (YYYY-MM format)
- Pull 6 series from FRED API: UNRATE, CSUSHPINSA, A191RL1Q225SBEA, CPIAUCSL, DFF, UMCSENT
- Use fredapi library with API key from environment variable FRED_API_KEY
- FALLBACK: if API unavailable, download CSVs manually from FRED website and merge
- For quarterly GDP (A191RL1Q225SBEA): forward-fill within quarter
- Merge onto loans by issue_month
- CRITICAL NOTE (in markdown): "These macro features are CRITICAL for the time-based
  split validation. Without macro covariates, the model would overpredict defaults
  on the 2017-2018 test set because the training period (2007-2015) covered the
  2008 crisis and recovery. Macro features enable the model to learn cycle-adjusted
  default patterns."
- Print statistics on merged macro data (mean, std, range per feature)

STEP 6: Analyze and handle missing values (for remaining columns):
- Print missingness % for ALL remaining features, sorted descending
- Tier 2 columns (70-85% missing — create flag then drop the original):
  * mths_since_last_record (84.11%) → create has_public_record flag, then drop
  * mths_since_recent_bc_dlq (77.01%) → create has_recent_bc_delinq flag, then drop
  * mths_since_last_major_derog (74.31%) → create has_major_derog flag, then drop
- Tier 3 columns (38-68% missing — create flag AND keep with imputation):
  * mths_since_recent_revol_delinq (67.25%) → create no_revol_delinq flag, fill median
  * mths_since_last_delinq (51.25%) → create no_delinq_history flag, fill median
  * il_util (47.28%) → create il_util_missing flag, fill median
  * mths_since_rcnt_il (40.25%) → create flag, fill median
  * open_act_il, open_il_12m, open_il_24m, total_bal_il, open_acc_6m, open_rv_12m,
    open_rv_24m, max_bal_bc, all_util, inq_fi, total_cu_tl, inq_last_12m
    (all ~38.31%) → create single installment_features_missing flag, fill each with median
  * IMPORTANT: These features are 61.69% populated across 1.39M records.
    V4 incorrectly estimated 78-81% missing from a 100K sample. Keep ALL for WOE/IV.
- Tier 4 columns (<10% missing):
  * emp_length (5.64%): create emp_length_unknown flag, fill median
  * mths_since_recent_inq (13.07%): create flag, fill median
  * remaining <1% missing: fill with median (numeric) or mode (categorical)
- RULE: for ANY feature with >10% missing, ALWAYS create a binary flag before imputing
- Print final feature count and missingness summary

STEP 7: Feature categorization — create markdown sections grouping features into:
- Borrower Demographics: annual_inc, emp_length, home_ownership, verification_status, dti, addr_state, zip_code
- Credit History (Core): fico_range_low, fico_range_high, earliest_cr_line, open_acc, total_acc,
  revol_util, revol_bal, pub_rec, delinq_2yrs, inq_last_6mths, mths_since_last_delinq
- Loan Characteristics: loan_amnt, term, int_rate, grade, sub_grade, purpose, installment, funded_amnt
- Credit Bureau (Extended): ~30 columns listed in CLAUDE.md
- Macroeconomic: UNRATE, CSUSHPINSA, A191RL1Q225SBEA, CPIAUCSL, DFF, UMCSENT
- Payment History (Leakage): total_pymnt, total_rec_prncp, recoveries, collection_recovery_fee, etc.

STEP 8: Create the time-based train/val/test split using issue_d:
- Train: issue_d year 2007-2015
- Validation: issue_d year 2016
- Test: issue_d year 2017-2018
- Save as: data/processed/train.parquet, val.parquet, test.parquet
- ALL files must include macro features
- Print split sizes and default rates per split
- Verify splits are non-overlapping by date

STEP 9: EDA visualizations (use matplotlib/seaborn, with clear titles and labels):
- Default rate by grade (A-G) — should be monotonically increasing
  Expected: A(~5-6%), B(~11%), C(~16%), D(~21%), E(~26%), F(~30%), G(~34%)
- Default rate by sub_grade — finer granularity
- Default rate by origination year (vintage)
- Default rate by term (36 vs 60 months) — expect 60-month to be higher
- Distribution of annual_inc, dti, fico_range_low by default status (overlaid KDEs)
- Correlation matrix of top 20 numeric features (heatmap)
- Geographic default rate by state (choropleth preferred, or bar chart)
- Portfolio composition over time: origination volume by grade per year (stacked bar)
- (NEW) Macro trends: plot UNRATE, HPI, GDP, CPI, DFF over time,
  color-coded by origination vintage (time series with vertical lines for regime changes)

STEP 9.5: COMPREHENSIVE DATA PROFILING (NEW — full-file validation)
This step generates a detailed profiling report. Save all results to
data/results/full_profiling_report.json and document key findings in markdown.

A. OUTLIER DETECTION (all numeric columns after Tier 1 drop):
- For each numeric column:
  * Compute Q1, Q3, IQR = Q3-Q1
  * Count values outside [Q1 - 3*IQR, Q3 + 3*IQR]
  * Compute p1, p5, p95, p99 percentiles
- PRIORITY outlier treatment (apply these BEFORE saving cleaned data):
  * annual_inc: cap at 99th percentile (max is $110M — clearly erroneous)
  * dti: set values < 0 or > 100 to NaN, then impute (range is [-1, 999])
  * revol_util: cap at 150% (max = 892.3; over-limit is possible up to ~150%)
  * last_fico_range_low/high: set values = 0 to NaN (FICO of 0 is invalid)
  * tot_coll_amt, tot_cur_bal, tot_hi_cred_lim, total_rev_hi_lim:
    cap at 99.5th percentile (max values are $9-10M, likely data errors)
  * bc_util, il_util, all_util: cap at 200% (some over-limit is real)
  * settlement_percentage: cap at 100% (max = 521% is data error)
- Print outlier summary table: column | n_outliers | pct | min | p1 | p99 | max | treatment
- Save to profiling report

B. DISTRIBUTION ANALYSIS (all numeric columns):
- Compute for each: mean, median, std, skewness, kurtosis
- Classify shape: Normal (|skew|<0.5), Moderate Skew (0.5-1), High Skew (>1)
- Flag features that may need log-transform for logistic regression (|skew|>2)
- Expected high-skew features: annual_inc, revol_bal, tot_cur_bal, total_pymnt,
  recoveries, collection_recovery_fee, delinq_amnt
- Plot KDE overlays (default=0 vs default=1) for: fico_range_low, dti, annual_inc,
  int_rate, revol_util, open_acc, total_acc, inq_last_6mths
- Print distribution summary table: column | mean | median | skew | kurtosis | shape

C. CORRELATION ANALYSIS:
- Compute Pearson correlation matrix for ALL numeric features
- Extract all pairs with |corr| > 0.80 — print as sorted table
- KNOWN high correlations to verify (don't alarm on these):
  * loan_amnt ↔ funded_amnt ↔ funded_amnt_inv ↔ installment (near 1.0)
  * fico_range_low ↔ fico_range_high (exactly 4-point offset, corr=1.0)
  * last_fico_range_low ↔ last_fico_range_high (same pattern)
  * out_prncp ↔ out_prncp_inv (near 1.0)
  * total_pymnt ↔ total_pymnt_inv (near 1.0)
- For each redundant pair, recommend which to KEEP:
  * Keep loan_amnt, drop funded_amnt, funded_amnt_inv
  * Keep fico_range_low, drop fico_range_high (or average them)
  * Keep last_fico_range_low, drop last_fico_range_high
  * Keep out_prncp, drop out_prncp_inv
  * Keep total_pymnt, drop total_pymnt_inv
- Generate annotated heatmap (top 30 features by non-null count)
- Save correlated pairs to profiling report

D. FULL-FILE SANITY CHECKS:
- Total rows after footer removal = 2,260,668 (if not, STOP and report)
- Terminal loans (Fully Paid + Charged Off + Default) ≈ 1,345,350
- Default rate ≈ 19.96%
- Grade monotonicity: default_rate(A) < B < C < D < E < F < G
- term values exactly {36, 60}
- emp_length parsed values in {0,1,2,...,10,NaN}
- int_rate range [5.31, 30.99], mean ≈ 13.09
- fico_range_low range [610, 845], mean ≈ 698.59
- loan_amnt range [500, 40000], mean ≈ $15,047
- No duplicate loans (check unique combinations of loan_amnt + issue_d + int_rate + annual_inc)
- Accounting check: total_pymnt ≈ total_rec_prncp + total_rec_int + total_rec_late_fee + recoveries

E. DEEP CATEGORICAL VALUE COUNTS:
- For ALL categorical features with <50 unique values: print full value_counts with pct
- Verify: loan_status has 9 values, home_ownership has {RENT, MORTGAGE, OWN, OTHER, NONE, ANY},
  purpose has 14 categories (debt_consolidation dominant ~47%), application_type is
  {Individual ~95%, Joint App ~5%}, grade is {A,B,C,D,E,F,G}, sub_grade is 35 values
- For emp_title: just unique count + top 20 (high cardinality, will be dropped)
- For zip_code, addr_state: unique count + top 10 + geographic distribution
- Document any unexpected values

SAVE all profiling results to data/results/full_profiling_report.json with structure:
{
  "outliers": {column: {n_outliers, pct, p1, p99, treatment}},
  "distributions": {column: {mean, median, skew, kurtosis, shape}},
  "correlations": {pair: {corr, recommendation}},
  "sanity_checks": {check_name: {expected, actual, status}},
  "categorical_counts": {column: {value: count}}
}

STEP 10: Save outputs

OUTPUT:
- data/processed/loans_cleaned.parquet (the full cleaned dataset with target column AND macro features)
- data/processed/train.parquet, val.parquet, test.parquet (the splits, WITH macro features)
- data/results/eda_summary.json (key statistics: total records, default rate,
  features kept, records per split, macro feature stats, sanity check values)
- data/results/full_profiling_report.json (comprehensive profiling: outliers, distributions,
  correlations, sanity checks, categorical counts — from Step 9.5)

QUALITY STANDARDS:
- Every code cell should be preceded by a markdown cell explaining what it does and WHY
- The first markdown cell should explain: "This mirrors the data preparation phase of
  behavioral scorecard monitoring and loss forecasting in my prior role. Macro features enable
  validation across economic regimes. Our time-based split tests the model's ability
  to generalize across the 2007-2015 crisis/recovery (train), 2016 expansion (val),
  and 2017-2018 (test)."
- Print intermediate sanity checks: default rate should be ~19.96% after filtering
- Default rate monotonic by grade: verify A < B < C < D < E < F < G
- All plots should have proper titles, axis labels, and be publication-quality
- No warnings in the output — handle deprecation warnings with proper code
- Macro features must be present in all output parquet files and documented in eda_summary.json
- Flag any anomalies (e.g., negative values in utilization ratios, income outliers)
```

### What to verify after:
- Default rate is ~19.96% (reasonable for terminal statuses only)
- Default rate increases monotonically from Grade A to G
- Train/val/test splits are non-overlapping by time
- No data leakage (test set features don't use post-origination info)
- Parquet files are saved and loadable
- Macro features (UNRATE, HPI, CPI, etc.) are present in train/val/test parquet files
- Macro data has no unexpected missing values (should merge cleanly by month)
- EDA notebook is publication-ready with clear markdown and visualizations
- Outlier treatment applied: annual_inc capped, dti cleaned, revol_util capped, FICO 0s handled
- full_profiling_report.json is saved and contains all 5 sections (A-E)
- No columns from the "DO NOT drop" list were accidentally dropped
- Binary missing flags created for all Tier 2 and Tier 3 columns before imputation
- Correlation pairs >0.80 documented and redundant columns identified for later removal
- **Git:** Commit and push after successful verification:
  git add notebooks/01_eda_data_cleaning.ipynb data/results/ src/
  git commit -m "Notebook 01: EDA, data cleaning, FRED macro integration, full profiling"
  git push

---

## Session 2: Notebook 02 — WOE/IV Feature Engineering

### Prompt:
```
Read the roadmap section "Days 3-4: WOE/IV Analysis & Feature Engineering" and
CLAUDE.md for the complete list of PD Model Features.

Build Notebook 02: WOE/IV Feature Engineering.

Also build the reusable module: src/woe_binning.py

INPUT:
- data/processed/train.parquet (training data only — never fit WOE on test data)
- config.py constants

PROCESS:

Part A — Build src/woe_binning.py first:
- A WOEBinner class that wraps optbinning.OptimalBinning
- Methods: fit(X, y), transform(X), fit_transform(X, y)
- Should compute and store: bin edges, WoE values, IV for each feature
- Method to generate a summary table: feature, bin, count, event_count,
  non_event_count, event_rate, woe, iv
- Method to plot WoE bins for a given feature
- Method to check monotonicity of bad rates across bins

Part B — Build Notebook 02:
1. Load training data from data/processed/train.parquet
2. Create engineered features:
   - credit_history_years = years since earliest_cr_line
   - fico_avg = (fico_range_low + fico_range_high) / 2
   - loan_to_income = loan_amnt / annual_inc (handle div by zero)
   - installment_to_income = installment / (annual_inc / 12)
   - total_credit_utilization = revol_bal / total_rev_hi_lim (handle div by zero)
   - delinq_flag = 1 if delinq_2yrs > 0
   - recent_inquiry_flag = 1 if inq_last_6mths > 2
   - high_dti_flag = 1 if dti > 30
3. Apply WOE binning to ALL candidate features (65-75 from CLAUDE.md):
   Include all Borrower, Credit, Loan, Geographic, and Bureau (extended) features
   Include macro features (UNRATE, CSUSHPINSA, A191RL1Q225SBEA, CPIAUCSL, DFF, UMCSENT)
   INCLUDE grade as a candidate feature (behavioral scorecard — grade is known)
   EXCLUDE int_rate and sub_grade (collinear with grade)
   DO NOT include leakage variables
4. Compute IV for all features and print a ranked IV summary table
   IMPORTANT: The dataset has ~65-75 candidate features. The IV screening will narrow
   this to ~25-35 features. Let the data decide — don't pre-filter based on assumptions.
5. Apply IV-based selection:
   - IV < 0.02: drop (not predictive)
   - IV 0.02-0.1: weak (flag for review)
   - IV 0.1-0.3: medium (include)
   - IV 0.3-0.5: strong (include)
   - IV > 0.5: investigate for data leakage
6. For grade: it will have very high IV (expected > 0.5).
   This is EXPECTED and NOT leakage — grade is a behavioral feature for the
   behavioral scorecard. Grade will be the strongest predictor and is included.
   int_rate and sub_grade are excluded (collinear with grade).
7. Validate monotonicity: plot bad rate by bin for each selected feature.
   Flag any rank-ordering breaks (acceptable if minor and documented).
8. Transform all selected features to WOE values
9. Save outputs

EXPECTED RESULTS:
- grade will have very high IV (>0.5) — INCLUDE in WOE binning for behavioral scorecard
- int_rate and sub_grade: DO NOT bin (excluded — collinear with grade)
- fico_range_low/high should have IV ~0.3-0.4 (strong predictors)
- annual_inc and dti should have IV ~0.1-0.2 (medium predictors)
- Macro features will have low IV (< 0.02) in WOE binning — this is expected.
  Their value emerges in ML models and stress testing, not in univariate IV screening.
- Final WOE-binned features should be 25-35 (from initial ~70)

OUTPUT:
- src/woe_binning.py (the reusable module)
- data/processed/woe_binning_results.pkl (fitted WOEBinner object)
- data/processed/iv_summary.csv (feature, IV, selection_status)
- data/processed/train_woe.parquet (WOE-transformed training data)
- data/processed/val_woe.parquet (WOE-transformed validation data)
- data/processed/test_woe.parquet (WOE-transformed test data)

QUALITY STANDARDS:
- The WOEBinner class should have full docstrings and type hints
- IV summary table should be printed and also saved as CSV
- Monotonicity violations should be flagged clearly in the notebook
- First markdown cell: connect to prior role behavioral scorecard monitoring
  (WoE, IV for VantageScore/FICO bins, utilization, DTI, inquiries)
- WOE transformation must be fit ONLY on training data, then applied
  to val/test — this is critical for preventing data leakage
- Include IV summary statistics: mean IV, median IV, count by selection status
```

### What to verify after:
- IV values make intuitive sense (FICO should be high, random features should be low)
- grade IS included in WOE binning and has very high IV (> 0.5)
- int_rate and sub_grade are EXCLUDED from WOE binning (collinear with grade)
- WOE is fit on train only, applied to val and test
- Monotonicity holds for most features (some minor violations are acceptable)
- No features with IV > 0.5 are suspiciously leaky (like total_pymnt or last_pymnt)
  — grade having IV > 0.5 is expected and legitimate for a behavioral scorecard
- Macro features may have low IV in univariate screening — this is fine
- **Git:** Commit and push after successful verification:
  git add notebooks/02_WOE_IV_Feature_Engineering.ipynb src/woe_binning.py data/processed/ data/results/
  git commit -m "Notebook 02: WOE/IV binning with grade included"
  git push

---

## Session 3: Notebook 03 — Behavioral PD Scorecard (Logistic Regression)

### Prompt:
```
Read the roadmap section "Days 5-7: PD Models — Scorecard" and CLAUDE.md.

Build Notebook 03: Behavioral PD Scorecard (Logistic Regression).

Also build: src/scorecard.py

IMPORTANT CONTEXT — Read this first:
This is a BEHAVIORAL scorecard for portfolio monitoring. The loans are already
on the books. Grade is a known, observed attribute — not a model output.
This is NOT an origination scorecard (where you'd exclude grade because
you're building the tool that assigns it).

INPUT:
- data/processed/train_woe.parquet, val_woe.parquet, test_woe.parquet
- data/processed/woe_binning_results.pkl
- data/processed/iv_summary.csv
- config.py (SCORECARD_BASE=600, SCORECARD_PDO=20)

PROCESS:

Part A — Build src/scorecard.py:
- A Scorecard class that:
  - Takes a fitted logistic regression model + WOE binning results
  - Converts model coefficients to scorecard points per bin
  - Score formula: Score = Offset + Factor × Σ(βi × WoEi)
    where Factor = PDO / ln(2), Offset = Base_Score - Factor × ln(Base_Odds)
  - Method: score(X) → returns credit scores for each observation
  - Method: generate_scorecard_table() → returns DataFrame with
    feature, bin_range, woe, coefficient, points
  - Method: score_to_pd(score) → converts score back to PD estimate
  - Method: feature_contributions(X) → returns per-feature point breakdown

Part B — Build Notebook 03:

Cell 1 — Markdown context:
"## Behavioral PD Scorecard — Logistic Regression

This is a behavioral scorecard for portfolio monitoring. These are loans
already on LendingClub's books.

**Why grade is included:** Grade is a known, observed attribute for existing
loans. It captures LendingClub's comprehensive risk assessment at origination
and is the single strongest predictor of default.

**Why int_rate and sub_grade are excluded:** int_rate is mechanically
determined by grade (near-perfect correlation). sub_grade is a finer partition
of grade creating the same collinearity. Including any of these alongside
grade would destabilize coefficient estimates.

**Why macro features are excluded:** In a logistic regression with time-based
train/test split, macro features confound with LendingClub's growth trajectory.
UNRATE falls steadily 2010→2015 while LC's volume and borrower mix change.
This produces inverted coefficient signs (higher unemployment appears to
*decrease* default — which is economically nonsensical). Macro features are
properly handled in the ML PD models (Notebook 04) and stress testing
(Notebooks 08-09) where non-linear models can disentangle these effects."

Cell 2 — Load data and IV summary

Cell 3 — Feature selection:
a. From iv_summary.csv, select WOE features with IV >= 0.05
   Grade should be the top feature (IV > 0.5).
b. EXCLUDE from the model:
   - int_rate_woe (if present — should not be, but verify)
   - sub_grade_woe (if present — should not be, but verify)
   - All 6 macro features: UNRATE, CSUSHPINSA, A191RL1Q225SBEA, CPIAUCSL, DFF, UMCSENT
   - Any raw (non-WOE) versions of features
c. Compute pairwise Pearson correlations among remaining WOE features.
   For any pair with |r| > 0.70, drop the feature with lower IV.
   Print which pairs were checked and which features were dropped.
d. For binary flags, include ONLY those with IV >= 0.02.
   Do NOT include all one-hot encoded categoricals.
e. Print the final feature list with IV values. Target: 10-15 features.

Cell 4 — Fit logistic regression:
- LogisticRegressionCV with penalty='l2', Cs=[0.001, 0.01, 0.1, 1.0, 10.0]
- 5-fold stratified CV, solver='lbfgs', max_iter=2000
- Report best C

Cell 5 — Coefficient sign enforcement:
- Print all coefficients with feature names
- Check: ALL WOE coefficients must be negative
- If any WOE coefficient is positive:
  a. Print warning: "Positive coefficient detected for [feature]: [value]"
  b. Remove that feature
  c. Refit the model
  d. Repeat until all WOE coefficients are negative
- The intercept can be any sign. Binary flag coefficients can be any sign.
- Print final coefficient table after enforcement

Cell 6 — Scorecard table generation (publication-ready)

Cell 7 — Score all datasets

Cell 8 — Metrics:
- AUC, Gini (= 2×AUC - 1), KS, Brier score for train, val, test
- Print comparison table
- Compute train-test AUC gap (must be < 0.03)
- TARGET (V5.1 CORRECTED): AUC 0.68-0.72, Gini 36-44%, KS 26-32%
  NOTE: Previous versions stated AUC ≥ 0.75 which assumed int_rate in model.
  V5.1 excludes int_rate; these are the correct targets for Camp B methodology.
  Literature benchmark: LR on LendingClub without leakage achieves AUC 0.66-0.71.
- HONEST REPORTING: If targets are missed, say so clearly. Do NOT
  lower thresholds to make results look better.

Cell 9 — RAG status (V5.1 CORRECTED):
- Green: Gini >= 42%
- Amber: Gini 36-42%
- Red: Gini < 36%

Cell 10 — Plots:
- ROC curve (train + test overlaid)
- KS plot
- Score distribution (good vs bad)
- Calibration plot (predicted vs actual by decile)
- Coefficient bar chart

Cell 11 — Credit policy analysis:
- Score cutoffs from min to max in steps of 10
- For each: approval_rate, default_rate, expected_loss, utility
- Plot tradeoff curve
- Find optimal cutoff maximizing approval_rate × (1 - default_rate)
- VERIFY: optimal cutoff should NOT be the minimum score

Cell 12 — Grade mapping (model scores to LC grades A-G)

Cell 13 — Save all outputs

OUTPUT:
- src/scorecard.py
- data/models/pd_logreg_model.pkl
- data/models/scorecard_object.pkl
- data/results/scorecard_table.csv
- data/results/pd_scorecard_metrics.json
- data/results/credit_policy_analysis.csv

TARGET METRICS (V5.1 CORRECTED): AUC 0.68-0.72, Gini 36-44%, KS 26-32%

QUALITY STANDARDS:
- Feature count must be 10-15 (not 60). If you end up with more than 20, STOP
  and re-examine your feature selection logic.
- ALL WOE coefficients must be negative. No exceptions.
- No macro features in the model. Zero. None.
- grade must be in the model (it should have the strongest coefficient)
- int_rate and sub_grade must NOT be in the model
- Scorecard table must be clean and printable — this is an interview deliverable
- Credit policy analysis must show meaningful discrimination at practical cutoffs
  (NOT 100% approval at the lowest score)
- RAG thresholds (V5.1): Green >= 42%, Amber 36-42%, Red < 36%
- Connect to prior role: "This mirrors the behavioral scorecard RAG framework from
  institutional monitoring reports."
- If AUC < 0.68, discuss honestly. Discuss whether additional features or
  different binning might help. Our actual test AUC of 0.6931 is within V5.1 target.
```

### What to verify after:
- ALL logistic regression WOE coefficients are negative (hard requirement)
- Feature count is 10-15 (not 60)
- grade IS in the model and has a strong negative coefficient
- int_rate, sub_grade, and all macro features are NOT in the model
- AUC is 0.68-0.72 on test set (actual: 0.6931 — within V5.1 target)
- Train-test AUC gap < 0.03
- Credit policy optimal cutoff is NOT the minimum score
- Scorecard table is clean and interpretable
- RAG status is Amber or Green per V5.1 thresholds (Green ≥ 42%, Amber 36-42%, Red < 36%)
- No quality gates were silently lowered
- **Git:** Commit and push after successful verification:
  git add notebooks/03_PD_Model_Scorecard.ipynb src/scorecard.py data/models/ data/results/
  git commit -m "Notebook 03: Behavioral PD scorecard with grade, disciplined feature selection"
  git push

---

## Session 4: Notebook 04 — ML Models (with Macro Covariates)

### Prompt:
```
Read the roadmap section on PD ML models (XGBoost/LightGBM) and CLAUDE.md.

Build Notebook 04: PD Model — XGBoost and LightGBM (with Macro Features).

CONTEXT NOTE: This is where macro features, int_rate, and sub_grade are included
alongside grade for maximum predictive power. The ML models serve as the
performance ceiling — tree-based models handle the non-linear interactions between
macro features and LC's growth trajectory that caused problems in the logistic
regression scorecard.

INPUT:
- data/processed/train.parquet, val.parquet, test.parquet (original features, not WOE)
- data/results/pd_scorecard_metrics.json (for comparison)

PROCESS:
1. Load original (non-WOE) data. Use all features EXCEPT post-origination
   variables that would cause leakage. The leakage list is in CLAUDE.md.
   INCLUDE grade, int_rate, AND sub_grade — these are ML models where
   collinearity is handled by the tree structure, not linear algebra.
   INCLUDE all 6 macro features (UNRATE, CSUSHPINSA, A191RL1Q225SBEA,
   CPIAUCSL, DFF, UMCSENT) from the parquet files.
   These are essential for generalization across economic regimes and their
   non-linear interactions are properly captured by tree-based models.

2. Handle categorical variables (grade, sub_grade, home_ownership, purpose, verification_status, addr_state)
   using appropriate encoding (label encoding for tree models).

3. XGBoost:
   - Use Optuna for hyperparameter tuning (30 trials minimum)
   - Tune: n_estimators, max_depth, learning_rate, subsample,
     colsample_bytree, min_child_weight, reg_alpha, reg_lambda
   - Use early_stopping_rounds=50 with validation set
   - Compute AUC, Gini, KS on test set
   - Report best hyperparameters and their values

4. LightGBM:
   - Same Optuna tuning approach
   - Compare training speed vs XGBoost
   - Note any differences in final metrics

5. SHAP analysis (for best model):
   - Global feature importance: SHAP summary plot (beeswarm)
   - Top 10 features bar plot (mean |SHAP|)
   - Include macro features in the top-N analysis and document their ranking
   - SHAP dependence plots for top 3 features (include macro if in top 3)
   - Single prediction explanation (pick one default and one non-default)
   - Document in markdown: "SHAP analysis reveals the model's decision rules.
     Macro features rank [X-Y] in importance, validating their inclusion for
     cycle-adjusted predictions."

6. Model comparison table:
   Metric | LogReg Scorecard | XGBoost | LightGBM
   Include: AUC, Gini, KS, Training Time, Number of Features Used
   Format for presentation (not raw output).

7. Discussion section (markdown):
   "Why would LendingClub use the scorecard for production despite lower AUC?"
   Answer: interpretability, regulatory requirements, monotonicity,
   model governance, OCC/Fed oversight — connect to IMR review process.
   Also note: "Macro features improve both scorecard and ML model performance,
   but the scorecard's interpretability is critical for credit policy implementation.
   The XGBoost model (AUC [Y]) demonstrates the predictive power available, but
   the scorecard (AUC [X]) is preferred for decision automation and regulatory defense."

OUTPUT:
- data/models/pd_xgboost_model.pkl
- data/models/pd_lgbm_model.pkl
- data/results/shap_values.pkl
- data/results/model_comparison.json (AUC, Gini, KS for all three models)
- data/results/xgboost_feature_importance.csv (SHAP-based)

TARGET METRICS (V5.1 CORRECTED): XGBoost AUC 0.71-0.73, Gini 42-46%, KS >= 30%
NOTE: Previous version stated AUC ≥ 0.80 which is unrealistic for origination-only
data with temporal split and no leakage features. Camp B methodology ceiling is ~0.73.
```

### What to verify after:
- XGBoost and LightGBM include macro features in their feature set
- SHAP analysis shows macro features alongside borrower features
- Macro features appear in top-10 feature importance rankings
- Model comparison shows improved AUC vs scorecard-only model
- Training time is reasonable (<5 minutes per model)
- Optuna tuning converges without errors
- **Git:** Commit and push after successful verification:
  git add notebooks/04_PD_Model_ML_Ensemble.ipynb data/models/ data/results/
  git commit -m "Notebook 04: ML PD models with macro features, SHAP analysis"
  git push

---

## Session 5: Notebooks 05 & 06 — EAD and LGD

### Prompt:
```
Read the roadmap sections for Days 8-9 (EAD and LGD models) and CLAUDE.md,
specifically the LGD formula update in V4.

Build Notebook 05: EAD Model AND Notebook 06: LGD Model.

Also build relevant functions in src/models.py.

For EAD (Notebook 05):
- Filter to defaulted loans only (status in ['Charged Off', 'Default'])
- Target: out_prncp (outstanding principal at default)
- Compute CCF = out_prncp / funded_amnt for each loan
- Build a Random Forest regressor to predict EAD
- Features: loan_amnt, term, grade, annual_inc, dti, fico_avg
- Compare to analytical amortization formula (simple calc of remaining balance)
- Add markdown explaining why EAD=1 assumption works for fully-drawn
  term loans (connect to my prior institution where EAD was assumed 1 for mortgages)
- Target: MAPE < 15%
- Print portfolio average CCF and compare to assumption

For LGD (Notebook 06):
- Filter to defaulted loans only
- LGD FORMULA UPDATE (V4):
  PRIMARY: LGD = 1 - ((recoveries - collection_recovery_fee) / out_prncp)
    where out_prncp = outstanding principal at default
    - recoveries: post-charge-off cash recovered (100% populated, float64)
    - collection_recovery_fee: fee paid to recovery agent (100% populated, float64)
    - Net recovery = recoveries - collection_recovery_fee

  CROSS-CHECK: LGD_simple = 1 - (total_rec_prncp / out_prncp)
    - Compare to primary formula; they should be close but not identical
    - total_rec_prncp includes pre-default principal repayments

  SANITY CHECK: Portfolio-average LGD should be ~0.83 (from LendingClub 10-K)

- Two-stage model:
  **Step 1 (Classification Phase)**: LogisticRegression — binary classification:
  any recovery? (recovery_flag = 1 if recoveries > 0 else 0)

  **Step 2 (Regression Phase)**: For loans with recovery, predict recovery_rate using
  GradientBoostingRegressor (Beta regression if possible, else standard regression
  with clipping to [0,1])
  recovery_rate = (recoveries - collection_recovery_fee) / out_prncp

  NOTE: "Step 1" and "Step 2" are model construction stages, NOT IFRS-9 stages

- Combined LGD = 1 - P(recovery) × E[recovery_rate | recovery]
- Compute portfolio-average LGD — should be approximately 0.83
  (based on 10-K: $49M recovery / $286M gross ALLL)
- LGD by grade: verify higher-risk grades have higher LGD (monotonic)
- Default recovery_rate should be seeded from the LGD model's portfolio-level output
  - LGD ≈ 83% → recovery_rate default = 0.17
- Target: MAE < 0.10
- Print portfolio average LGD and compare to 0.83 benchmark
- Print LGD by grade (A-G) and verify monotonicity

OUTPUT:
- data/models/ead_model.pkl
- data/models/lgd_stage1_model.pkl, lgd_stage2_model.pkl
- data/results/ead_metrics.json (MAPE, portfolio_avg_ccf, sample_comparisons)
- data/results/lgd_metrics.json (portfolio_avg_lgd, lgd_by_grade, mae)
```

### What to verify after:
- EAD MAPE is < 15%
- LGD portfolio average is ~0.80-0.85 (close to 0.83 benchmark)
- LGD increases from Grade A to G (monotonic)
- Both models are serializable (pkl files load without error)
- LGD formula properly uses recoveries and collection_recovery_fee
- **Git:** Commit and push after successful verification:
  git add notebooks/05_EAD_Model.ipynb notebooks/06_LGD_Model.ipynb src/models.py data/models/ data/results/
  git commit -m "Notebooks 05-06: EAD and LGD models with recovery-based formula"
  git push

---

## Session 5.5: Notebook 05.5 — Prepayment Model (NEW)

### Prompt:
```
Build Notebook 05.5: Prepayment Model for Competing Risks.

This is a NEW session between EAD/LGD (Session 5) and ECL/Vintage (Session 6).

INPUT:
- data/processed/loans_cleaned.parquet (full dataset with all loans)
- data/processed/train.parquet, val.parquet, test.parquet (splits)

PROCESS:

1. Identify prepaid loans:
   - Filter to status == 'Fully Paid'
   - Calculate actual_life = last_pymnt_d - issue_d (in months)
   - Identify "early prepayment": actual_life < 0.8 × term
     (where term is 36 or 60 months)
   - Create binary indicator: is_prepaid = 1 if early prepayment, 0 otherwise
   - Document in markdown: "Prepayment is a competing risk alongside default.
     A loan that prepays can never default. This affects ECL computation via DCF."

2. Empirical prepayment rates:
   - Compute CPR (Conditional Prepayment Rate) by term:
     * 36-month loans: X% of remaining balance prepays per month (average)
     * 60-month loans: Y% of remaining balance prepays per month (average)
   - CPR by vintage: track if older vintages prepay faster/slower
   - CPR by grade: A/B grades faster than F/G
   - CPR by macro: does prepayment accelerate in low-rate environment (low DFF)?
   - Create CPR lookup table: term × grade × vintage × macro_regime

3. Survival analysis by segment:
   - For each segment (term × grade × vintage), compute Kaplan-Meier survival curves
   - Show proportion of loans surviving (not prepaid) over months on book
   - Compare across grades and terms
   - Document empirical CPR alongside survival estimates

4. Compare to historical data:
   - Compute realized CPR for each cohort (actual prepayment / original balance)
   - Compare model output to realized
   - Flag outliers or inconsistencies

5. Save outputs for use in ECL computation:
   - data/models/prepayment_model.pkl
   - data/results/prepayment_rates.csv (lookup: term × grade × vintage → CPR)
   - data/results/prepayment_metrics.json
   - data/results/liquidation_curves.csv (empirical paydown by term and vintage)

OUTPUT:
- Notebook 05.5 with full EDA and model development
- data/models/prepayment_model.pkl
- data/results/prepayment_rates.csv (lookup: term × grade × vintage → CPR)
- data/results/prepayment_metrics.json (validation stats)
- data/results/liquidation_curves.csv (empirical paydown by term and vintage)

QUALITY STANDARDS:
- First markdown cell: "Prepayment is a competing risk. In a DCF-ECL model, each month
  a loan faces three outcomes: stay current, default, or prepay. These rates compound
  over the loan's life."
- Model should have reasonable discrimination (AUC ≥ 0.70) — if using logistic regression
- CPR by term should be plausible: 36-month loans tend to prepay faster than 60-month
- Connection to prior role: "This mirrors mortgage prepayment analysis from portfolio management
  at my prior institution"
- NOTE: The LendingClub public dataset provides loan-level terminal outcomes, not monthly
  payment history. We use survival analysis on time-to-event data rather than month-level
  hazard modeling. The empirical CPR lookup table is the standard industry approach.
```

### What to verify after:
- Prepayment identification logic is correct (actual_life < 0.8 × term)
- CPR for 36-month loans is higher than 60-month (expected)
- CPR increases with lower interest rate environment (DFF)
- Liquidation curves show smooth paydown, no anomalies
- prepayment_rates.csv is loadable and has expected dimensions
- **Git:** Commit and push after successful verification:
  git add notebooks/05_5_Prepayment_Model.ipynb data/models/ data/results/
  git commit -m "Notebook 05.5: Prepayment model with survival analysis and CPR lookup"
  git push

---

## Session 6: Notebook 07 — ECL, Vintage Analysis, Roll Rates (Major Updates)

### Prompt:
```
Read the roadmap section for Days 10-11 (ECL, Vintage Analysis, Roll Rates) and CLAUDE.md.

Build Notebook 07: ECL Computation, Vintage Analysis, and Flow Rate Analysis.

MAJOR UPDATES IN V6:
- Synthetic monthly panel reconstruction (NEW)
- Forward-only flow rates with documented limitations
- Flow Through Rate (FTR) computation and trending
- DCF-ECL with competing risks (prepayment rates from Notebook 5.5)
- Dual-mode flow rate computation: rolling average + CECL-compliant
- Three ECL views: Pre-FEG, Central, Post-FEG

Also build: src/ecl_engine.py, src/flow_rates.py

PROCESS:

**COMPONENT 0 — SYNTHETIC MONTHLY PANEL CONSTRUCTION (NEW)**:

Before computing flow rates, construct a synthetic monthly panel from loan-level data:

1. For each loan, create one row per month from `issue_d` to either:
   - `last_pymnt_d` + charge-off lag (for Charged Off loans)
   - `last_pymnt_d` (for Fully Paid loans)
   - Data snapshot date Q4 2018 (for Current/Late loans at snapshot)

2. Assign monthly DPD status:
   - Fully Paid loans: Current every month until payoff month
   - Charged Off loans: Current from origination until delinquency onset.
     Delinquency onset = last_pymnt_d + 1 month. Then 30 DPD = onset,
     60 DPD = onset + 1 month, ..., Charge-off at ~120 DPD (onset + 4 months)
   - Current at snapshot: Current every month until snapshot date
   - Late at snapshot: Current until estimated delinquency onset, then progressive DPD buckets

3. Assign monthly balance:
   - Performing months: scheduled balance from amortization formula
   - Delinquent months: freeze balance at last performing month (approximation)

4. Aggregate into monthly receivables by DPD bucket

Output: data/processed/synthetic_monthly_panel.parquet
Estimated size: ~2.2M loans × ~24 avg months ≈ 50-60M rows. Use chunked processing.

**Document assumptions explicitly in the notebook:**
"This synthetic panel reconstructs approximate monthly DPD status from loan-level
terminal outcomes. Performing loans are assumed current until payoff or default
onset. Defaulted loans are back-calculated from last payment date assuming
standard 30-day DPD progression to charge-off at 120+ DPD."

**Document limitations explicitly:**
"Curing is unobservable. Intermediate delinquencies for eventually-performing
loans are invisible. Flow rates represent forward (worsening) transitions only.
Balances for delinquent months are approximate (scheduled, not actual with
penalty interest)."

**COMPONENT 1: Simple ECL = PD × EAD × LGD** ✅
   - Compute by grade (A-G), by vintage year, by purpose
   - Portfolio ALLL ratio = total ECL / total outstanding
   - Compare to LendingClub 10-K ALLL ratio of 5.7%
   - Print results table: Grade | Vintage | ECL | ALLL Ratio

**COMPONENT 2: DCF-based ECL (with competing risks)**:
   - For each grade pool, project monthly cash flows over remaining life
   - THREE competing outcomes per month:
     * P(stay current) × monthly_payment
     * P(default) × (1 - LGD) × remaining_balance
     * P(prepay) × remaining_balance
   - Use marginal PD for default timing (from Notebook 3 or 4)
   - Use prepayment rates from Notebook 5.5
   - Discount at effective interest rate
   - ECL = Contractual CF (NPV) - Expected CF (NPV)

**COMPONENT 3: Vintage analysis** ✅
   - Cumulative default rate curves by origination year vs MOB
   - Marginal PD curves with 6-month rolling average smoothing
   - Vintage comparison table: default rate at MOB 6/12/18/24/36
   - Identify outlier vintages (2011, 2012 expected to have high defaults)

**COMPONENT 4: FLOW RATE ANALYSIS (CRITICAL UPDATE)**:
   - Build Receivables Tracker in institutional format: monthly dollar balances
     by DPD bucket (Current, 30+, 60+, 90+, 120+, 150+, 180+) with
     account counts, GCO (Gross Charge-Off), Recovery, NCO (Net Charge-Off)
   - Compute flow rates as simple ratios BELOW the dollar receivables:
     * 30+ Flow Rate = 30 DPD(this month) / Current(last month)
     * 60+ Flow Rate = 60 DPD(this month) / 30 DPD(last month)
     * ... continuing through each bucket to GCO
   - **RENAME**: "Forward Default Flow Rate Analysis"
     - Make clear these are one-directional: Current → 30+ → 60+ → ... → GCO
     - Remove any reference to curing rates or two-way transition matrices
   - **FLOW THROUGH RATE (NEW)**:
     * FTR = (Current→30+) × (30+→60+) × ... × (150+→180+) × (180+→GCO)
     * This tracks the cumulative probability a loan moves from current to GCO
     * Compute time series of FTR by grade and vintage
     * Use as cross-check against PD model outputs:
       "If model PD is 20% and FTR is 25%, investigate why"
     * Save FTR time series to data/results/flow_through_rate.csv
   - Segment by grade and by vintage
   - Track flow rate trends — identify acceleration patterns
   - **Note on data source**: "These flow rates are derived from synthetic
     monthly panel reconstruction from loan-level terminal outcomes. In a
     production environment with monthly payment tapes, curing rates and
     two-way transition matrices would be observable."

**COMPONENT 5: DUAL-MODE FLOW RATE COMPUTATION**:
   - Save two versions of flow rates:
     * **Extended Rolling Average (Operational)**:
       - Lookback: 6 months of historical data
       - Smooth: 6-month rolling average
       - Project: extend flat for forecast period
       - File: data/results/flow_rates_extend.csv
     * **CECL-Compliant Rates (Regulatory)**:
       - Phase 1 (Reasonable & Supportable): 24 months, macro-adjusted rates
       - Phase 2 (Reversion): 12-month transition from Phase 1 to Phase 3
       - Phase 3 (Historical): revert to long-term historical average
       - Reversion method: straight-line interpolation
       - File: data/results/flow_rates_cecl.csv
   - Both files should be saved; both will be used in Streamlit

**COMPONENT 6: THREE ECL VIEWS**:
   - **Pre-FEG**: pure model output
     * Flow rates: 6-month rolling average (no macro overlay)
     * No scenario weighting
     * File: data/results/ecl_prefeg.csv
   - **Central**: baseline macro overlay applied to flow rates
     * Macro-adjusted flow rates per economic scenario (baseline)
     * Unemployment-sensitive flow rates (higher UNRATE → higher flow rates)
     * File: data/results/ecl_central.csv
   - **Post-FEG**: weighted across scenarios + qualitative adjustments
     * Compute in Notebook 09 (macro scenarios)
     * Will be 3-scenario average with weights
     * File: data/results/ecl_postfeg.csv

**COMPONENT 7: ALLL tracker**:
   - Monthly ECL reserve level
   - Reserve build/release (provision expense)
   - NCO coverage ratio (NCO / ALLL)

**ADD AN HONEST LIMITATIONS SECTION AT THE END:**
"This analysis uses synthetically reconstructed monthly DPD status from loan-level
terminal outcomes. In a production environment with monthly payment tapes, curing
rates and two-way transitions would be observable, enabling more precise flow rate
estimation and CECL compliance. The framework and methodology are identical to
production implementation — the input granularity differs."

OUTPUT:
- src/ecl_engine.py, src/flow_rates.py
- data/processed/synthetic_monthly_panel.parquet (synthetic monthly data — ~50-60M rows)
- data/results/ecl_by_grade.csv
- data/results/ecl_by_vintage.csv
- data/results/vintage_curves.csv
- data/results/receivables_tracker.csv
- data/results/flow_rates.csv (primary output)
- data/results/flow_rates_extend.csv (NEW: operational mode)
- data/results/flow_rates_cecl.csv (NEW: CECL-compliant mode)
- data/results/flow_through_rate.csv (NEW: FTR time series)
- data/results/ecl_dcf_results.json
- data/results/ecl_prefeg.csv, ecl_central.csv (NEW: Pre-FEG and Central views)
```

### What to verify after:
- Synthetic monthly panel is created with expected dimensions (~50-60M rows)
- Receivables tracker sums correctly (balances reconcile month-to-month)
- Flow rates are between 0 and 1 (valid probabilities)
- Flow Through Rate is less than individual flow rates (cumulative effect)
- ECL by grade is monotonic (higher-risk grades have higher ECL)
- CECL flow rates revert to historical average after R&S period
- All three ECL views (Pre-FEG, Central, Post-FEG) are saved
- Limitations section clearly documents synthetic reconstruction
- **Git:** Commit and push after successful verification:
  git add notebooks/07_ECL_Vintage_FlowRates.ipynb src/ecl_engine.py src/flow_rates.py data/processed/synthetic_monthly_panel.parquet data/results/
  git commit -m "Notebook 07: ECL, vintage analysis, synthetic flow rates with documented limitations"
  git push

---

## Session 7: Notebook 08 — Model Validation

### Prompt:
```
Read the roadmap section for Days 12-14 (Model Validation) and CLAUDE.md.

Build Notebook 08: Model Validation and Monitoring.

Also build: src/validation.py

This notebook directly mirrors institutional quarterly monitoring report
with Gini, PSI, CSI, VDI and RAG framework.

SESSION 6 CONTEXT:
- EAD/LGD models are confirmed working (not falling back to flat constants)
- DCF-ECL at 6.09% ALLL on full 1.35M sample (organic, no parameter tuning)
- LGD mean: 0.884, EAD mean: ~$8,703
- PD Scorecard test AUC: 0.6931, Gini = 38.62%
- lgd_stage1_model.pkl is dict-wrapped {'model': LogisticRegression, 'scaler': StandardScaler}
  — must unwrap before validation

Build src/validation.py with these functions:
- compute_gini(y_true, y_pred) → Gini coefficient
- compute_ks(y_true, y_pred) → KS statistic + plot data
- compute_psi(expected, actual, n_bins=10) → PSI value + bin-level detail
- compute_csi(train_feature, test_feature, n_bins=10) → CSI value
- compute_vdi(train_feature, test_feature) → VDI value
- rag_status(metric, metric_type) → 'GREEN'/'AMBER'/'RED' based on thresholds
- generate_monitoring_report(results_dict) → formatted summary

RAG THRESHOLDS (V5.1 CORRECTED — use these, not the old 55%/45% thresholds):
- PD Scorecard: Green (Gini ≥ 42%), Amber (36-42%), Red (< 36%)
- ML Models: Green (Gini ≥ 46%), Amber (42-46%), Red (< 42%)
- PSI: Green (< 0.10), Amber (0.10-0.25), Red (≥ 0.25)
- EAD: Green (MAPE < 15%), Amber (15-25%), Red (≥ 25%)
- LGD: Green (MAE < 0.10), Amber (0.10-0.15), Red (≥ 0.15)

The notebook should produce:

1. PD Model Discrimination: AUC (with 95% CI via bootstrap), Gini, KS plot, CAP curve
   - Validate BOTH scorecard (Notebook 03) AND ML models (Notebook 04)
   - Print side-by-side comparison table

2. Calibration: Hosmer-Lemeshow, calibration plot by decile, Brier score

3. EAD/LGD Model Validation (NEW — Session 6 confirmed models fire):
   - EAD: MAPE on held-out defaulted loans, portfolio avg CCF, residual analysis
   - LGD Stage 1: AUC for binary recovery prediction
     IMPORTANT: lgd_stage1_model.pkl is a dict. Unwrap: model = pkl['model'], scaler = pkl['scaler']
     Apply scaler.transform() before predict_proba()
   - LGD Stage 2: MAE, RMSE for recovery rate
   - Combined LGD: portfolio avg vs 0.83 benchmark, LGD by grade monotonicity
   - Feature engineering required: fico_avg = (fico_range_low + fico_range_high) / 2,
     grade_enc = grade mapped to ordinal {A:0, B:1, ..., G:6}

4. Stability: PSI for each test period (2016, 2017, 2018 separately),
   CSI for each feature, VDI for each feature

5. RAG status table — the showpiece:
   Model | Metric | Value | Threshold | RAG Status
   Should look exactly like a bank's quarterly monitoring report
   Color coding: Green (✓), Amber (△), Red (✗)
   Include rows for: PD Scorecard, PD XGBoost, PD LightGBM, EAD, LGD Stage 1, LGD Combined

6. Out-of-time performance: Gini/AUC on 2016, 2017, 2018 separately

7. Backtesting:
   - Predicted cumulative default rate vs actual by vintage
   - ECL backtesting: Session 6 DCF-ECL 6.09% vs 10-K 5.7% (0.39pp gap expected —
     no management overlays in our model)
   - Prepayment backtesting: predicted vs actual by term
   (Note: PD model metrics use real data; flow-rate ECL uses synthetic panel)

8. EXTERNAL VALIDATION (NEW in V4):
   - Load benchmark_population_2014.csv from data/raw/
   - Contains: FICO score, delinquency bucket, PERFORMANCE_OUTCOME (GOOD/BAD)
   - PSI computation: Compare your PD model's score distribution vs benchmark FICO
   - External calibration: Score benchmark population with your model,
     compare predicted PD to actual PERFORMANCE_OUTCOME
   - Display as: PSI table with RAG status + calibration chart
   - Interview framing: "I validated my model against LendingClub's internal
     benchmark population, mirroring institutional approaches to external validation"

OUTPUT:
- src/validation.py
- data/results/validation_report.json
- data/results/rag_status_table.csv
- data/results/psi_by_period.csv
- data/results/csi_by_feature.csv
- data/results/ead_validation.json (NEW: EAD validation metrics)
- data/results/lgd_validation.json (NEW: LGD validation metrics)
- data/results/external_validation_psi.csv (benchmark population PSI)
```

### What to verify after:
- PD Scorecard Gini = 38.62% on test (within V5.1 Amber range 36-42%)
- ML model Gini > 42% on test
- KS statistic is > 20% for all periods
- PSI is < 0.10 (low population shift)
- EAD MAPE < 15%
- LGD portfolio average between 0.80-0.90 (actual: 0.884)
- Out-of-time Gini is stable across 2016, 2017, 2018
- RAG status table includes ALL models (PD, EAD, LGD) and is formatted correctly
- External benchmark validation shows reasonable PSI and calibration
- **Git:** Commit and push after successful verification:
  git add notebooks/08_Model_Validation.ipynb src/validation.py data/results/
  git commit -m "Notebook 08: Model validation with RAG framework, EAD/LGD validation, external benchmark"
  git push

---

## Session 8: Notebook 09 — Macro Scenarios & Stress Testing

### Prompt:
```
Read the roadmap section for Notebook 09 (Macro Scenarios & Strategy) and CLAUDE.md.

Build Notebook 09: Macro Scenario and Strategy Analysis.

Also build: src/macro_scenarios.py

MAJOR UPDATE IN V3/V4: Stress applied at flow rate level, not final ECL.

SESSION 6 CONTEXT:
- Pre-FEG DCF-ECL = 6.09% ALLL on full 1.35M loans (organic, no tuning). This is
  the baseline anchor for all stress scenarios.
- ecl_central.csv currently exists but is a PLACEHOLDER (identical to ecl_prefeg.csv).
  This notebook MUST regenerate ecl_central.csv with actual macro regression overlay.
- ecl_postfeg.csv does NOT exist yet. This notebook creates it as the weighted
  3-scenario average.
- credit_policy_analysis.csv was not produced in Session 3. Include it in the
  Strategy Analysis section below.
- All three ECL views (Pre-FEG, Central, Post-FEG) should bracket 5-8% ALLL
  to remain plausible vs LC's 10-K 5.7%.

PROCESS:

1. Pull FRED data: unemployment (UNRATE), GDP growth, HPI (USSTHPI)
   Use fredapi library with API key from environment variable
   Document in markdown: Why macro scenarios matter for regulatory capital

2. Map historical macro conditions to vintage default rates and flow rates
   - Cluster years by macro regime (crisis 2008-09, recovery 2010-12, expansion 2013-15, etc.)
   - Compute flow rates for each regime
   - Document relationship: UNRATE ↔ flow_rates (elasticity)
   - Show scatter plots: UNRATE vs 30+ flow rate

3. Define scenarios with specific macro parameters:
   - Baseline (60% weight): current trajectory (UNRATE at 4%, GDP growth 2%)
   - Mild Downturn (25%): unemployment +1.5pp, GDP -0.5%
   - Stress (15%): unemployment +3pp, GDP -3%
   - Document weighting rationale

4. **FLOW RATE STRESS (CRITICAL UPDATE V3/V4)**:
   - Don't stress the final ECL — stress the flow rates
   - Apply multiplicative stress to each flow rate:
     * Baseline: flow_rate(t)
     * Mild: flow_rate(t) × (1 + stress_multiplier_mild)
       where stress_multiplier depends on macro factor sensitivity
     * Stress: flow_rate(t) × (1 + stress_multiplier_stress)
   - Example: 15% stress per flow rate → cumulative flow-through increases by ~75%
   - Show NON-LINEAR effect: stress compounds through the waterfall
   - Compute ECL for each scenario using stressed flow rates
   - Create visualization: side-by-side flow rate stress comparison
     * Pre-FEG flow rates vs Central (macro-stressed) vs Post-FEG (scenario-weighted)
   - FEG toggle applies to BOTH Operational and CECL modes
   - In Operational mode, stressed rates extend flat; in CECL mode, stressed rates apply during Phase 1 only then revert
   - **Note**: "Flow rate stress is applied to synthetically derived rates.
     The compounding demonstration is still valuable and correct mathematically."

5. **REGENERATE ECL VIEWS**:
   - **Central (REGENERATE):** Apply macro regression to flow rates under Baseline scenario.
     Overwrite data/results/ecl_central.csv (currently placeholder = Pre-FEG)
   - **Post-FEG (CREATE NEW):** Weighted average across all 3 scenarios:
     ECL_weighted = 0.60 × ECL_baseline + 0.25 × ECL_mild + 0.15 × ECL_stress
     Save to data/results/ecl_postfeg.csv
   - Show impact on ALLL ratio under each scenario and each view
   - Verify: Pre-FEG (6.09%) ≤ Central ≤ Post-FEG (all should be in 5-8% range)

6. Sensitivity analysis:
   - Unemployment ±1% impact on flow rates (not final ECL)
   - Recovery rate sensitivity (10% to 25%)
   - Scorecard cutoff sensitivity (approval rate vs loss)
   - Create tornado chart showing ranking of sensitivities

7. CREDIT STRATEGY ANALYSIS (NOTE: credit_policy_analysis.csv was created in Session 7; extend/refine here with macro-adjusted scenarios):
   - **Credit policy optimization:** For each scorecard cutoff, compute approval rate,
     default rate, expected loss, risk-adjusted return. Save to credit_policy_analysis.csv
   - Grade-level profitability: interest income minus expected loss per grade
   - Credit expansion: "What if we loosen Grade G cutoff by 10 points?"
   - Vintage root cause: "Why is 2017 underperforming 2016?"
   - Pricing analysis: is each grade priced to cover its expected loss?
   - Document margins by grade: spread vs ECL

OUTPUT:
- src/macro_scenarios.py
- data/results/ecl_by_scenario.csv (baseline, mild, stress)
- data/results/ecl_central.csv (REGENERATED with actual macro overlay)
- data/results/ecl_postfeg.csv (NEW: weighted scenario average)
- data/results/flow_rates_by_scenario.csv (stressed flow rates)
- data/results/flow_rate_stress_comparison.csv (baseline vs mild vs stress)
- data/results/credit_policy_analysis.csv (created in Session 7; refined here with macro scenarios)
- data/results/strategy_analysis.csv
- data/results/sensitivity_results.json
- data/results/macro_scenarios.json
```

### What to verify after:
- Stress multipliers are positive (flow rates increase under stress)
- Cumulative FTR stress is larger than individual flow rate stress (compounding effect)
- ECL increases under stress scenario (ECL_stress > ECL_baseline)
- ecl_central.csv is DIFFERENT from ecl_prefeg.csv (macro overlay applied)
- ecl_postfeg.csv exists and shows weighted average
- All three ECL views bracket 5-8% ALLL (plausible vs 10-K 5.7%)
- credit_policy_analysis.csv exists with meaningful discrimination at practical cutoffs
- Sensitivity analysis shows plausible relationships (higher UNRATE → higher default flow)
- Grade profitability analysis shows reasonable spreads
- **Git:** Commit and push after successful verification:
  git add notebooks/09_Macro_Scenarios_Stress_Testing.ipynb src/macro_scenarios.py data/results/
  git commit -m "Notebook 09: Macro scenarios, stress testing, strategy analysis, ECL views regenerated"
  git push

---

## Session 9: Streamlit App — Engine Modules (V3/V4 Redesign)

### Prompt:
```
Read the Streamlit architecture document at
docs/Streamlit_App_Technical_Architecture.md.

Build the engine modules with the V4 REDESIGN:
- app/engine/flow_rate_engine.py
- app/engine/ecl_projector.py (REDESIGNED)
- app/engine/prepayment.py (NEW)
- app/engine/liquidation.py
- app/engine/vintage_analyzer.py
- app/engine/macro_overlay.py

**CRITICAL V4 REDESIGN — ECLProjector:**

The ECLProjector class is redesigned as follows:

OLD (V2):
  projector = ECLProjector(flow_rates, pd_model, lgd_model, ...)

NEW (V3/V4):
  projector = ECLProjector(pd_model, lgd_model)
  # Flow rates NOT passed in __init__

NEW METHOD:
  flow_rates = projector.compute_forecast_flow_rates(
    lookback_months=6,
    method='extend' | 'cecl',  # operational vs regulatory
    rs_period_months=24,
    reversion_method='straight_line',
    loaded_receivables_tracker=...,
    macro_scenario='baseline'
  )

- The ECLProjector loads receivables snapshot from a tracker file
  (passed to compute_forecast_flow_rates)
- Flow rates are computed ON-DEMAND from the tracker, not pre-computed
- Supports DUAL-MODE forecasting:
  * 'extend' mode: rolling average extension (operational)
  * 'cecl' mode: R&S period + reversion to historical (regulatory)
- Macro overlay applied during flow rate computation if macro_scenario != 'baseline'

Additional methods:
- project(n_months, scenario='baseline') → DataFrame with monthly balances, ECL
- export_to_excel(filename) → institutional format receivables tracker format
- compute_flow_through_rate() → FTR time series for validation

**NEW MODULE: app/engine/prepayment.py**
- compute_conditional_prepayment_rate(term, grade, vintage, macro_state)
- get_prepayment_assumptions() → CPR lookup table
- apply_prepayment_to_projection() → updates remaining balance by prepayment

Use the flow rates computed in Notebook 07 (data/results/flow_rates.csv,
data/results/flow_rates_extend.csv, data/results/flow_rates_cecl.csv)
as inputs to the flow rate computation.

**ADD data limitation notes in module docstrings:**
"Flow rates derived from synthetic monthly panel reconstruction. Production
implementation would use observed monthly payment data."

Include comprehensive unit tests in tests/ for the ECLProjector:
- Test that balances sum correctly
- Test that projection with zero new originations converges to zero
- Test dual-mode produces different results ('extend' vs 'cecl')
- Test CECL mode reverts to historical averages after R&S period
- Test flow through rate computation
- Test that stress scenario produces higher ECL than baseline
- Test that export produces valid Excel file

Don't build any Streamlit UI yet — just the engine logic.
```

### What to verify after:
- ECLProjector takes only pd_model and lgd_model in __init__
- compute_forecast_flow_rates() method exists and returns DataFrame
- Dual-mode forecasting works ('extend' vs 'cecl' produce different results)
- CECL mode reverts to historical average after R&S period
- Flow Through Rate is computed and saved
- Unit tests pass (at least 7 test cases)
- Data limitation notes are in docstrings
- **Git:** Commit and push after successful verification:
  git add app/engine/ tests/
  git commit -m "Streamlit engine modules with V4 ECLProjector redesign"
  git push

---

## Session 10: Streamlit App — UI Pages (V4 Updates)

### Prompt:
```
Read the Streamlit architecture document, specifically the page specifications and V4 updates.

Build the Streamlit app UI with V4 UPDATES:
- app/streamlit_app.py (main entry point)
- app/pages/01_portfolio_overview.py
- app/pages/02_roll_rate_analysis.py (RENAMED: was "02_rollrate")
- app/pages/03_vintage_performance.py
- app/pages/04_ecl_forecasting.py (MAJOR UPDATES)
- app/pages/05_scenario_analysis.py (UPDATES)
- app/pages/06_model_monitoring.py

Use the engine modules from the previous session.
Load data from data/processed/ and data/results/.

Use Plotly for all interactive charts.
Use st.cache_data for data loading.
Use st.metric() for KPI cards.

**MAJOR UPDATES TO PAGES:**

Page 04 — ECL Forecasting (MAJOR REDESIGN):
- Add mode selector at top:
  Radio buttons: "Operational Forecast (PyCraft)" | "CECL Reserve Estimation"
  (This determines whether to use 'extend' or 'cecl' method in compute_forecast_flow_rates)
- Add FEG toggle:
  Radio buttons: "Pre-FEG | Central | Post-FEG"
  (Loads different ECL files from Session 6: ecl_prefeg.csv, ecl_central.csv, ecl_postfeg.csv)
- Input controls: liquidation factor slider, new originations input,
  recovery rate slider, scenario selector
- "Run Forecast" button triggers ECLProjector.project()
- Display results: projection charts + downloadable Excel
- (NEW) Flow Through Rate KPI card on ECL page
- (NEW) Data limitation disclaimer: "Flow rates derived from synthetic monthly panel
  reconstruction. Production implementation would use observed monthly payment data."

Page 01 — Portfolio Overview (ADDITIONS):
- Add Flow Through Rate KPI card on portfolio dashboard
  (Load from data/results/flow_through_rate.csv, show latest month)

Page 05 — Scenario Analysis (MAJOR ADDITIONS):
- Add stress visualization: side-by-side flow rate comparison
  Load data/results/flow_rates_by_scenario.csv
  Plot: baseline flow rates vs mild downturn vs stress scenario
  Show cumulative effect on ECL and FTR

All pages:
- Keep existing functionality
- Use new dual-mode engine
- Add data limitation disclaimer in sidebar

For the ECL Forecasting page (04):
- Dual liquidation factor UI:
  * Operational mode: simple slider (0.7-1.0, default 0.95)
  * CECL mode: term-level inputs (36-month CPR slider, 60-month CPR slider)
- (NEW) Upload/Export Assumptions buttons on ECL page
  * Upload: CSV with custom assumptions (CPR, recovery_rate, etc.)
  * Export: download current assumptions to CSV

For Model Monitoring page (06):
- Display the RAG status table from data/results/rag_status_table.csv
- Color-code: green/amber/red
- Include Gini trend chart and PSI analysis
- (NEW) External benchmark validation results

Prioritize getting Pages 01-04 fully working first.
Pages 05-06 can be simpler.
```

### What to verify after:
- Mode selector works and switches between operational/CECL
- FEG toggle loads correct ECL file (Pre-FEG vs Central vs Post-FEG)
- Flow Through Rate displays on both portfolio overview and ECL pages
- Stress visualization shows side-by-side flow rate comparison
- Upload/Export Assumptions buttons are functional
- ECL forecast runs without errors and produces downloadable Excel
- All charts are interactive (Plotly)
- Data limitation disclaimer appears in sidebar and relevant pages
- **Git:** Commit and push after successful verification:
  git add app/pages/ app/streamlit_app.py
  git commit -m "Streamlit UI pages with mode selector, FEG toggle, and data limitations"
  git push

---

## Session 11: Streamlit App — AI Chatbot (If Time Permits)

### Prompt:
```
Read the AI Analyst section of the Streamlit architecture document.

Build the AI chatbot page:
- app/pages/07_ai_analyst.py
- app/components/chatbot.py

The chatbot should:
1. Use the Anthropic API (claude-opus-4-6 model)
2. Load portfolio context from session state (receivables, ECL, model metrics,
   flow rates, vintage curves, prepayment rates)
3. Accept file uploads (CSV, Excel) and parse them for analysis
4. Maintain conversation history within the session
5. Include suggested questions in the sidebar

API key should come from .streamlit/secrets.toml:
ANTHROPIC_API_KEY = "sk-ant-..."

The system prompt should establish the AI as a credit risk analyst with
deep knowledge of CECL, DCF, roll-rate analysis, vintage analysis,
macro scenario frameworks, prepayment modeling, and synthetic monthly panel
reconstruction. It should reference the specific portfolio data loaded in context.

Test with these sample questions:
- "Which grade has the highest delinquency migration rate?"
- "Compare vintage 2016 vs 2017 at MOB 24"
- "Write a quarterly loss forecast memo"
```

### What to verify after:
- Chatbot page loads without errors
- Conversation history is maintained
- Responses reference specific portfolio data
- File uploads parse correctly
- API calls to Anthropic complete successfully
- **Git:** Commit and push after successful verification:
  git add app/pages/07_ai_analyst.py app/components/
  git commit -m "Streamlit AI analyst page with portfolio context"
  git push

---

## Session 12: Polish and README

### Prompt:
```
Review all notebooks in notebooks/ and all source files in src/.

Do the following:

1. Ensure every notebook starts with a markdown cell that includes:
   - Notebook title and purpose
   - Prior role experience connection (specific project reference)
   - What inputs it reads and what outputs it produces

2. Ensure all src/ modules have:
   - Module-level docstring
   - Function-level docstrings with Args, Returns, Examples
   - Type hints on all function signatures

3. Create a comprehensive README.md that includes:
   - Project title and one-paragraph summary
   - Framing: "Portfolio management and loss forecasting tool"
     (not just "credit risk modeling project")
   - Key results table (AUC, Gini, KS, ECL vs 10-K benchmark)
   - Screenshots or descriptions of the Streamlit dashboard
   - How to run: setup instructions, data download, notebook execution order,
     Streamlit launch command
   - Methodology overview: PD → EAD → LGD → ECL pipeline, with DCF approach,
     prepayment modeling, and flow rate analysis
   - Project structure diagram
   - Technologies used
   - V6 enhancements section:
     * Behavioral scorecard with grade (portfolio monitoring, not origination)
     * Discipline feature selection (IV ≥ 0.05, correlations < 0.70, 10-15 features)
     * Macro features reserved for ML and stress testing
     * Synthetic monthly panel reconstruction for flow rates
     * Forward-only flow rate analysis with documented limitations
     * Honest data limitations section explaining curing invisibility

4. Add a "Data Limitations" section to README.md explaining:
   - Monthly DPD status is synthetically reconstructed from loan-level outcomes
   - Curing events are unobservable
   - Forward-only flow rates represent worsening transitions
   - In production with monthly payment tapes, two-way transitions and curing rates
     would be observable
   - Framework and methodology are identical to production; input granularity differs

5. Verify requirements.txt has all packages with pinned versions

6. Run the full notebook sequence (01-09) to confirm no import errors
   or missing file dependencies

7. Make a clean git commit with a meaningful message
```

### What to verify after:
- All notebooks have proper opening markdown cells
- All src/ functions have docstrings and type hints
- README.md is comprehensive and current
- requirements.txt has all packages
- No import errors when running notebooks sequentially
- Git commit is clean and well-messaged
- **Git:** Final commit and push:
  git add notebooks/ src/ app/ README.md requirements.txt
  git commit -m "Session 12: Polish, documentation, and data limitations"
  git push

---

## Troubleshooting Common Claude Code Issues

### Issue: Claude Code tries to use too much memory loading the full dataset
**Fix:** Add to your prompt: "Use chunked reading with pd.read_csv(chunksize=100000)
if memory is an issue, or use dtype specifications to reduce memory footprint.
Consider using float32 instead of float64 for numeric columns."

### Issue: Claude Code generates a notebook that's all code, no markdown
**Fix:** Add: "Every code cell MUST be preceded by a markdown cell explaining what
it does, why we're doing it, and how it connects to industry practice."

### Issue: Claude Code tries to install packages that aren't in requirements.txt
**Fix:** Point it to requirements.txt: "Only use packages listed in requirements.txt.
Do not install any additional packages without asking me first."

### Issue: Claude Code writes monolithic functions instead of using src/ modules
**Fix:** Add: "All reusable logic should go in the appropriate src/ module.
Notebooks should import from src/ and call functions, not define them inline.
Only notebook-specific plotting and analysis code should be in the notebook itself."

### Issue: Context window fills up during a long session
**Fix:** Start a new session. Tell Claude Code: "I'm continuing from the previous
session. Read the project state from these files: [list the key outputs from the
previous session]. Don't re-read the full roadmap — just the relevant section for
today's task."

### Issue: Claude Code overwrites or ignores CLAUDE.md decisions
**Fix:** Be explicit in the prompt: "Follow the technical decisions in CLAUDE.md —
specifically: L2 regularization (not L1), time-based split (not random),
target variable definition (drop right-censored statuses), macro feature integration,
flow rate stress testing, prepayment modeling, ECLProjector redesign. Do NOT deviate
from these decisions."

### Issue: Data cleaning fails due to data quirks not documented
**Fix:** In Session 1, include data quirks section from CLAUDE.md: "CRITICAL: the dataset
has these known quirks: [list from Known Data Quirks section]. Handle each explicitly
in your cleaning code with comments explaining why."

### Issue: FRED API failures or missing data
**Fix:** In Session 1, include fallback: "If FRED API fails, download CSVs manually
from FRED website (fredapi documentation has links) and merge by month. Store in
a fallback directory and document which method was used in the notebook."

### Issue: Flow rate computation produces invalid values (>1 or <0)
**Fix:** Add validation: "Check that all flow rates are in [0, 1] range. If any exceed
this range, investigate the source data (negative balances, data errors). Flag and
document any anomalies. Print a sanity check table: flow_rate_name | min | max | mean."

### Issue: LGD formula inconsistency
**Fix:** Add in Session 5 prompt: "Use ONLY the V4 formula: LGD = 1 - ((recoveries -
collection_recovery_fee) / out_prncp). Do NOT use recoveries / loan_amnt. Cross-check
with LGD_simple = 1 - (total_rec_prncp / out_prncp) and document why they differ."

### Issue: Synthetic monthly panel is too large to process
**Fix:** Use chunked processing: "Process the synthetic panel construction in chunks
of 100K loans at a time. Write each chunk to parquet and append. This keeps memory
usage manageable for the ~2.2M loan dataset."

### Issue: Flow rates exceed 1.0 in some buckets
**Fix:** Add diagnostic logic: "Print a table of flow rates by grade and month.
If any exceed 1.0, it indicates a data error (more loans moving to a bucket than
were in the source bucket). Investigate and cap at 1.0 with a note in the notebook."

### Issue: LGD Stage 1 model fails with "'dict' object has no attribute 'feature_names_in_'"
**Fix:** lgd_stage1_model.pkl is saved as a dict `{'model': LogisticRegression, 'scaler': StandardScaler}`,
not a bare model. Unwrap: `model = pkl['model']`, `scaler = pkl['scaler']`. Apply
`scaler.transform(X)` before `model.predict_proba()`. lgd_stage2_model.pkl is a bare
GradientBoostingRegressor (no wrapping needed).

### Issue: EAD/LGD models fall back to flat constants (LGD=0.83, EAD=funded_amnt)
**Fix:** Models expect `fico_avg` and `grade_enc` columns that don't exist in loans_cleaned.parquet.
Add feature engineering BEFORE prediction:
- `fico_avg = (fico_range_low + fico_range_high) / 2`
- `grade_enc = grade.map({g: i for i, g in enumerate(['A','B','C','D','E','F','G'])})`
Also: bare `except:` blocks silently swallow the KeyError. Always use `except Exception as e`
and log the error.

### Issue: DCF-ECL monthly total doesn't match batch total (e.g., 3.6× overcounting)
**Fix:** In monthly DCF loss extraction, use `if losses[t] != 0` instead of `if losses[t] > 0`.
In early loan periods, prepayment returns full remaining balance > scheduled payment,
producing legitimate negative monthly losses. Discarding negatives inflates the monthly total.

---

## V6 Summary of Changes from V5

### Sessions 0-1: No Changes
Sessions 0 and 1 remain exactly as V5. Data cleanup and macro feature integration are unchanged.

### Session 2 (WOE/IV): Grade Included
Grade is now included as a WOE binning candidate (expected IV > 0.5). int_rate and
sub_grade are excluded (collinear with grade). iv_summary.csv will now include grade.

### Session 3 (PD Scorecard): COMPLETE REWRITE
The behavioral scorecard now:
- **INCLUDES grade** as the strongest predictor (IV > 0.5)
- **EXCLUDES int_rate and sub_grade** (mechanical collinearity with grade)
- **EXCLUDES macro features** (confound in linear models; reserved for ML and stress testing)
- **Feature selection discipline**: IV ≥ 0.05, |correlation| < 0.70, target 10-15 features
- **Coefficient sign enforcement**: ALL WOE coefficients must be negative
- **Target metrics (V5.1 CORRECTED)**: AUC 0.68-0.72, Gini 36-44%, KS 26-32%
  (Previous versions incorrectly stated AUC ≥ 0.75 which assumed int_rate in model)
- **RAG thresholds (V5.1 CORRECTED)**: Green (Gini ≥ 42%), Amber (36-42%), Red (< 36%)
  (Previous versions incorrectly stated Green ≥ 55%, Amber 45-55%, Red < 45%)
- **Actual results**: Test AUC = 0.6931, Gini = 38.62% → Amber (within V5.1 target)

Rationale: This is a behavioral scorecard for portfolio monitoring (loans already on books),
not an origination scorecard. Grade is a known attribute and the single strongest predictor.

### Session 4 (ML Models): Context Note Added
ML models now include a note that macro features, int_rate, and sub_grade belong here
(not in the scorecard). Tree-based models handle non-linear interactions correctly.

### Session 5 (EAD/LGD): No Changes
EAD and LGD remain as V5.

### Session 5.5 (Prepayment): Revised Approach
Replaced month-level logistic regression (requires monthly observations) with:
- Kaplan-Meier survival curves by segment
- Cox proportional hazard model
- Empirical CPR lookup table (term × grade × vintage × macro regime)

This is the standard industry approach for prepayment modeling and works with loan-level
terminal outcomes.

### Session 6 (ECL/Vintage/Flow Rates): MAJOR REVISION
Added synthetic monthly panel reconstruction at the beginning:
1. Back-calculate monthly DPD status from loan-level terminal outcomes
2. Construct monthly receivables tracker by DPD bucket
3. Compute forward-only flow rates (Current → 30+ → 60+ → ... → GCO)
4. Document assumptions and limitations explicitly

**Forward-Only Flow Rates** (not two-way transitions):
- Current → 30+ → 60+ → 90+ → 120+ → 150+ → 180+ → Charge-off
- Curing is unobservable; we only track worsening transitions
- Balances for delinquent months are approximate (scheduled, not actual)
- Framework and methodology are production-standard; input granularity is approximate

**Documentation**: All notebooks include explicit limitations sections explaining the
synthetic reconstruction and how production implementation would differ.

### Session 7 (Model Validation): EXPANDED SCOPE
- Backtesting reframed as "predicted cumulative default rate vs actual cumulative default
  rate by vintage" (fully real data) rather than "predicted ECL vs losses" (which depends
  on synthetic flow rates).
- **NEW:** EAD/LGD model validation added (Session 6 confirmed models fire properly)
- **CORRECTED:** RAG thresholds updated to V5.1 (Green ≥ 42%, not ≥ 55%)
- **NEW:** ECL backtesting section referencing Session 6 DCF-ECL of 6.09%
- lgd_stage1_model.pkl dict-wrapping documented for proper validation

### Session 8 (Macro Scenarios): CRITICAL SESSION 6 DEPENDENCIES
- **MUST regenerate** ecl_central.csv (currently placeholder = ecl_prefeg.csv)
- **MUST create** ecl_postfeg.csv (weighted 3-scenario average)
- credit_policy_analysis.csv already exists from Session 7 (refine with macro-adjusted scenarios)
- Pre-FEG DCF-ECL of 6.09% is the baseline anchor for stress scenarios
- Flow rate stress section notes that stress is applied to synthetically derived rates.
  The compounding math is correct; the base rates are approximate.
- Strategy analysis section (grade profitability, credit expansion, vintage root cause)
  is entirely based on real data.

### Sessions 9-10 (Streamlit): Data Limitation Disclaimers
- Engine modules include data limitation notes in docstrings
- UI pages include disclaimer: "Flow rates derived from synthetic monthly panel reconstruction.
  Production implementation would use observed monthly payment data."
- Data limitations appear in sidebar info box on flow rate and ECL pages

### Session 11 (AI Analyst): Updated System Prompt
Chatbot knowledge base includes synthetic monthly panel reconstruction, forward-only flow
rates, and data limitations. Interview framing guidance provided.

### Session 12 (Polish): Data Limitations Section
README.md includes "Data Limitations" section explaining:
- Synthetic monthly panel reconstruction from loan-level outcomes
- Curing invisibility
- Forward-only flow rates
- Production equivalence with input granularity caveat

---

## Conclusion

This V6 Prompting Guide incorporates all V5.1 amendments and data gap assessments into
a complete, cohesive framework. Key principles:

1. **Every prompt specifies exact file paths, not relative paths**
2. **Every prompt includes Known Data Quirks upfront so Claude Code doesn't discover issues**
3. **Every prompt includes SANITY CHECKS with specific numerical ranges**
4. **Every prompt documents prior role framing for interview readiness**
5. **Every prompt enforces quality gates with specific metrics**
6. **Data limitations are documented honestly, not hidden**

**V6 distinguishes the behavioral scorecard** (portfolio monitoring with grade included,
macro excluded) from **ML performance ceiling** (all features including macro).

**V6 revises flow rate analysis** around synthetic monthly panel reconstruction with
forward-only transitions and explicit documentation of limitations.

**V6 maintains analytical rigor** — the methodology and framework are production-standard;
the data granularity is approximate due to the public dataset's loan-level structure.

The CLAUDE.md file is your contract with Claude Code. All session prompts reference it,
ensuring consistency across sessions.

Good luck building!
