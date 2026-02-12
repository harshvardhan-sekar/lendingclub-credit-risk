# Claude Code — Session-by-Session Prompting Guide
## LendingClub Credit Risk Analytics Project
## Version 5 (V5) — February 2026

### Changes from V4:
This version incorporates major framework updates:
- **HSBC references replaced with generic framing** for broader applicability
- **Prior role/institutional framing** instead of specific employer references
- **New Originations contextual note** for ECL Projector
- **FEG/stress applies to both modes** documentation
- **Recovery rate connection** to LGD model outputs
- **Full-file profiling corrections**: Column drop strategy restructured into 4 tiers. 6 columns previously dropped (38% missing) now kept for WOE/IV. Outlier flags for 14 columns. Comprehensive 5-part profiling checklist (outliers, distributions, correlations, validation, categoricals) embedded in Session 1.
- All other V4 content preserved: known data quirks, column categories, exact file paths, FRED integration, quality gates

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
- These macro features are CRITICAL for time-based split validation.
  Without them, model overfits to economic regime of train period.
- Macro features included in all parquet files: train.parquet, val.parquet, test.parquet

### PD Models
- Logistic regression scorecard (Notebook 03):
  - Features: WOE-transformed borrower characteristics (NO grade/int_rate)
  - Include macro features (unemployment, GDP, HPI at minimum)
  - L2 regularization (Ridge), hyperparameter tuned via 5-fold stratified CV
  - Output: probability of default + scorecard points
  - Target: AUC ≥ 0.75, Gini ≥ 55%
- ML models (Notebook 04):
  - XGBoost and LightGBM with Optuna tuning
  - Include macro features alongside borrower features
  - SHAP analysis shows macro + borrower feature importance
  - Target: AUC ≥ 0.80

### Basel Framework Integration
- PD models MUST include macro features to cover full economic cycle
- This is Basel requirement: models must be validated across regimes
- Macro features (FRED merge) provide cycle information
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

### Prepayment Model (NEW)
- Competing risk alongside default
- Identify prepaid loans: "Fully Paid" status where actual_life << contractual_term
- Model: conditional prepayment rate by month, loan characteristics, vintage, macro
- Output: empirical prepayment rates by term (36 vs 60), vintage, grade
- Used in DCF-ECL (Notebook 07) and Streamlit forecasting
- Prepayment rates feed liquidation curves for operational and CECL forecasts

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
- Segment by grade and vintage
- Track trends and identify acceleration patterns
- **Flow Through Rate (NEW)**: Diagonal multiplication of all intermediate rates
  - FTR = (Current→30+) × (30+→60+) × ... × (150+→180+) × (180+→GCO)
  - Tracks cumulative delinquency progression
  - Cross-check against PD model

### ECL Views and Adjustment Methods
- **Pre-FEG**: pure model output, no adjustments
  - Rolling 6-month average flow rates
  - No macro overlay
- **Central**: baseline macro overlay applied to flow rates
  - Macro-adjusted flow rates per scenario
- **Post-FEG**: weighted across scenarios + qualitative adjustments
  - Compute in Notebook 09 (stress scenarios)
  - Multi-scenario average with expert judgment layer

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

### Streamlit UI Enhancements
- ECL page includes mode selector: "Operational Forecast" vs "CECL Reserve Estimation"
- FEG toggle: Pre-FEG | Central | Post-FEG radio buttons
- Flow Through Rate KPI on portfolio dashboard + ECL page
- Dual liquidation factor UI: simple slider (operational) vs term-level (CECL)
- Upload/Export Assumptions buttons on ECL page
- Scenario page: side-by-side flow rate stress comparison visualization

### Model Validation (Notebook 08)
- Discrimination: AUC, Gini, KS, CAP curve, bootstrap CI
- Calibration: Hosmer-Lemeshow, decile calibration, Brier score
- Stability: PSI, CSI, VDI for each test period
- RAG status: Green (Gini ≥60%), Amber (50-60%), Red (<50%)
- Out-of-time performance by vintage year (2016, 2017, 2018)
- Backtesting: predicted ECL vs realized losses
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
- PD scorecard: behavioral scorecard monitoring + RAG framework
- WOE/IV binning: mirrors VantageScore/FICO binning, utilization, DTI analysis
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
6. For features like int_rate and grade that will have very high IV
   (because they're assigned based on credit risk), discuss whether to include.
   Decision: EXCLUDE int_rate and grade/sub_grade from the scorecard since
   they're outcomes of the credit decision process, not independent predictors.
   Keep them for ML models (Notebook 04).
7. Validate monotonicity: plot bad rate by bin for each selected feature.
   Flag any rank-ordering breaks (acceptable if minor and documented).
8. Transform all selected features to WOE values
9. Save outputs

EXPECTED RESULTS:
- grade and sub_grade will have very high IV (>0.5) — flag as potential leakage
- int_rate will have very high IV — note it's assigned based on grade (circular)
- fico_range_low/high should have IV ~0.3-0.4 (strong predictors)
- annual_inc and dti should have IV ~0.1-0.2 (medium predictors)
- Macro features (UNRATE, CSUSHPINSA) should have notable IV due to correlation with defaults
- Final selected features should be 25-35 (from initial ~70)

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
- int_rate and grade are excluded from scorecard features
- WOE is fit on train only, applied to val and test
- Monotonicity holds for most features (some minor violations are acceptable)
- No features with IV > 0.5 are suspiciously leaky (like total_pymnt or last_pymnt)
- Macro features appear in the final selection with reasonable IV values

---

## Session 3: Notebook 03 — PD Scorecard (with Macro Covariates)

### Prompt:
```
Read the roadmap section "Days 5-7: PD Models — Scorecard" and CLAUDE.md.

Build Notebook 03: PD Model Scorecard (Logistic Regression with Macro Features).

Also build: src/scorecard.py

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

Part B — Build Notebook 03:
1. Load WOE-transformed data
2. Select features with IV > 0.1 (from iv_summary.csv), excluding grade/int_rate
3. CRITICAL MACRO ADDITION:
   - Include macro features from train/val/test parquets: UNRATE, CSUSHPINSA,
     A191RL1Q225SBEA (GDP), CPIAUCSL
   - These features should be INCLUDED in the PD model alongside WOE-transformed
     borrower features
   - Rationale (document in markdown): "Basel requires full economic cycle coverage.
     Macro features carry cycle information. Without them, the model overfits to the
     training period's economic regime and fails to generalize to out-of-time test sets.
     This is demonstrated by FRED data showing distinct regimes: 2007-08 crisis, 2009-12
     recovery with elevated unemployment, and 2013-15 expansion."
   - Scale macro features (zero-mean, unit variance) before inclusion
4. Fit LogisticRegression with:
   - penalty='l2' (Ridge)
   - Tune C using 5-fold stratified CV on training data (try C = [0.001, 0.01, 0.1, 1, 10])
   - solver='lbfgs', max_iter=1000
   - Report best_C value
5. Verify all coefficients are negative (higher WoE = lower risk = lower log-odds of default)
   - If any coefficient is positive, investigate and document why
6. Generate the scorecard table and display it formatted (publication-ready)
7. Score all datasets (train, val, test)
8. Compute metrics on each dataset:
   - AUC, Gini (= 2×AUC - 1), KS statistic
   - Print a comparison table: Metric | Train | Validation | Test
   - The gap between train and test AUC should be small (<0.03) — if larger, discuss overfitting
   - EXPECTED: AUC >= 0.75, Gini >= 55%
9. Plot:
   - ROC curve (train + test overlaid, with AUC values in legend)
   - KS plot (cumulative good vs cumulative bad distribution)
   - Score distribution: good vs bad (overlaid histograms with proper density)
   - Calibration plot: predicted PD vs actual default rate by score decile
10. Credit policy analysis (THE STRATEGY LAYER — this is critical):
    - For score cutoffs from min to max (in steps of 10):
      compute approval_rate, expected_default_rate, expected_loss_rate
    - Plot: Approval Rate vs Expected Default Rate (tradeoff curve)
    - Find the cutoff that maximizes: approval_rate × (1 - default_rate)
      as a simple optimization
    - Create a "Credit Policy Table":
      Score Cutoff | Approval Rate | Default Rate | Expected Loss | Recommendation
    - Document in markdown: "This mirrors the Credit Strategy team's daily decision process
      at LendingClub. The tradeoff curve shows the business decision: tighter policy =
      lower default but lower volume; looser policy = higher volume but higher losses."
11. Grade mapping:
    - Map score ranges to LendingClub grades A-G
    - Compare model-assigned grades vs actual grades
    - Flag any grade reversals (e.g., model assigns A but LC assigned G)
12. RAG status summary:
    - Gini >= 60% → Green, 50-60% → Amber, <50% → Red
    - Report RAG status for train, val, test
    - Document thresholds clearly
13. Save outputs

OUTPUT:
- src/scorecard.py (reusable module)
- data/models/pd_logreg_model.pkl
- data/models/scorecard_object.pkl
- data/results/scorecard_table.csv
- data/results/pd_scorecard_metrics.json (AUC, Gini, KS, best_C, macro_features_used)
- data/results/credit_policy_analysis.csv

TARGET METRICS: AUC >= 0.75, Gini >= 55%, KS >= 30%

QUALITY STANDARDS:
- Scorecard table must be clean and printable — this is an interview deliverable
- Credit policy analysis section should have its own markdown header explaining
  that this is what LendingClub's Credit Strategy team does daily
- Connect to prior role: "This mirrors the behavioral scorecard RAG framework from
  institutional monitoring reports. Gini thresholds (Green/Amber/Red) are derived from regulatory
  guidance on model stability and discrimination."
- Document explicitly: macro features are included alongside borrower features
- If Gini < 55%, discuss potential reasons (feature coverage, economic regime effects)
  and whether additional features would help
- Print macro feature coefficients separately — show their relative importance
```

### What to verify after:
- All logistic regression coefficients are negative
- AUC is reasonable (≥0.75) and train-test gap is small (<0.03)
- Scorecard table is clean and interpretable (publication-ready)
- Credit policy analysis shows a clear tradeoff curve
- RAG status is green or amber (not red)
- Macro features are in the model and have reasonable coefficients
- Grade mapping shows general alignment with actual grades (some drift acceptable)

---

## Session 4: Notebook 04 — ML Models (with Macro Covariates)

### Prompt:
```
Read the roadmap section on PD ML models (XGBoost/LightGBM) and CLAUDE.md.

Build Notebook 04: PD Model — XGBoost and LightGBM (with Macro Features).

INPUT:
- data/processed/train.parquet, val.parquet, test.parquet (original features, not WOE)
- data/results/pd_scorecard_metrics.json (for comparison)

PROCESS:
1. Load original (non-WOE) data. Use all features EXCEPT post-origination
   variables that would cause leakage. The leakage list is in CLAUDE.md.
   Include grade and int_rate here since these are ML models, not scorecards.
   CRITICAL: INCLUDE macro features (UNRATE, CSUSHPINSA, A191RL1Q225SBEA, CPIAUCSL, DFF, UMCSENT)
   from the parquet files. These are essential for generalization across economic regimes.

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

TARGET METRICS: XGBoost AUC >= 0.80, KS >= 35%
```

### What to verify after:
- XGBoost and LightGBM include macro features in their feature set
- SHAP analysis shows macro features alongside borrower features
- Macro features appear in top-10 feature importance rankings
- Model comparison shows improved AUC vs scorecard-only model
- Training time is reasonable (<5 minutes per model)
- Optuna tuning converges without errors

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

2. Build prepayment model:
   - Filter to loan-month level: for each loan, track prepaid vs active by month
   - Features: term, grade, annual_inc, dti, loan_amnt, fico_avg, vintage, MOB (months on book)
   - Include macro features from parquets: UNRATE, CSUSHPINSA (HPI), A191RL1Q225SBEA
   - Target: prepaid_flag (1 if loan prepaid in this month, 0 otherwise)
   - Build LogisticRegression with L2 regularization
   - Segment analysis: by term (36 vs 60), by grade, by vintage

3. Empirical prepayment rates:
   - Compute CPR (Conditional Prepayment Rate) by term:
     * 36-month loans: X% of remaining balance prepays per month (average)
     * 60-month loans: Y% of remaining balance prepays per month (average)
   - CPR by vintage: track if older vintages prepay faster/slower
   - CPR by grade: A/B grades faster than F/G
   - CPR by macro: does prepayment accelerate in low-rate environment (low DFF)?
   - Create CPR lookup table: term × grade × vintage × macro_regime

4. Compare to historical data:
   - Compute realized CPR for each cohort (actual prepayment / original balance)
   - Compare model output to realized
   - Flag outliers or inconsistencies

5. Save outputs for use in ECL computation:
   - data/models/prepayment_model.pkl
   - data/results/prepayment_rates.csv (term, grade, vintage, cpr)
   - data/results/prepayment_metrics.json

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
- Model should have reasonable discrimination (AUC ≥ 0.70)
- CPR by term should be plausible: 36-month loans tend to prepay faster than 60-month
- Connection to prior role: "This mirrors mortgage prepayment analysis from portfolio management
  at my prior institution"
```

### What to verify after:
- Prepayment model AUC ≥ 0.70
- CPR for 36-month loans is higher than 60-month (expected)
- CPR increases with lower interest rate environment (DFF)
- Liquidation curves show smooth paydown, no anomalies
- prepayment_rates.csv is loadable and has expected dimensions

---

## Session 6: Notebook 07 — ECL, Vintage Analysis, Roll Rates (Major Updates)

### Prompt:
```
Read the roadmap section for Days 10-11 (ECL, Vintage Analysis, Roll Rates) and CLAUDE.md.

Build Notebook 07: ECL Computation, Vintage Analysis, and Flow Rate Analysis.

MAJOR UPDATES IN V3/V4:
- Flow Through Rate (FTR) computation and trending
- DCF-ECL with competing risks (prepayment rates from Notebook 5.5)
- Dual-mode flow rate computation: rolling average + CECL-compliant
- Three ECL views: Pre-FEG, Central, Post-FEG

Also build: src/ecl_engine.py, src/flow_rates.py

PROCESS:

1. Simple ECL = PD × EAD × LGD
   - Compute by grade (A-G), by vintage year, by purpose
   - Portfolio ALLL ratio = total ECL / total outstanding
   - Compare to LendingClub 10-K ALLL ratio of 5.7%
   - Print results table: Grade | Vintage | ECL | ALLL Ratio

2. DCF-based ECL (mirror LendingClub 10-K methodology with competing risks):
   - For each grade pool, project monthly cash flows over remaining life
   - THREE competing outcomes per month:
     * P(stay current) × monthly_payment
     * P(default) × (1 - LGD) × remaining_balance
     * P(prepay) × remaining_balance
   - Use marginal PD for default timing (from Notebook 3 or 4)
   - Use prepayment rates from Notebook 5.5
   - Discount at effective interest rate
   - ECL = Contractual CF (NPV) - Expected CF (NPV)

3. Vintage analysis (mirrors institutional practice):
   - Cumulative default rate curves by origination year vs MOB
   - Marginal PD curves with 6-month rolling average smoothing
   - Vintage comparison table: default rate at MOB 6/12/18/24/36
   - Identify outlier vintages (2011, 2012 expected to have high defaults)

4. **FLOW RATE ANALYSIS (CRITICAL UPDATE)**:
   - Build Receivables Tracker in institutional format: monthly dollar balances
     by DPD bucket (Current, 30+, 60+, 90+, 120+, 150+, 180+) with
     account counts, GCO (Gross Charge-Off), Recovery, NCO (Net Charge-Off)
   - Compute flow rates as simple ratios BELOW the dollar receivables:
     * 30+ Flow Rate = 30 DPD(this month) / Current(last month)
     * 60+ Flow Rate = 60 DPD(this month) / 30 DPD(last month)
     * ... continuing through each bucket to GCO
   - **FLOW THROUGH RATE (NEW)**:
     * FTR = (Current→30+) × (30+→60+) × ... × (150+→180+) × (180+→GCO)
     * This tracks the cumulative probability a loan moves from current to GCO
     * Compute time series of FTR by grade and vintage
     * Use as cross-check against PD model outputs:
       "If model PD is 20% and FTR is 25%, investigate why"
     * Save FTR time series to data/results/flow_through_rate.csv
   - Segment by grade and by vintage
   - Track flow rate trends — identify acceleration patterns
   - These flow rates feed the Streamlit forecasting tool

5. **DUAL-MODE FLOW RATE COMPUTATION (NEW)**:
   - Save two versions of flow rates:
     * **Extended Rolling Average (Operational)**:
       - Lookback: 6 months of historical data
       - Smooth: 6-month rolling average
       - Project: extend flat for forecast period
       - File: data/results/flow_rates_extend.csv
     * **CECL-Compliant Rates (Regulatory)**:
       - Phase 1 (Reasonable & Supportable period): 24 months, macro-adjusted rates
       - Phase 2 (Reversion): 12-month transition from Phase 1 to Phase 3
       - Phase 3 (Historical): revert to long-term historical average (e.g., 10-year avg)
       - Reversion method: straight-line interpolation
       - File: data/results/flow_rates_cecl.csv
   - Both files should be saved; both will be used in Streamlit

6. **THREE ECL VIEWS (NEW)**:
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
   - Document clearly which mode is which in the notebook

7. ALLL tracker:
   - Monthly ECL reserve level
   - Reserve build/release (provision expense)
   - NCO coverage ratio (NCO / ALLL)

OUTPUT:
- src/ecl_engine.py, src/flow_rates.py
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
- Receivables tracker sums correctly (balances reconcile month-to-month)
- Flow rates are between 0 and 1 (valid probabilities)
- Flow Through Rate is less than individual flow rates (cumulative effect)
- ECL by grade is monotonic (higher-risk grades have higher ECL)
- CECL flow rates revert to historical average after R&S period
- All three ECL views (Pre-FEG, Central, Post-FEG) are saved

---

## Session 7: Notebook 08 — Model Validation

### Prompt:
```
Read the roadmap section for Days 12-14 (Model Validation) and CLAUDE.md.

Build Notebook 08: Model Validation and Monitoring.

Also build: src/validation.py

This notebook directly mirrors institutional quarterly monitoring report
with Gini, PSI, CSI, VDI and RAG framework.

Build src/validation.py with these functions:
- compute_gini(y_true, y_pred) → Gini coefficient
- compute_ks(y_true, y_pred) → KS statistic + plot data
- compute_psi(expected, actual, n_bins=10) → PSI value + bin-level detail
- compute_csi(train_feature, test_feature, n_bins=10) → CSI value
- compute_vdi(train_feature, test_feature) → VDI value
- rag_status(metric, metric_type) → 'GREEN'/'AMBER'/'RED' based on thresholds
- generate_monitoring_report(results_dict) → formatted summary

The notebook should produce:
1. Discrimination: AUC (with 95% CI via bootstrap), Gini, KS plot, CAP curve
2. Calibration: Hosmer-Lemeshow, calibration plot by decile, Brier score
3. Stability: PSI for each test period (2016, 2017, 2018 separately),
   CSI for each feature, VDI for each feature
4. RAG status table — the showpiece:
   Metric | Value | Threshold | RAG Status
   Should look exactly like a bank's quarterly monitoring report
   Color coding: Green (✓), Amber (△), Red (✗)
5. Out-of-time performance: Gini/AUC on 2016, 2017, 2018 separately
6. Backtesting: predicted ECL vs actual realized losses by vintage
7. EXTERNAL VALIDATION (NEW in V4):
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
- data/results/external_validation_psi.csv (NEW: benchmark population PSI)
```

### What to verify after:
- Gini on train and test are both > 50% and close to each other
- KS statistic is > 20% for all periods
- PSI is < 0.1 (low population shift)
- Out-of-time Gini is stable across 2016, 2017, 2018
- RAG status table is saved and formatted correctly
- External benchmark validation shows reasonable PSI and calibration

---

## Session 8: Notebook 09 — Macro Scenarios & Stress Testing

### Prompt:
```
Read the roadmap section for Notebook 09 (Macro Scenarios & Strategy) and CLAUDE.md.

Build Notebook 09: Macro Scenario and Strategy Analysis.

Also build: src/macro_scenarios.py

MAJOR UPDATE IN V3/V4: Stress applied at flow rate level, not final ECL.

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

5. Weighted average ECL computation across scenarios
   - ECL_weighted = 0.60 × ECL_baseline + 0.25 × ECL_mild + 0.15 × ECL_stress
   - Show impact on ALLL ratio under each scenario

6. Sensitivity analysis:
   - Unemployment ±1% impact on flow rates (not final ECL)
   - Recovery rate sensitivity (10% to 25%)
   - Scorecard cutoff sensitivity (approval rate vs loss)
   - Create tornado chart showing ranking of sensitivities

7. CREDIT STRATEGY ANALYSIS:
   - Grade-level profitability: interest income minus expected loss per grade
   - Credit expansion: "What if we loosen Grade G cutoff by 10 points?"
   - Vintage root cause: "Why is 2017 underperforming 2016?"
   - Pricing analysis: is each grade priced to cover its expected loss?
   - Document margins by grade: spread vs ECL

OUTPUT:
- src/macro_scenarios.py
- data/results/ecl_by_scenario.csv (baseline, mild, stress)
- data/results/flow_rates_by_scenario.csv (NEW: stressed flow rates)
- data/results/flow_rate_stress_comparison.csv (NEW: baseline vs mild vs stress)
- data/results/strategy_analysis.csv
- data/results/sensitivity_results.json
```

### What to verify after:
- Stress multipliers are positive (flow rates increase under stress)
- Cumulative FTR stress is larger than individual flow rate stress (compounding effect)
- ECL increases under stress scenario (ECL_stress > ECL_baseline)
- Sensitivity analysis shows plausible relationships (higher UNRATE → higher default flow)
- Grade profitability analysis shows reasonable spreads

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
macro scenario frameworks, and prepayment modeling. It should reference
the specific portfolio data loaded in context.

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
   - V4 enhancements section (data quirks, column categories, external validation)

4. Verify requirements.txt has all packages with pinned versions
5. Run the full notebook sequence (01-09) to confirm no import errors
   or missing file dependencies
6. Make a clean git commit with a meaningful message
```

### What to verify after:
- All notebooks have proper opening markdown cells
- All src/ functions have docstrings and type hints
- README.md is comprehensive and current
- requirements.txt has all packages
- No import errors when running notebooks sequentially
- Git commit is clean and well-messaged

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

---

## Conclusion

This V5 Prompting Guide is your complete reference for building a production-ready
credit risk platform. Key principles:

1. **Every prompt specifies exact file paths, not relative paths**
2. **Every prompt lists Known Data Quirks upfront so Claude Code doesn't discover issues**
3. **Every prompt includes SANITY CHECKS with specific numerical ranges**
4. **Every prompt documents prior role framing for interview readiness — without employer specifics**
5. **Every prompt enforces quality gates with specific metrics (AUC ≥ 0.75, Gini ≥ 55%, etc.)**

The CLAUDE.md file is your contract with Claude Code. Update it only when you make
major technical decisions. All session prompts reference CLAUDE.md, ensuring consistency.

V5 changes: All HSBC references have been replaced with generic framing (prior role,
institutional, PyCraft, Sherwood) to enable broader applicability. New contextual
notes added for New Originations in ECL Projector, FEG/stress dual-mode behavior,
and recovery rate seeding from LGD model.

Good luck building!
