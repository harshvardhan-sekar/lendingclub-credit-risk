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
  - grade IS included as a WOE candidate feature (assigned at origination, legitimate for portfolio monitoring)
  - int_rate and sub_grade are EXCLUDED (int_rate is mechanically determined by grade; sub_grade is a finer version creating collinearity)
- Macro features from FRED API merged by origination month (Notebook 01):
  - UNRATE (unemployment rate)
  - CSUSHPINSA (Case-Shiller HPI, home prices)
  - A191RL1Q225SBEA (GDP growth rate)
  - CPIAUCSL (Consumer Price Index)
  - DFF (Federal Funds Rate)
  - UMCSENT (University of Michigan Consumer Sentiment)
- Macro features are merged in Session 1 and present in all parquets.
  They are used in ML models (Session 4), stress testing (Session 8),
  and ECL computation (Session 6) — but NOT in the logistic regression
  scorecard (Session 3) where linear confounding with LC's growth
  trajectory inverts their signs.
- Macro features included in all parquet files: train.parquet, val.parquet, test.parquet

### PD Models
- Logistic regression scorecard (Notebook 03 — PD Scorecard with Grade):
  - Features: WOE-transformed borrower characteristics INCLUDING grade
  - EXCLUDE int_rate and sub_grade (redundant with grade, near-perfect collinearity)
  - EXCLUDE all 6 macro features (confound with LC's growth trajectory in linear model)
  - Feature selection: IV ≥ 0.05, pairwise |correlation| < 0.70, target 10-15 final features
  - All WOE coefficients MUST be negative; remove and refit if any are positive
  - Binary flags: include only those with IV ≥ 0.02
  - L2 regularization (Ridge), hyperparameter tuned via 5-fold stratified CV
  - Output: probability of default + scorecard points
  - Target: AUC ≥ 0.75, Gini ≥ 50%, KS ≥ 35%
  - RAG: Green (Gini ≥ 55%), Amber (45-55%), Red (< 45%)
- ML models (Notebook 04 — Performance Ceiling):
  - XGBoost and LightGBM with Optuna tuning
  - Include ALL features: grade + int_rate + sub_grade + macro features + borrower features
  - This is where macro features, int_rate, and sub_grade belong — tree models handle non-linear interactions
  - SHAP analysis shows macro + borrower feature importance
  - Target: AUC ≥ 0.80, Gini ≥ 60%

### Basel Framework Integration
- ML PD models (Notebook 04) include macro features to cover full economic cycle
- This is Basel requirement: models must be validated across regimes
- Macro features (FRED merge) provide cycle information for ML models and stress testing
- Logistic regression scorecard (Notebook 03) does NOT include macro features
  (they confound with LC's origination volume growth in a linear model)
- Flow rates computed monthly from synthetic panel to capture delinquency migration patterns

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

### Prepayment Model (V6 REVISED)
- Competing risk alongside default
- Identify prepaid loans: "Fully Paid" status where actual_life < 0.8 × contractual_term
- Model: survival analysis / empirical CPR lookup table (NOT month-level logistic regression)
  - No monthly payment data available; use loan-level time-to-event
  - Kaplan-Meier survival curves by segment (term × grade × vintage)
  - Empirical CPR lookup table — standard industry approach
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

### Data Limitations (V6 NEW)
- The LendingClub public dataset (accepted_2007_to_2018Q4.csv) provides one row per loan
  with origination features and terminal outcomes. NO monthly payment history available.
  The PMTHIST file is no longer publicly available.
- Sessions 0-5 (PD, ML, EAD, LGD): Use real observed data. Fully reliable.
- Session 5.5 (Prepayment): Uses loan-level survival analysis. Real data.
- Sessions 6-10 (Flow Rates, ECL, Stress Testing, Streamlit): Use synthetically
  reconstructed monthly DPD status from terminal outcomes.
- Synthetic panel construction: Back-calculate monthly DPD from issue_d, last_pymnt_d,
  loan_status, and amortization schedule. ~50-60M rows.
- Curing is UNOBSERVABLE. Flow rates are FORWARD-ONLY (Current → 30+ → ... → GCO).
- Dollar balances for delinquent months are approximate (scheduled, not actual).
- The framework is identical to production implementation; only input granularity differs.

### Forward Default Flow Rates and Receivables Tracking (V6 RENAMED)
- Use institutional format: Receivables Tracker with dollar balances by DPD bucket
- Flow rates = simple bucket ratios (NOT account-level transition matrices)
- Flow rates are FORWARD-ONLY: Current → 30+ → 60+ → ... → GCO
  - No curing rates or two-way transition matrices (curing is unobservable)
- Example: 30+ Flow Rate = 30 DPD(t) / Current(t-1)
- Segment by grade and vintage
- Track trends and identify acceleration patterns
- **Flow Through Rate**: Diagonal multiplication of all intermediate rates
  - FTR = (Current→30+) × (30+→60+) × ... × (150+→180+) × (180+→GCO)
  - Tracks cumulative delinquency progression
  - Cross-check against PD model
- NOTE: "In production with monthly payment tapes, curing rates and two-way
  transitions would be observable, enabling more precise flow rate estimation."

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
- RAG status: Green (Gini ≥ 55%), Amber (45-55%), Red (< 45%)
- Out-of-time performance by vintage year (2016, 2017, 2018)
- Backtesting: predicted cumulative default rate vs actual cumulative default rate by vintage
  (NOTE: PD model metrics use real data; flow-rate ECL metrics use synthetic panel)
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
- Use consistent random_state=42 everywhere

### Prior Role Connection
- PD scorecard with grade: portfolio monitoring scorecard + RAG framework
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
