#!/usr/bin/env python3
"""Generate Notebook 01: EDA and Data Cleaning with FRED Macro Integration."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}

cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip()))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip()))


# ==============================================================================
# TITLE
# ==============================================================================
md("""
# Notebook 01: EDA and Data Cleaning
## LendingClub Credit Risk Analytics — With FRED Macroeconomic Integration

This mirrors the data preparation phase of behavioral scorecard monitoring and
loss forecasting in my prior role. Macro features enable validation across
economic regimes. Our time-based split tests the model's ability to generalize
across the 2007-2015 crisis/recovery (train), 2016 expansion (val), and
2017-2018 (test).

### Process Overview
1. Load raw data and remove footer rows
2. Drop Tier 1 columns (>93% missing or non-feature)
3. Parse data types (term, emp_length, dates)
4. Filter to terminal loan statuses and create target variable
5. Merge FRED macroeconomic data by origination month
6. Handle missing values with tiered strategy (flags + imputation)
7. Apply priority outlier treatments
8. Create time-based train/validation/test split
9. EDA visualizations
10. Comprehensive data profiling report
11. Save all outputs
""")

# ==============================================================================
# IMPORTS
# ==============================================================================
md("## Setup: Imports and Configuration")

code("""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 100,
})

sys.path.insert(0, str(Path('..').resolve()))
from config import *

print(f"Project root: {PROJECT_ROOT}")
print(f"Data raw path: {DATA_RAW_PATH}")

# Ensure output directories exist
DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
DATA_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
""")

# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================
md("""
## Step 1: Load Raw Data

The raw CSV contains 2,260,701 rows (including 33 footer/summary rows).
We drop footer rows where `loan_amnt` is null or non-numeric, leaving exactly
2,260,668 usable loan records.
""")

code("""
raw_file = DATA_RAW_PATH / 'accepted_2007_to_2018Q4.csv'
print(f"Loading {raw_file}...")
df = pd.read_csv(raw_file, low_memory=False)
print(f"Raw data shape: {df.shape}")

# Drop footer/summary rows
rows_before = len(df)
df = df[pd.to_numeric(df['loan_amnt'], errors='coerce').notna()].copy()
rows_after = len(df)
print(f"Rows before: {rows_before:,}")
print(f"Rows after footer removal: {rows_after:,}")
print(f"Footer rows removed: {rows_before - rows_after}")
assert rows_after == 2_260_668, f"Expected 2,260,668 rows, got {rows_after:,}"
print(f"\\nColumn count: {len(df.columns)}")
print(f"\\nData types:\\n{df.dtypes.value_counts()}")
""")

# ==============================================================================
# STEP 2: DROP TIER 1 COLUMNS
# ==============================================================================
md("""
## Step 2: Drop Tier 1 Columns (>93% Missing or Non-Feature)

- **100% empty**: member_id
- **Non-feature**: id, url, pymnt_plan (constant), policy_code (constant)
- **94-95% empty**: sec_app_* columns, revol_bal_joint, desc, dti_joint, etc.
- **Hardship/settlement** (>97% null): 20 columns

**Important**: We DO NOT drop columns with 38-68% missing (open_act_il, il_util,
mths_since_last_delinq, etc.) — these are kept for WOE/IV analysis.
""")

code("""
tier1_drop = [
    'member_id',
    'id', 'url', 'pymnt_plan', 'policy_code',
    'sec_app_fico_range_low', 'sec_app_fico_range_high',
    'sec_app_earliest_cr_line', 'sec_app_inq_last_6mths',
    'sec_app_mort_acc', 'sec_app_open_acc', 'sec_app_revol_util',
    'sec_app_open_act_il', 'sec_app_num_rev_accts',
    'sec_app_chargeoff_within_12_mths',
    'sec_app_collections_12_mths_ex_med',
    'sec_app_mths_since_last_major_derog',
    'revol_bal_joint', 'desc',
    'dti_joint', 'annual_inc_joint', 'verification_status_joint',
    'hardship_type', 'hardship_reason', 'hardship_status',
    'deferral_term', 'hardship_amount', 'hardship_start_date',
    'hardship_end_date', 'payment_plan_start_date', 'hardship_length',
    'hardship_dpd', 'hardship_loan_status',
    'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
    'orig_projected_additional_accrued_interest',
    'debt_settlement_flag_date', 'settlement_status',
    'settlement_date', 'settlement_amount', 'settlement_percentage',
    'settlement_term',
]

cols_before = len(df.columns)
existing_drops = [c for c in tier1_drop if c in df.columns]
df = df.drop(columns=existing_drops)
cols_after = len(df.columns)

print(f"Columns before: {cols_before}")
print(f"Columns after:  {cols_after}")
print(f"Dropped:        {cols_before - cols_after}")

# Verify DO NOT DROP columns are still present
do_not_drop = [
    'open_act_il', 'open_il_12m', 'open_il_24m', 'total_bal_il',
    'open_acc_6m', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
    'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
    'il_util', 'mths_since_rcnt_il',
    'mths_since_recent_revol_delinq', 'mths_since_last_delinq',
]
kept = [c for c in do_not_drop if c in df.columns]
print(f"\\nVerified {len(kept)}/{len(do_not_drop)} 'DO NOT DROP' columns retained")
""")

# ==============================================================================
# STEP 3: PARSE DATA TYPES
# ==============================================================================
md("""
## Step 3: Parse Data Types

Handle known data quirks:
- **term**: Leading spaces + text → extract integer (36 or 60)
- **emp_length**: Text values → numeric (0-10)
- **Dates**: Text format 'MMM-YYYY' → datetime
- **int_rate** and **revol_util**: Already float64
""")

code("""
# term
df['term'] = df['term'].str.strip().str.extract(r'(\\d+)').astype(int)
print(f"term unique: {sorted(df['term'].unique())}")
assert set(df['term'].unique()) == {36, 60}

# emp_length
emp_map = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
    '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
    '8 years': 8, '9 years': 9, '10+ years': 10,
}
df['emp_length'] = df['emp_length'].map(emp_map)
print(f"emp_length unique: {sorted(df['emp_length'].dropna().unique())}")
print(f"emp_length missing: {df['emp_length'].isna().mean():.2%}")

# Dates
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')
df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%Y')
print(f"\\nissue_d range: {df['issue_d'].min()} to {df['issue_d'].max()}")
print(f"last_pymnt_d missing: {df['last_pymnt_d'].isna().mean():.2%}")

# Verify int_rate and revol_util
print(f"\\nint_rate dtype: {df['int_rate'].dtype}, sample: {df['int_rate'].head(3).tolist()}")
print(f"revol_util dtype: {df['revol_util'].dtype}, sample: {df['revol_util'].head(3).tolist()}")
""")

# ==============================================================================
# STEP 4: FILTER TERMINAL STATUSES + CREATE TARGET
# ==============================================================================
md("""
## Step 4: Filter to Terminal Loan Statuses and Create Target

- **default=1**: Charged Off, Default
- **default=0**: Fully Paid
- **Drop**: Current, In Grace Period, Late variants, credit policy variants

Expected: ~1,345,350 terminal loans with ~19.96% default rate.
""")

code("""
print("Loan status distribution (full dataset):")
status_counts = df['loan_status'].value_counts()
for status, count in status_counts.items():
    pct = count / len(df) * 100
    print(f"  {status:55s} {count:>10,} ({pct:.2f}%)")

# Filter
terminal_statuses = DEFAULT_STATUSES + NON_DEFAULT_STATUSES
df = df[df['loan_status'].isin(terminal_statuses)].copy()
df['default'] = df['loan_status'].isin(DEFAULT_STATUSES).astype(int)
df = df.drop(columns=['loan_status'])

n_total = len(df)
n_default = df['default'].sum()
default_rate = n_default / n_total

print(f"\\n{'='*60}")
print(f"Terminal loans: {n_total:,}")
print(f"  Fully Paid: {(n_total - n_default):,} ({1 - default_rate:.2%})")
print(f"  Default:    {n_default:,} ({default_rate:.2%})")
print(f"{'='*60}")
print(f"Working DataFrame shape: {df.shape}")
""")

# ==============================================================================
# STEP 5: FRED MACRO MERGE
# ==============================================================================
md("""
## Step 5: Merge FRED Macroeconomic Data

**These macro features are CRITICAL for the time-based split validation.**
Without macro covariates, the model would overpredict defaults on the 2017-2018
test set because the training period (2007-2015) covered the 2008 crisis and
recovery. Macro features enable the model to learn cycle-adjusted default patterns.

| Series ID | Description |
|-----------|-------------|
| UNRATE | Unemployment Rate |
| CSUSHPINSA | Case-Shiller Home Price Index |
| A191RL1Q225SBEA | Real GDP Growth Rate (quarterly) |
| CPIAUCSL | Consumer Price Index |
| DFF | Federal Funds Rate |
| UMCSENT | Consumer Sentiment Index |
""")

code("""
df['issue_month'] = df['issue_d'].dt.to_period('M')

macro_monthly = None

# --- Attempt 1: fredapi with API key ---
try:
    from fredapi import Fred
    fred_api_key = os.environ.get('FRED_API_KEY')
    if not fred_api_key:
        raise ValueError("FRED_API_KEY not set")
    fred = Fred(api_key=fred_api_key)
    print("Connected to FRED API...")
    macro_frames = []
    for sid in FRED_SERIES:
        data = fred.get_series(sid, observation_start='2006-01-01',
                               observation_end='2019-12-31')
        s = data.to_frame(name=sid)
        s.index = pd.to_datetime(s.index)
        macro_frames.append(s)
        print(f"  {sid}: {len(data)} obs")
    macro_all = pd.concat(macro_frames, axis=1)
    macro_monthly = macro_all.resample('MS').mean()
    macro_monthly['A191RL1Q225SBEA'] = macro_monthly['A191RL1Q225SBEA'].ffill()
    macro_monthly = macro_monthly.ffill().bfill()
    print("FRED data downloaded via API")
except Exception as e:
    print(f"FRED API: {e}")

# --- Attempt 2: Direct CSV from FRED website ---
if macro_monthly is None:
    print("Attempting fallback: FRED CSV download...")
    try:
        base = "https://fred.stlouisfed.org/graph/fredgraph.csv"
        macro_frames = []
        for sid in FRED_SERIES:
            url = f"{base}?id={sid}&cosd=2006-01-01&coed=2019-12-31"
            temp = pd.read_csv(url, na_values=['.'])
            date_col = [c for c in temp.columns if 'date' in c.lower()][0]
            temp[date_col] = pd.to_datetime(temp[date_col])
            temp = temp.set_index(date_col)
            temp.columns = [sid]
            temp[sid] = pd.to_numeric(temp[sid], errors='coerce')
            macro_frames.append(temp)
            print(f"  {sid}: {len(temp)} obs")
        macro_all = pd.concat(macro_frames, axis=1)
        macro_monthly = macro_all.resample('MS').mean()
        macro_monthly['A191RL1Q225SBEA'] = macro_monthly['A191RL1Q225SBEA'].ffill()
        macro_monthly = macro_monthly.ffill().bfill()
        print("FRED data downloaded via CSV fallback")
    except Exception as e2:
        print(f"CSV fallback failed: {e2}")

# --- Merge ---
if macro_monthly is not None:
    macro_monthly.index = macro_monthly.index.to_period('M')
    df = df.merge(macro_monthly, left_on='issue_month', right_index=True, how='left')
    print(f"\\nMacro feature statistics:")
    print(df[FRED_SERIES].describe().round(2))
    macro_missing = df[FRED_SERIES].isna().sum()
    if macro_missing.any():
        print(f"\\nMissing macro values:\\n{macro_missing[macro_missing > 0]}")
    else:
        print("\\nNo missing macro values after merge")
else:
    print("\\nMACRO DATA NOT AVAILABLE. Creating NaN placeholders.")
    print("Set FRED_API_KEY: export FRED_API_KEY='your_key'")
    print("Register at: https://fred.stlouisfed.org/")
    for sid in FRED_SERIES:
        df[sid] = np.nan
""")

# ==============================================================================
# STEP 6: HANDLE MISSING VALUES
# ==============================================================================
md("""
## Step 6: Analyze and Handle Missing Values

### Tiered Strategy:
- **Tier 2** (70-85% missing): Create binary flag, then drop original
- **Tier 3** (38-68% missing): Create binary flag AND keep with median imputation
- **Tier 4** (<10% missing): Create flag if >10%, then impute

**Rule**: For ANY feature with >10% missing, ALWAYS create a binary flag.
""")

code("""
# Print full missingness table
missing = df.isnull().mean().sort_values(ascending=False)
missing_pct = missing[missing > 0]
print(f"Features with missing values: {len(missing_pct)}")
print(f"\\n{'Column':<45} {'Missing %':>10}")
print("-" * 57)
for col, pct in missing_pct.head(40).items():
    tier = ""
    if pct > 0.70: tier = " [T2]"
    elif pct > 0.30: tier = " [T3]"
    elif pct > 0.10: tier = " [T3/4]"
    print(f"  {col:<43} {pct:>9.2%}{tier}")
""")

code("""
# ── Tier 2: 70-85% missing — create flag then drop original ──
tier2_cols = {
    'mths_since_last_record': 'has_public_record',
    'mths_since_recent_bc_dlq': 'has_recent_bc_delinq',
    'mths_since_last_major_derog': 'has_major_derog',
}
for col, flag in tier2_cols.items():
    if col in df.columns:
        df[flag] = df[col].notna().astype(int)
        df = df.drop(columns=[col])
        print(f"  Tier 2: '{col}' -> flag '{flag}', dropped original")

# ── Tier 3: 38-68% missing — create flag AND keep with imputation ──
tier3_individual = {
    'mths_since_recent_revol_delinq': 'no_revol_delinq',
    'mths_since_last_delinq': 'no_delinq_history',
    'il_util': 'il_util_missing',
    'mths_since_rcnt_il': 'mths_since_rcnt_il_missing',
}
for col, flag in tier3_individual.items():
    if col in df.columns:
        df[flag] = df[col].isna().astype(int)
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"  Tier 3: '{col}' -> flag '{flag}', median={med:.1f}")

# Group flag for installment-related features (~38.31% missing)
inst_features = [
    'open_act_il', 'open_il_12m', 'open_il_24m', 'total_bal_il',
    'open_acc_6m', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
    'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m',
]
existing_inst = [c for c in inst_features if c in df.columns]
if existing_inst:
    df['installment_features_missing'] = df[existing_inst[0]].isna().astype(int)
    for col in existing_inst:
        med = df[col].median()
        df[col] = df[col].fillna(med)
    print(f"  Tier 3: {len(existing_inst)} installment features -> group flag, medians")

# ── Tier 4: <10% missing ──
df['emp_length_unknown'] = df['emp_length'].isna().astype(int)
df['emp_length'] = df['emp_length'].fillna(df['emp_length'].median())
print(f"  Tier 4: emp_length -> flag, median")

if 'mths_since_recent_inq' in df.columns:
    df['mths_since_recent_inq_missing'] = df['mths_since_recent_inq'].isna().astype(int)
    df['mths_since_recent_inq'] = df['mths_since_recent_inq'].fillna(
        df['mths_since_recent_inq'].median())
    print(f"  Tier 4: mths_since_recent_inq -> flag, median")

# Remaining <1% missing
for col in df.columns:
    if df[col].isna().any():
        pct = df[col].isna().mean()
        if pct > 0 and pct < 0.10:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                df[col] = df[col].fillna(df[col].median())
            elif df[col].dtype == 'object':
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])

# Check remaining
remaining = df.select_dtypes(include=[np.number]).isnull().sum()
remaining = remaining[remaining > 0]
print(f"\\nNumeric features still missing: {len(remaining)}")
print(f"Final feature count: {len(df.columns)}")
""")

# ==============================================================================
# STEP 6b: PRIORITY OUTLIER TREATMENT
# ==============================================================================
md("""
## Step 6b: Priority Outlier Treatment

Apply critical outlier corrections BEFORE splitting. These address known data
quality issues from full-file profiling.
""")

code("""
print("Priority Outlier Treatments:")
print("=" * 60)

# annual_inc: cap at 99th percentile (max $110M)
p99 = df['annual_inc'].quantile(0.99)
n = (df['annual_inc'] > p99).sum()
df['annual_inc'] = df['annual_inc'].clip(upper=p99)
print(f"  annual_inc: capped {n:,} at p99=${p99:,.0f}")

# dti: set < 0 or > 100 to NaN, impute
n = ((df['dti'] < 0) | (df['dti'] > 100)).sum()
df.loc[(df['dti'] < 0) | (df['dti'] > 100), 'dti'] = np.nan
df['dti'] = df['dti'].fillna(df['dti'].median())
print(f"  dti: cleaned {n:,} values outside [0,100]")

# revol_util: cap at 150%
if 'revol_util' in df.columns:
    n = (df['revol_util'] > 150).sum()
    df['revol_util'] = df['revol_util'].clip(upper=150)
    print(f"  revol_util: capped {n:,} at 150%")

# last_fico: set 0 to NaN
for fc in ['last_fico_range_low', 'last_fico_range_high']:
    if fc in df.columns:
        n = (df[fc] == 0).sum()
        df.loc[df[fc] == 0, fc] = np.nan
        df[fc] = df[fc].fillna(df[fc].median())
        print(f"  {fc}: fixed {n:,} zero values")

# Large balance columns: cap at p99.5
for col in ['tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 'total_rev_hi_lim']:
    if col in df.columns:
        p995 = df[col].quantile(0.995)
        n = (df[col] > p995).sum()
        df[col] = df[col].clip(upper=p995)
        print(f"  {col}: capped {n:,} at p99.5=${p995:,.0f}")

# Utilization: cap at 200%
for col in ['bc_util', 'il_util', 'all_util']:
    if col in df.columns:
        n = (df[col] > 200).sum()
        df[col] = df[col].clip(upper=200)
        print(f"  {col}: capped {n:,} at 200%")

print("\\nOutlier treatments applied.")
""")

# ==============================================================================
# STEP 7: FEATURE CATEGORIZATION
# ==============================================================================
md("""
## Step 7: Feature Categorization

### Borrower Demographics
`annual_inc`, `emp_length`, `home_ownership`, `verification_status`, `dti`, `addr_state`, `zip_code`

### Credit History (Core)
`fico_range_low`, `fico_range_high`, `earliest_cr_line`, `open_acc`, `total_acc`,
`revol_util`, `revol_bal`, `pub_rec`, `delinq_2yrs`, `inq_last_6mths`, `mths_since_last_delinq`

### Loan Characteristics
`loan_amnt`, `term`, `int_rate`, `grade`, `sub_grade`, `purpose`, `installment`, `funded_amnt`

### Credit Bureau (Extended)
~30 columns: `acc_open_past_24mths`, `avg_cur_bal`, `bc_open_to_buy`, `bc_util`,
`mort_acc`, `num_actv_bc_tl`, `num_actv_rev_tl`, `pct_tl_nvr_dlq`, `percent_bc_gt_75`,
`pub_rec_bankruptcies`, `tax_liens`, `tot_cur_bal`, `tot_hi_cred_lim`, etc.

### Macroeconomic (FRED)
`UNRATE`, `CSUSHPINSA`, `A191RL1Q225SBEA`, `CPIAUCSL`, `DFF`, `UMCSENT`

### Payment History (Leakage — LGD/EAD only, NOT PD)
`out_prncp`, `total_pymnt`, `total_rec_prncp`, `recoveries`, `collection_recovery_fee`, etc.

### Engineered Flags
`has_public_record`, `has_recent_bc_delinq`, `has_major_derog`, `no_revol_delinq`,
`no_delinq_history`, `il_util_missing`, `installment_features_missing`, `emp_length_unknown`
""")

# ==============================================================================
# STEP 8: TIME-BASED SPLIT
# ==============================================================================
md("""
## Step 8: Time-Based Train/Validation/Test Split

- **Train**: 2007-2015 (crisis, recovery, expansion)
- **Validation**: 2016 (expansion)
- **Test**: 2017-2018 (late expansion)

No random shuffling — credit models MUST validate out-of-time.
""")

code("""
df['issue_year'] = df['issue_d'].dt.year

train = df[df['issue_year'] <= TRAIN_END_YEAR].copy()
val = df[df['issue_year'] == VAL_YEAR].copy()
test = df[df['issue_year'] >= TEST_START_YEAR].copy()

# Verify non-overlapping
assert train['issue_d'].max() < val['issue_d'].min(), "Train/Val overlap!"
assert val['issue_d'].max() < test['issue_d'].min(), "Val/Test overlap!"

print("Time-Based Split Summary:")
print("=" * 70)
print(f"{'Split':<12} {'Records':>10} {'Default Rate':>14} {'Date Range'}")
print("-" * 70)
for name, sdf in [('Train', train), ('Validation', val), ('Test', test)]:
    n = len(sdf)
    dr = sdf['default'].mean()
    d_min = sdf['issue_d'].min().strftime('%Y-%m')
    d_max = sdf['issue_d'].max().strftime('%Y-%m')
    print(f"  {name:<10} {n:>10,} {dr:>13.2%}   {d_min} to {d_max}")
print("-" * 70)
print(f"  {'Total':<10} {len(df):>10,} {df['default'].mean():>13.2%}")

macro_present = [s for s in FRED_SERIES if s in train.columns]
print(f"\\nMacro features in splits: {len(macro_present)}/{len(FRED_SERIES)}")

# Save
train.to_parquet(DATA_PROCESSED_PATH / 'train.parquet', index=False)
val.to_parquet(DATA_PROCESSED_PATH / 'val.parquet', index=False)
test.to_parquet(DATA_PROCESSED_PATH / 'test.parquet', index=False)
print(f"\\nSaved: train.parquet ({len(train):,}), val.parquet ({len(val):,}), test.parquet ({len(test):,})")
""")

# ==============================================================================
# STEP 9: EDA VISUALIZATIONS
# ==============================================================================
md("""
## Step 9: EDA Visualizations

Publication-quality charts exploring default rates, distributions, correlations,
and portfolio composition.
""")

# --- 9a: Default rate by grade ---
md("### 9a: Default Rate by Grade")
code("""
grade_stats = df.groupby('grade')['default'].agg(['mean', 'count']).reset_index()
grade_stats.columns = ['grade', 'default_rate', 'count']
grade_stats = grade_stats.set_index('grade').loc[GRADE_ORDER].reset_index()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
bars = axes[0].bar(grade_stats['grade'], grade_stats['default_rate'] * 100,
                   color=sns.color_palette('YlOrRd', 7))
axes[0].set_xlabel('Grade'); axes[0].set_ylabel('Default Rate (%)')
axes[0].set_title('Default Rate by Grade')
for bar, r in zip(bars, grade_stats['default_rate']):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{r:.1%}', ha='center', fontsize=10)

axes[1].bar(grade_stats['grade'], grade_stats['count'],
            color=sns.color_palette('Blues', 7))
axes[1].set_xlabel('Grade'); axes[1].set_ylabel('Loans')
axes[1].set_title('Loan Volume by Grade')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
plt.tight_layout()
plt.savefig(DATA_RESULTS_PATH / 'default_rate_by_grade.png', dpi=150, bbox_inches='tight')
plt.show()

# Verify monotonicity
rates = grade_stats['default_rate'].values
mono = all(rates[i] < rates[i+1] for i in range(len(rates)-1))
print(f"\\nGrade monotonicity: {'PASS' if mono else 'FAIL'}")
for _, row in grade_stats.iterrows():
    print(f"  {row['grade']}: {row['default_rate']:.2%} ({row['count']:,})")
""")

# --- 9b: Default rate by sub-grade ---
md("### 9b: Default Rate by Sub-Grade")
code("""
sg = df.groupby('sub_grade')['default'].mean().reset_index()
sg.columns = ['sub_grade', 'default_rate']

fig, ax = plt.subplots(figsize=(18, 6))
colors = [plt.cm.YlOrRd(GRADE_ORDER.index(s[0]) / 7) for s in sg['sub_grade']]
ax.bar(range(len(sg)), sg['default_rate'] * 100, color=colors)
ax.set_xticks(range(len(sg)))
ax.set_xticklabels(sg['sub_grade'], rotation=45, ha='right', fontsize=8)
ax.set_xlabel('Sub-Grade'); ax.set_ylabel('Default Rate (%)')
ax.set_title('Default Rate by Sub-Grade')
plt.tight_layout()
plt.savefig(DATA_RESULTS_PATH / 'default_rate_by_subgrade.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# --- 9c: Default rate by vintage ---
md("### 9c: Default Rate by Origination Year (Vintage)")
code("""
vint = df.groupby('issue_year')['default'].agg(['mean', 'count']).reset_index()
vint.columns = ['year', 'default_rate', 'count']

fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()
ax1.bar(vint['year'], vint['count'], alpha=0.3, color='steelblue', label='Volume')
ax2.plot(vint['year'], vint['default_rate'] * 100, 'ro-', lw=2, ms=8, label='Default Rate')
ax1.set_xlabel('Year'); ax1.set_ylabel('Loans', color='steelblue')
ax2.set_ylabel('Default Rate (%)', color='red')
ax1.set_title('Default Rate and Volume by Vintage')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
plt.savefig(DATA_RESULTS_PATH / 'default_rate_by_vintage.png', dpi=150, bbox_inches='tight')
plt.show()

for _, r in vint.iterrows():
    print(f"  {int(r['year'])}: {r['default_rate']:.2%} ({int(r['count']):,})")
""")

# --- 9d: Default rate by term ---
md("### 9d: Default Rate by Term")
code("""
ts = df.groupby('term')['default'].agg(['mean', 'count']).reset_index()
ts.columns = ['term', 'default_rate', 'count']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([f"{int(t)}m" for t in ts['term']], ts['default_rate'] * 100,
              color=['steelblue', 'coral'], width=0.5)
ax.set_ylabel('Default Rate (%)')
ax.set_title('Default Rate by Loan Term')
for bar, r in zip(bars, ts['default_rate']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{r:.1%}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(DATA_RESULTS_PATH / 'default_rate_by_term.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# --- 9e: Distribution KDEs ---
md("### 9e: Feature Distributions by Default Status")
code("""
kde_feats = ['fico_range_low', 'dti', 'annual_inc', 'int_rate',
             'revol_util', 'open_acc', 'total_acc', 'inq_last_6mths']
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, feat in enumerate(kde_feats):
    ax = axes.flatten()[i]
    if feat in df.columns:
        d0 = df.loc[df['default'] == 0, feat].dropna()
        d1 = df.loc[df['default'] == 1, feat].dropna()
        lo = min(d0.quantile(0.01), d1.quantile(0.01))
        hi = max(d0.quantile(0.99), d1.quantile(0.99))
        ax.hist(d0.clip(lo, hi), bins=50, density=True, alpha=0.5,
                color='steelblue', label='Non-Default')
        ax.hist(d1.clip(lo, hi), bins=50, density=True, alpha=0.5,
                color='coral', label='Default')
        ax.set_title(feat); ax.legend(fontsize=8)

plt.suptitle('Feature Distributions by Default Status', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(DATA_RESULTS_PATH / 'distributions_by_default.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# --- 9f: Correlation heatmap ---
md("### 9f: Correlation Matrix (Top 20 Features)")
code("""
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude = ['_missing', '_unknown', 'has_', 'no_', 'default', 'issue_year']
top_num = [c for c in numeric_cols if not any(p in c for p in exclude)]
non_null = df[top_num].notna().sum().sort_values(ascending=False)
top_20 = non_null.head(20).index.tolist()

corr = df[top_20].corr()
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, square=True, linewidths=0.5,
            annot_kws={'size': 8})
ax.set_title('Correlation Matrix - Top 20 Numeric Features')
plt.tight_layout()
plt.savefig(DATA_RESULTS_PATH / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# --- 9g: Geographic ---
md("### 9g: Geographic Default Rate")
code("""
st = df.groupby('addr_state')['default'].agg(['mean', 'count']).reset_index()
st.columns = ['state', 'default_rate', 'count']
st = st.sort_values('default_rate', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
bot = st.head(10)
axes[0].barh(bot['state'], bot['default_rate'] * 100, color='steelblue')
axes[0].set_xlabel('Default Rate (%)'); axes[0].set_title('10 Lowest Default Rate States')
top = st.tail(10)
axes[1].barh(top['state'], top['default_rate'] * 100, color='coral')
axes[1].set_xlabel('Default Rate (%)'); axes[1].set_title('10 Highest Default Rate States')
plt.tight_layout()
plt.savefig(DATA_RESULTS_PATH / 'geographic_default_rate.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# --- 9h: Portfolio composition ---
md("### 9h: Portfolio Composition Over Time")
code("""
comp = df.groupby(['issue_year', 'grade']).size().unstack(fill_value=0)
comp = comp[GRADE_ORDER]

fig, ax = plt.subplots(figsize=(14, 7))
comp.plot(kind='bar', stacked=True, ax=ax,
          color=sns.color_palette('YlOrRd', 7))
ax.set_xlabel('Origination Year'); ax.set_ylabel('Loans')
ax.set_title('Portfolio Composition by Grade Over Time')
ax.legend(title='Grade', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
plt.tight_layout()
plt.savefig(DATA_RESULTS_PATH / 'portfolio_composition.png', dpi=150, bbox_inches='tight')
plt.show()
""")

# --- 9i: Macro trends ---
md("### 9i: Macroeconomic Trends Over Origination Period")
code("""
macro_present = [s for s in FRED_SERIES if s in df.columns and df[s].notna().any()]

if macro_present:
    macro_agg = df.groupby(df['issue_d'].dt.to_period('M'))[macro_present].mean()
    macro_agg.index = macro_agg.index.to_timestamp()

    titles = {
        'UNRATE': 'Unemployment Rate (%)',
        'CSUSHPINSA': 'Case-Shiller HPI',
        'A191RL1Q225SBEA': 'Real GDP Growth (%)',
        'CPIAUCSL': 'Consumer Price Index',
        'DFF': 'Federal Funds Rate (%)',
        'UMCSENT': 'Consumer Sentiment',
    }

    n_plots = min(len(macro_present), 6)
    rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(16, 4 * rows))
    axes = axes.flatten() if n_plots > 2 else [axes] if n_plots == 1 else axes.flatten()

    for i, col in enumerate(macro_present[:6]):
        ax = axes[i]
        ax.plot(macro_agg.index, macro_agg[col], lw=2, color='steelblue')
        ax.axvline(pd.Timestamp('2008-09-15'), color='red', ls='--', alpha=0.5, label='Lehman')
        ax.axvline(pd.Timestamp('2015-12-16'), color='green', ls='--', alpha=0.5, label='Rate Hike')
        ax.set_title(titles.get(col, col))
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Macroeconomic Trends with Regime Markers', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(DATA_RESULTS_PATH / 'macro_trends.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("No macro data available for plotting.")
""")

# ==============================================================================
# STEP 9.5: COMPREHENSIVE DATA PROFILING
# ==============================================================================
md("""
## Step 9.5: Comprehensive Data Profiling

Full-file validation report covering:
- A. Outlier detection (post-treatment summary)
- B. Distribution analysis
- C. Correlation analysis (high pairs)
- D. Sanity checks
- E. Categorical value counts
""")

# --- 9.5A: Outlier Detection ---
md("### 9.5A: Outlier Detection Summary")
code("""
profiling = {'outliers': {}, 'distributions': {}, 'correlations': {},
             'sanity_checks': {}, 'categorical_counts': {}}

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
skip_cols = ['default', 'issue_year'] + FRED_SERIES

outlier_rows = []
for col in numeric_cols:
    if col in skip_cols:
        continue
    s = df[col].dropna()
    if len(s) == 0:
        continue
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        continue
    lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
    n_out = ((s < lo) | (s > hi)).sum()
    pct_out = n_out / len(s)
    p1, p99 = s.quantile(0.01), s.quantile(0.99)
    outlier_rows.append({
        'column': col, 'n_outliers': int(n_out), 'pct': round(pct_out, 4),
        'min': round(float(s.min()), 2), 'p1': round(float(p1), 2),
        'p99': round(float(p99), 2), 'max': round(float(s.max()), 2),
    })
    profiling['outliers'][col] = {
        'n_outliers': int(n_out), 'pct': round(pct_out, 4),
        'p1': round(float(p1), 2), 'p99': round(float(p99), 2),
    }

outlier_df = pd.DataFrame(outlier_rows).sort_values('pct', ascending=False)
print("Outlier Summary (top 20 by %):")
print(outlier_df.head(20).to_string(index=False))
""")

# --- 9.5B: Distribution Analysis ---
md("### 9.5B: Distribution Analysis")
code("""
dist_rows = []
for col in numeric_cols:
    if col in skip_cols:
        continue
    s = df[col].dropna()
    if len(s) < 10:
        continue
    sk = float(s.skew())
    ku = float(s.kurtosis())
    shape = 'Normal' if abs(sk) < 0.5 else 'Moderate Skew' if abs(sk) < 1 else 'High Skew'
    needs_log = abs(sk) > 2
    dist_rows.append({
        'column': col, 'mean': round(float(s.mean()), 2),
        'median': round(float(s.median()), 2),
        'std': round(float(s.std()), 2),
        'skew': round(sk, 2), 'kurtosis': round(ku, 2),
        'shape': shape, 'log_transform': needs_log,
    })
    profiling['distributions'][col] = {
        'mean': round(float(s.mean()), 2), 'median': round(float(s.median()), 2),
        'skew': round(sk, 2), 'kurtosis': round(ku, 2), 'shape': shape,
    }

dist_df = pd.DataFrame(dist_rows)
print(f"Distribution summary ({len(dist_df)} features):")
print(f"  Normal: {(dist_df['shape'] == 'Normal').sum()}")
print(f"  Moderate Skew: {(dist_df['shape'] == 'Moderate Skew').sum()}")
print(f"  High Skew: {(dist_df['shape'] == 'High Skew').sum()}")
print(f"  Need log-transform: {dist_df['log_transform'].sum()}")
print("\\nHigh-skew features:")
print(dist_df[dist_df['log_transform']]['column'].tolist())
""")

# --- 9.5C: Correlation Analysis ---
md("### 9.5C: Correlation Analysis (|corr| > 0.80)")
code("""
all_numeric = [c for c in numeric_cols if c not in ['default', 'issue_year']]
corr_full = df[all_numeric].corr()

# Extract pairs with |corr| > 0.80
high_corr = []
for i in range(len(corr_full.columns)):
    for j in range(i+1, len(corr_full.columns)):
        c = corr_full.iloc[i, j]
        if abs(c) > 0.80:
            high_corr.append({
                'feature_1': corr_full.columns[i],
                'feature_2': corr_full.columns[j],
                'correlation': round(float(c), 4),
            })

high_corr_df = pd.DataFrame(high_corr).sort_values('correlation', ascending=False, key=abs)
print(f"Pairs with |corr| > 0.80: {len(high_corr_df)}")
print(high_corr_df.to_string(index=False))

# Known redundant pairs — recommendations
keep_drop = {
    ('loan_amnt', 'funded_amnt'): 'Keep loan_amnt, drop funded_amnt',
    ('loan_amnt', 'funded_amnt_inv'): 'Keep loan_amnt, drop funded_amnt_inv',
    ('fico_range_low', 'fico_range_high'): 'Keep fico_range_low, drop fico_range_high',
    ('last_fico_range_low', 'last_fico_range_high'): 'Keep last_fico_range_low',
    ('out_prncp', 'out_prncp_inv'): 'Keep out_prncp, drop out_prncp_inv',
    ('total_pymnt', 'total_pymnt_inv'): 'Keep total_pymnt, drop total_pymnt_inv',
}
print("\\nRedundancy Recommendations:")
for pair, rec in keep_drop.items():
    f1, f2 = pair
    if f1 in corr_full.columns and f2 in corr_full.columns:
        c = corr_full.loc[f1, f2]
        print(f"  {f1} <-> {f2}: corr={c:.4f} | {rec}")
        profiling['correlations'][f'{f1}__{f2}'] = {
            'corr': round(float(c), 4), 'recommendation': rec,
        }
""")

# --- 9.5D: Sanity Checks ---
md("### 9.5D: Full-File Sanity Checks")
code("""
checks = {}

# Row count (full dataset was 2,260,668 before terminal filter)
checks['terminal_loans'] = {
    'expected': '~1,345,350', 'actual': len(df),
    'status': 'PASS' if abs(len(df) - 1345350) < 5000 else 'CHECK',
}

# Default rate
dr = df['default'].mean()
checks['default_rate'] = {
    'expected': '~19.96%', 'actual': f'{dr:.4%}',
    'status': 'PASS' if abs(dr - 0.1996) < 0.01 else 'CHECK',
}

# Grade monotonicity
gr = df.groupby('grade')['default'].mean()
gr = gr.reindex(GRADE_ORDER)
mono = all(gr.iloc[i] < gr.iloc[i+1] for i in range(len(gr)-1))
checks['grade_monotonicity'] = {
    'expected': 'A < B < C < D < E < F < G', 'actual': str(mono), 'status': 'PASS' if mono else 'FAIL',
}

# Term values
term_vals = set(df['term'].unique())
checks['term_values'] = {
    'expected': '{36, 60}', 'actual': str(term_vals),
    'status': 'PASS' if term_vals == {36, 60} else 'FAIL',
}

# emp_length range
el = set(df['emp_length'].dropna().unique())
checks['emp_length_range'] = {
    'expected': '{0..10}', 'actual': str(sorted(el)),
    'status': 'PASS' if el.issubset(set(range(11))) else 'CHECK',
}

# int_rate range
ir_min, ir_max, ir_mean = df['int_rate'].min(), df['int_rate'].max(), df['int_rate'].mean()
checks['int_rate'] = {
    'expected': '[5.31, 30.99], mean~13.09',
    'actual': f'[{ir_min:.2f}, {ir_max:.2f}], mean={ir_mean:.2f}',
    'status': 'PASS' if 5 < ir_min < 6 and 30 < ir_max < 32 else 'CHECK',
}

# fico_range_low
fl_min, fl_max, fl_mean = df['fico_range_low'].min(), df['fico_range_low'].max(), df['fico_range_low'].mean()
checks['fico_range_low'] = {
    'expected': '[610, 845], mean~698.59',
    'actual': f'[{fl_min:.0f}, {fl_max:.0f}], mean={fl_mean:.2f}',
    'status': 'PASS' if 600 < fl_min < 620 and 840 < fl_max < 850 else 'CHECK',
}

# loan_amnt
la_min, la_max, la_mean = df['loan_amnt'].min(), df['loan_amnt'].max(), df['loan_amnt'].mean()
checks['loan_amnt'] = {
    'expected': '[500, 40000], mean~$15,047',
    'actual': f'[{la_min:,.0f}, {la_max:,.0f}], mean=${la_mean:,.0f}',
    'status': 'PASS' if 400 < la_min < 600 and 39000 < la_max < 41000 else 'CHECK',
}

# Accounting check
if all(c in df.columns for c in ['total_pymnt', 'total_rec_prncp', 'total_rec_int',
                                   'total_rec_late_fee', 'recoveries']):
    calc = df['total_rec_prncp'] + df['total_rec_int'] + df['total_rec_late_fee'] + df['recoveries']
    diff = (df['total_pymnt'] - calc).abs()
    pct_close = (diff < 1.0).mean()
    checks['accounting_check'] = {
        'expected': 'total_pymnt ~ sum of components',
        'actual': f'{pct_close:.2%} within $1',
        'status': 'PASS' if pct_close > 0.95 else 'CHECK',
    }

profiling['sanity_checks'] = checks

print("Sanity Checks:")
print("=" * 80)
for name, chk in checks.items():
    status = chk['status']
    print(f"  [{status:>5}] {name}: expected={chk['expected']}, actual={chk['actual']}")
""")

# --- 9.5E: Categorical Counts ---
md("### 9.5E: Categorical Value Counts")
code("""
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical features: {len(cat_cols)}")

for col in cat_cols:
    nunique = df[col].nunique()
    if col in ['emp_title', 'title']:
        print(f"\\n{col}: {nunique:,} unique values (high cardinality)")
        print(f"  Top 10: {df[col].value_counts().head(10).to_dict()}")
    elif col in ['zip_code']:
        print(f"\\n{col}: {nunique:,} unique values")
        print(f"  Top 10: {df[col].value_counts().head(10).to_dict()}")
    elif nunique < 50:
        print(f"\\n{col} ({nunique} unique):")
        vc = df[col].value_counts()
        for v, cnt in vc.items():
            pct = cnt / len(df) * 100
            print(f"  {v:45s} {cnt:>10,} ({pct:.2f}%)")
        profiling['categorical_counts'][col] = {
            str(k): int(v) for k, v in vc.to_dict().items()
        }

# addr_state summary
if 'addr_state' in df.columns:
    print(f"\\naddr_state: {df['addr_state'].nunique()} unique states")
    print(f"  Top 10: {df['addr_state'].value_counts().head(10).to_dict()}")
""")

# ==============================================================================
# STEP 10: SAVE OUTPUTS
# ==============================================================================
md("""
## Step 10: Save All Outputs

Final outputs:
- `data/processed/loans_cleaned.parquet` — full cleaned dataset with macro features
- `data/processed/train.parquet`, `val.parquet`, `test.parquet` — time-based splits
- `data/results/eda_summary.json` — key statistics
- `data/results/full_profiling_report.json` — comprehensive profiling
""")

code("""
# Save cleaned dataset
df.to_parquet(DATA_PROCESSED_PATH / 'loans_cleaned.parquet', index=False)
print(f"Saved loans_cleaned.parquet: {df.shape}")

# Save profiling report
with open(DATA_RESULTS_PATH / 'full_profiling_report.json', 'w') as f:
    json.dump(profiling, f, indent=2, default=str)
print(f"Saved full_profiling_report.json")

# Save EDA summary
macro_stats = {}
for s in FRED_SERIES:
    if s in df.columns and df[s].notna().any():
        macro_stats[s] = {
            'mean': round(float(df[s].mean()), 4),
            'std': round(float(df[s].std()), 4),
            'min': round(float(df[s].min()), 4),
            'max': round(float(df[s].max()), 4),
        }

eda_summary = {
    'total_records': int(len(df)),
    'default_rate': round(float(df['default'].mean()), 4),
    'features_kept': int(len(df.columns)),
    'train_records': int(len(train)),
    'val_records': int(len(val)),
    'test_records': int(len(test)),
    'train_default_rate': round(float(train['default'].mean()), 4),
    'val_default_rate': round(float(val['default'].mean()), 4),
    'test_default_rate': round(float(test['default'].mean()), 4),
    'macro_features': macro_stats,
    'sanity_checks': profiling['sanity_checks'],
}

with open(DATA_RESULTS_PATH / 'eda_summary.json', 'w') as f:
    json.dump(eda_summary, f, indent=2, default=str)
print(f"Saved eda_summary.json")

print("\\n" + "=" * 60)
print("NOTEBOOK 01 COMPLETE")
print("=" * 60)
print(f"  Records: {len(df):,}")
print(f"  Features: {len(df.columns)}")
print(f"  Default rate: {df['default'].mean():.2%}")
print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
print(f"  Macro features: {len(macro_stats)}/{len(FRED_SERIES)}")
""")

# ==============================================================================
# ASSEMBLE AND SAVE
# ==============================================================================
nb.cells = cells

import os
outdir = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, '01_EDA_and_Data_Cleaning.ipynb')

with open(outpath, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook created: {outpath}")
print(f"Total cells: {len(cells)}")
