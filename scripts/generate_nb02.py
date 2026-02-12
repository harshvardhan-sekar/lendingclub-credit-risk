#!/usr/bin/env python3
"""
Generate Notebook 02: WOE/IV Feature Engineering.

Run:  python scripts/generate_nb02.py
"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11.0"},
})

def md(source: str):
    nb.cells.append(nbf.v4.new_markdown_cell(source.strip()))

def code(source: str):
    nb.cells.append(nbf.v4.new_code_cell(source.strip()))

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 0 — Title
# ═══════════════════════════════════════════════════════════════════════════════
md("""
# Notebook 02: WOE/IV Feature Engineering

## Prior Role Connection
This notebook directly mirrors behavioral scorecard monitoring work — computing
Weight of Evidence (WoE), Information Value (IV), and Characteristic Stability
Index (CSI) for credit portfolio variables. The same methodology applies to
VantageScore/FICO bins, utilization ratios, DTI, inquiries, open tradelines,
months on book, and delinquency indicators.

## Objective
1. Engineer derived features from the cleaned training data
2. Compute optimal WoE bins for **all candidate PD features** (~70 variables)
3. Rank features by IV and apply selection thresholds
4. Validate monotonicity of bad rates across bins
5. Save WoE-transformed datasets for the logistic regression scorecard (Notebook 03)

## Key Design Decisions
- **grade IS included** — it is a legitimate behavioral feature for portfolio monitoring
- **int_rate and sub_grade are EXCLUDED** — collinear with grade
- **Macro features are included** as candidates — expected to have low IV (< 0.02)
  in univariate screening; their value emerges in ML models (Notebook 04)
- WoE bins are fit **only on training data** (2007-2015), then applied to val/test
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Setup
# ═══════════════════════════════════════════════════════════════════════════════
md("""
## Step 1: Setup — Imports and Configuration
""")

code("""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Project imports — resolve parent of notebooks/ to find config.py and src/
sys.path.insert(0, str(Path('..').resolve()))
from config import (
    DATA_PROCESSED_PATH, DATA_RESULTS_PATH, DATA_MODELS_PATH,
    TARGET_COL, RANDOM_STATE,
)
from src.woe_binning import WOEBinner

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 80)
pd.set_option("display.float_format", "{:.4f}".format)

print("Setup complete.")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Load data
# ═══════════════════════════════════════════════════════════════════════════════
md("""
## Step 2: Load Training Data

We load **only the training set** for fitting WoE bins — the validation and test
sets will be transformed later using the fitted binner (no data leakage).
""")

code("""
train = pd.read_parquet(DATA_PROCESSED_PATH / "train.parquet")
val   = pd.read_parquet(DATA_PROCESSED_PATH / "val.parquet")
test  = pd.read_parquet(DATA_PROCESSED_PATH / "test.parquet")

y_train = train[TARGET_COL]
y_val   = val[TARGET_COL]
y_test  = test[TARGET_COL]

print(f"Train: {train.shape[0]:,} rows × {train.shape[1]} cols  |  Default rate: {y_train.mean():.4f}")
print(f"Val:   {val.shape[0]:,} rows × {val.shape[1]} cols  |  Default rate: {y_val.mean():.4f}")
print(f"Test:  {test.shape[0]:,} rows × {test.shape[1]} cols  |  Default rate: {y_test.mean():.4f}")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 — Feature engineering
# ═══════════════════════════════════════════════════════════════════════════════
md("""
## Step 3: Engineered Features

Create derived variables that capture risk dimensions not directly available
in the raw columns.  These mirror standard scorecard practice: credit age,
average FICO, payment-to-income, and binary flags.
""")

code("""
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Create derived features for scorecard modeling.\"\"\"
    df = df.copy()

    # credit_history_years: years since earliest credit line
    if "earliest_cr_line" in df.columns:
        ref_date = pd.Timestamp("2018-12-31")
        ecl = pd.to_datetime(df["earliest_cr_line"], errors="coerce")
        df["credit_history_years"] = (ref_date - ecl).dt.days / 365.25

    # fico_avg: midpoint of FICO range
    if "fico_range_low" in df.columns and "fico_range_high" in df.columns:
        df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    # loan_to_income: loan amount relative to annual income
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)

    # installment_to_income: monthly payment burden
    if "installment" in df.columns and "annual_inc" in df.columns:
        monthly_inc = df["annual_inc"].replace(0, np.nan) / 12
        df["installment_to_income"] = df["installment"] / monthly_inc

    # total_credit_utilization: revolving balance / total revolving limit
    if "revol_bal" in df.columns and "total_rev_hi_lim" in df.columns:
        df["total_credit_utilization"] = (
            df["revol_bal"] / df["total_rev_hi_lim"].replace(0, np.nan)
        )

    # Binary flags
    if "delinq_2yrs" in df.columns:
        df["delinq_flag"] = (df["delinq_2yrs"] > 0).astype(int)

    if "inq_last_6mths" in df.columns:
        df["recent_inquiry_flag"] = (df["inq_last_6mths"] > 2).astype(int)

    if "dti" in df.columns:
        df["high_dti_flag"] = (df["dti"] > 30).astype(int)

    return df


train = engineer_features(train)
val   = engineer_features(val)
test  = engineer_features(test)

new_feats = [
    "credit_history_years", "fico_avg", "loan_to_income",
    "installment_to_income", "total_credit_utilization",
    "delinq_flag", "recent_inquiry_flag", "high_dti_flag",
]
print("Engineered features created:")
for f in new_feats:
    if f in train.columns:
        non_null = train[f].notna().sum()
        print(f"  {f}: non-null={non_null:,}, mean={train[f].mean():.4f}")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 — Define candidate features
# ═══════════════════════════════════════════════════════════════════════════════
md("""
## Step 4: Define Candidate Features for WoE Binning

We include **all PD-eligible features** from the CLAUDE.md specification:
- Borrower characteristics (income, employment, home ownership, etc.)
- Credit bureau variables (FICO, delinquency, inquiries, tradelines, etc.)
- Loan attributes (amount, term, grade, purpose, etc.)
- Geographic (state, zip)
- Macro (FRED series — expected low IV)
- Engineered features (Step 3)
- Binary flags from Notebook 01 (no_delinq_history, etc.)

**Excluded:**
- `int_rate` and `sub_grade` (collinear with grade)
- All leakage variables (post-origination outcomes)
- ID / date / administrative columns
""")

code("""
# ── Leakage variables — NEVER use for PD ──
LEAKAGE_COLS = {
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt",
    "last_pymnt_d", "last_fico_range_high", "last_fico_range_low",
    "next_pymnt_d", "last_credit_pull_d", "hardship_flag",
    "debt_settlement_flag",
}

# ── Non-feature / ID / date columns ──
NON_FEATURE_COLS = {
    TARGET_COL, "issue_d", "issue_month", "issue_year",
    "earliest_cr_line", "emp_title", "title", "zip_code",
}

# ── Excluded due to collinearity with grade ──
EXCLUDED_COLLINEAR = {"int_rate", "sub_grade"}

# ── Drop columns that are in the dataset but not suitable for WoE ──
DROP_COLS = LEAKAGE_COLS | NON_FEATURE_COLS | EXCLUDED_COLLINEAR

# ── All remaining columns become candidates ──
candidate_features = sorted([
    c for c in train.columns if c not in DROP_COLS
])

print(f"Total candidate features for WoE binning: {len(candidate_features)}")
print(f"  Excluded leakage:     {len(LEAKAGE_COLS)}")
print(f"  Excluded non-feature: {len(NON_FEATURE_COLS)}")
print(f"  Excluded collinear:   {len(EXCLUDED_COLLINEAR)} (int_rate, sub_grade)")
print()
print("Candidate features:")
for i, f in enumerate(candidate_features, 1):
    dtype = train[f].dtype
    null_pct = train[f].isna().mean() * 100
    print(f"  {i:3d}. {f:<40s} dtype={str(dtype):<15s} null={null_pct:.1f}%")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 — Fit WOE
# ═══════════════════════════════════════════════════════════════════════════════
md("""
## Step 5: Fit WoE Bins on Training Data

We use `WOEBinner` (from `src/woe_binning.py`) which wraps `optbinning.OptimalBinning`
to find optimal bins for each candidate feature.  Bins are fit **only** on the
training set (2007-2015).
""")

code("""
binner = WOEBinner(
    max_n_bins=10,
    min_bin_size=0.05,
    monotonic_trend="auto",
    cat_cutoff=0.01,
    random_state=RANDOM_STATE,
)

binner.fit(train, y_train, feature_names=candidate_features)
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 — IV summary
# ═══════════════════════════════════════════════════════════════════════════════
md("""
## Step 6: Information Value (IV) Summary

IV interpretation (standard credit risk thresholds):

| IV Range | Predictive Power | Action |
|----------|-----------------|--------|
| < 0.02   | Not predictive  | Drop   |
| 0.02–0.10 | Weak           | Flag for review |
| 0.10–0.30 | Medium         | Include |
| 0.30–0.50 | Strong         | Include |
| > 0.50   | Suspicious / Very Strong | Investigate — may indicate leakage |

**Note:** `grade` will appear with IV > 0.5.  This is **expected and legitimate** —
grade captures LendingClub's internal risk assessment and is a known strong predictor
for behavioral scorecards.
""")

code("""
iv_df = binner.iv_summary()

print("=" * 80)
print("INFORMATION VALUE (IV) SUMMARY — All Candidate Features")
print("=" * 80)
print(iv_df.to_string(index=False))
print()

# Summary statistics
print("\\n" + "=" * 80)
print("IV SUMMARY STATISTICS")
print("=" * 80)
print(f"  Total features binned:   {len(iv_df)}")
print(f"  Mean IV:                 {iv_df['iv'].mean():.4f}")
print(f"  Median IV:               {iv_df['iv'].median():.4f}")
print(f"  Max IV:                  {iv_df['iv'].max():.4f}  ({iv_df.iloc[0]['feature']})")
print(f"  Min IV:                  {iv_df['iv'].min():.6f}  ({iv_df.iloc[-1]['feature']})")
print()
status_counts = iv_df["selection_status"].value_counts()
print("Count by selection status:")
for status in ["suspicious", "strong", "medium", "weak", "drop"]:
    count = status_counts.get(status, 0)
    print(f"  {status:<15s}: {count:3d}")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 — IV bar chart
# ═══════════════════════════════════════════════════════════════════════════════
md("""
### IV Visualization — Top Features by Predictive Power
""")

code("""
fig = binner.plot_top_features(n=30, figsize=(14, 10))
plt.savefig(DATA_RESULTS_PATH / "iv_top_features.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: data/results/iv_top_features.png")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 — Feature selection
# ═══════════════════════════════════════════════════════════════════════════════
md("""
## Step 7: IV-Based Feature Selection

Apply standard IV thresholds.  Features with IV > 0.5 are flagged as
"suspicious" but `grade` is a **known exception** — it captures
LendingClub's risk assessment and is legitimately strong.

We include all features with IV ≥ 0.02.  Features in the "weak" band
(0.02–0.10) are included but flagged for review in Notebook 03 during
multivariate selection.
""")

code("""
# ── Override: grade is legitimate even if IV > 0.5 ──
iv_df.loc[iv_df["feature"] == "grade", "selection_status"] = "strong"

# ── Feature selection ──
selected_mask = iv_df["selection_status"].isin(["weak", "medium", "strong", "suspicious"])
selected_features = iv_df.loc[selected_mask, "feature"].tolist()
dropped_features  = iv_df.loc[~selected_mask, "feature"].tolist()

# Suspicious features (IV > 0.5) excluding grade
suspicious = iv_df.loc[
    (iv_df["selection_status"] == "suspicious") & (iv_df["feature"] != "grade")
]

print("=" * 80)
print("FEATURE SELECTION RESULTS")
print("=" * 80)
print(f"  Selected features (IV >= 0.02):  {len(selected_features)}")
print(f"  Dropped features  (IV <  0.02):  {len(dropped_features)}")
print()

if len(suspicious) > 0:
    print("⚠  SUSPICIOUS features (IV > 0.5, excluding grade):")
    for _, row in suspicious.iterrows():
        print(f"    {row['feature']}: IV = {row['iv']:.4f}")
    print("  → Review these for potential data leakage before Notebook 03.")
else:
    print("  No suspicious features (IV > 0.5) besides grade — clean.")

print()
print("SELECTED features:")
for i, f in enumerate(selected_features, 1):
    iv_val = iv_df.loc[iv_df["feature"] == f, "iv"].values[0]
    status = iv_df.loc[iv_df["feature"] == f, "selection_status"].values[0]
    print(f"  {i:3d}. {f:<40s} IV={iv_val:.4f}  [{status}]")

print()
print("DROPPED features (IV < 0.02):")
for f in dropped_features:
    iv_val = iv_df.loc[iv_df["feature"] == f, "iv"].values[0]
    print(f"  - {f:<40s} IV={iv_val:.6f}")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 — Monotonicity
# ═══════════════════════════════════════════════════════════════════════════════
md("""
## Step 8: Monotonicity Validation

For a well-specified scorecard, the bad rate (event rate) should increase or
decrease monotonically across WoE bins.  Minor violations are acceptable and
documented; severe violations may indicate data quality issues.
""")

code("""
mono_report = binner.monotonicity_report()
mono_report = mono_report.merge(
    iv_df[["feature", "iv", "selection_status"]],
    on="feature", how="left",
)

# Focus on selected features
mono_selected = mono_report[mono_report["feature"].isin(selected_features)]

print("=" * 80)
print("MONOTONICITY REPORT — Selected Features")
print("=" * 80)
n_mono = mono_selected["monotonic"].sum()
n_total = len(mono_selected)
print(f"  Monotonic: {n_mono} / {n_total}  ({n_mono/n_total*100:.1f}%)")
print()

violations = mono_selected[~mono_selected["monotonic"]].sort_values("n_violations", ascending=False)
if len(violations) > 0:
    print("Features with monotonicity violations:")
    print(violations[["feature", "direction", "n_violations", "iv"]].to_string(index=False))
    print()
    print("Note: Minor violations are acceptable for optbinning auto-monotonic mode.")
    print("These features remain candidates; Notebook 03 multivariate selection")
    print("will further validate rank-ordering.")
else:
    print("All selected features show monotonic bad rates — excellent.")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 — WoE plots for top features
# ═══════════════════════════════════════════════════════════════════════════════
md("""
### WoE Bin Plots — Top Selected Features

Visual inspection of WoE patterns and event rates for the most predictive features.
""")

code("""
# Plot top 12 features by IV
top_feats = iv_df.loc[iv_df["feature"].isin(selected_features)].head(12)["feature"].tolist()

fig, axes = plt.subplots(4, 3, figsize=(20, 22))
axes_flat = axes.flatten()

for i, feat in enumerate(top_feats):
    binner.plot_woe(feat, ax=axes_flat[i])

# Hide unused axes
for j in range(len(top_feats), len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("WoE Bins — Top 12 Features by IV", fontsize=16, fontweight="bold", y=1.01)
fig.tight_layout()
plt.savefig(DATA_RESULTS_PATH / "woe_top12_features.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: data/results/woe_top12_features.png")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11 — WoE plots for remaining selected features
# ═══════════════════════════════════════════════════════════════════════════════
md("""
### WoE Bin Plots — Remaining Selected Features
""")

code("""
remaining_feats = [f for f in selected_features if f not in top_feats]

if remaining_feats:
    n_rows = (len(remaining_feats) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(20, 5.5 * n_rows))
    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and len(remaining_feats) == 1 else axes.flatten()

    for i, feat in enumerate(remaining_feats):
        binner.plot_woe(feat, ax=axes_flat[i])

    for j in range(len(remaining_feats), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("WoE Bins — Remaining Selected Features", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    plt.savefig(DATA_RESULTS_PATH / "woe_remaining_features.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: data/results/woe_remaining_features.png")
else:
    print("No remaining features to plot.")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12 — Detailed bin tables
# ═══════════════════════════════════════════════════════════════════════════════
md("""
## Step 9: Detailed Bin Tables — Top Features

These tables show per-bin statistics: count, event count, non-event count,
event rate, WoE, and IV contribution.  This is the core WoE/IV deliverable.
""")

code("""
print("=" * 100)
print("DETAILED BIN TABLES — Top 10 Features by IV")
print("=" * 100)

for feat in iv_df.loc[iv_df["feature"].isin(selected_features)].head(10)["feature"]:
    iv_val = binner.iv_[feat]
    mono = binner.check_monotonicity(feat)
    print(f"\\n{'─' * 100}")
    print(f"Feature: {feat}  |  IV: {iv_val:.4f}  |  Monotonic: {mono['monotonic']}  ({mono['direction']})")
    print(f"{'─' * 100}")
    table = binner.bin_table(feat)
    display_cols = ["bin", "count", "event_count", "non_event_count", "event_rate", "woe", "iv"]
    display_cols = [c for c in display_cols if c in table.columns]
    print(table[display_cols].to_string(index=False))
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13 — Transform all splits
# ═══════════════════════════════════════════════════════════════════════════════
md("""
## Step 10: WoE Transformation

Transform the training, validation, and test sets using the fitted WoE bins.
**Critical:** The binner was fit only on training data — applying it to val/test
prevents information leakage.
""")

code("""
train_woe = binner.transform(train)
val_woe   = binner.transform(val)
test_woe  = binner.transform(test)

# Add target column
train_woe[TARGET_COL] = y_train.values
val_woe[TARGET_COL]   = y_val.values
test_woe[TARGET_COL]  = y_test.values

# Keep only selected features + target
keep_cols = [f for f in selected_features if f in train_woe.columns] + [TARGET_COL]
train_woe = train_woe[keep_cols]
val_woe   = val_woe[keep_cols]
test_woe  = test_woe[keep_cols]

print(f"WoE-transformed shapes:")
print(f"  Train: {train_woe.shape}")
print(f"  Val:   {val_woe.shape}")
print(f"  Test:  {test_woe.shape}")
print(f"  Features: {len(keep_cols) - 1} (+ target)")
print()

# Quick sanity check — WoE values should be finite
for name, df in [("train", train_woe), ("val", val_woe), ("test", test_woe)]:
    n_inf = np.isinf(df.select_dtypes(include=[np.number]).values).sum()
    n_nan = df.isna().sum().sum()
    print(f"  {name}: inf={n_inf}, NaN={n_nan}")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 14 — WoE correlation heatmap
# ═══════════════════════════════════════════════════════════════════════════════
md("""
### WoE Feature Correlation Matrix

High pairwise correlation between WoE features may cause multicollinearity
in the logistic regression scorecard.  Notebook 03 will apply a |ρ| < 0.70
threshold for final feature selection.
""")

code("""
woe_feats = [c for c in train_woe.columns if c != TARGET_COL]
corr_matrix = train_woe[woe_feats].corr()

fig, ax = plt.subplots(figsize=(18, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=False, cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
)
ax.set_title("WoE Feature Correlation Matrix", fontsize=14, fontweight="bold")
fig.tight_layout()
plt.savefig(DATA_RESULTS_PATH / "woe_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# Flag highly correlated pairs
high_corr = []
for i in range(len(woe_feats)):
    for j in range(i + 1, len(woe_feats)):
        r = abs(corr_matrix.iloc[i, j])
        if r >= 0.70:
            high_corr.append((woe_feats[i], woe_feats[j], round(r, 3)))

if high_corr:
    print(f"\\nHighly correlated pairs (|r| >= 0.70): {len(high_corr)}")
    for f1, f2, r in sorted(high_corr, key=lambda x: -x[2]):
        print(f"  {f1:<35s} × {f2:<35s}  |r| = {r:.3f}")
    print("\\n→ These will be addressed by pairwise correlation filter in Notebook 03.")
else:
    print("\\nNo highly correlated pairs (|r| >= 0.70) — multicollinearity is low.")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 15 — Save outputs
# ═══════════════════════════════════════════════════════════════════════════════
md("""
## Step 11: Save All Outputs

| Output | Description |
|--------|-------------|
| `woe_binning_results.pkl` | Fitted WOEBinner object (contains all bin edges, WoE mappings, IV) |
| `iv_summary.csv` | Feature × IV × selection status table |
| `train_woe.parquet` | WoE-transformed training data (selected features + target) |
| `val_woe.parquet` | WoE-transformed validation data |
| `test_woe.parquet` | WoE-transformed test data |
""")

code("""
# ── Save fitted binner ──
binner_path = DATA_PROCESSED_PATH / "woe_binning_results.pkl"
with open(binner_path, "wb") as f:
    pickle.dump(binner, f)
print(f"Saved: {binner_path}")

# ── Save IV summary ──
iv_path = DATA_PROCESSED_PATH / "iv_summary.csv"
iv_df.to_csv(iv_path, index=False)
print(f"Saved: {iv_path}")

# ── Save WoE-transformed data ──
train_woe.to_parquet(DATA_PROCESSED_PATH / "train_woe.parquet", index=False)
val_woe.to_parquet(DATA_PROCESSED_PATH / "val_woe.parquet", index=False)
test_woe.to_parquet(DATA_PROCESSED_PATH / "test_woe.parquet", index=False)
print(f"Saved: train_woe.parquet ({train_woe.shape})")
print(f"Saved: val_woe.parquet   ({val_woe.shape})")
print(f"Saved: test_woe.parquet  ({test_woe.shape})")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 16 — Final summary
# ═══════════════════════════════════════════════════════════════════════════════
md("""
## Summary & Next Steps
""")

code("""
print("=" * 80)
print("NOTEBOOK 02 SUMMARY — WoE/IV Feature Engineering")
print("=" * 80)
print()
print(f"  Total candidate features evaluated:  {len(candidate_features)}")
print(f"  Successfully binned:                 {len(binner.binners_)}")
print(f"  Selected (IV >= 0.02):               {len(selected_features)}")
print(f"  Dropped  (IV <  0.02):               {len(dropped_features)}")
print()

# Selection breakdown
print("Selection breakdown:")
for status in ["strong", "medium", "weak"]:
    count = len(iv_df[iv_df["selection_status"] == status])
    feats = iv_df.loc[iv_df["selection_status"] == status, "feature"].tolist()
    print(f"  {status:<12s}: {count:3d}  {feats[:5]}{'...' if len(feats) > 5 else ''}")

print()
mono_ok = binner.monotonicity_report()
mono_ok = mono_ok[mono_ok["feature"].isin(selected_features)]
n_mono = mono_ok["monotonic"].sum()
print(f"  Monotonicity: {n_mono}/{len(mono_ok)} selected features are monotonic")
print()
print("Files saved:")
print("  data/processed/woe_binning_results.pkl")
print("  data/processed/iv_summary.csv")
print("  data/processed/train_woe.parquet")
print("  data/processed/val_woe.parquet")
print("  data/processed/test_woe.parquet")
print("  data/results/iv_top_features.png")
print("  data/results/woe_top12_features.png")
print("  data/results/woe_correlation_matrix.png")
print()
print("→ Next: Notebook 03 — PD Scorecard with Grade")
print("  Will apply pairwise correlation filter (|r| < 0.70),")
print("  verify all WoE coefficients are negative,")
print("  and build L2-regularized logistic regression scorecard.")
""")

# ═══════════════════════════════════════════════════════════════════════════════
# Write notebook
# ═══════════════════════════════════════════════════════════════════════════════
out_path = Path(__file__).resolve().parent.parent / "notebooks" / "02_WOE_IV_Feature_Engineering.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)

print(f"Notebook written to {out_path}")
print(f"Total cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type == 'markdown')} markdown, {sum(1 for c in nb.cells if c.cell_type == 'code')} code)")
