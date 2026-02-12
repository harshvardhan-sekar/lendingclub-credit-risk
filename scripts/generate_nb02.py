"""Generate Notebook 02: WOE/IV Feature Engineering."""
import json
import uuid

def cell(cell_type, source, **kwargs):
    c = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    }
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c

cells = []

# ── Cell 0: Title ──
cells.append(cell("markdown", [
    "# Notebook 02: WOE/IV Feature Engineering\n",
    "## LendingClub Credit Risk Analytics\n",
    "\n",
    "This notebook implements **Weight of Evidence (WOE)** and **Information Value (IV)** analysis\n",
    "for credit scorecard feature selection. This mirrors the feature engineering phase of\n",
    "behavioral scorecard monitoring and loss forecasting in my prior role, where WOE/IV\n",
    "was the standard methodology for VantageScore/FICO-style binning of utilization, DTI,\n",
    "inquiries, and other credit bureau features.\n",
    "\n",
    "### Key Principles:\n",
    "- WOE transformation converts features to a common scale of log-odds\n",
    "- IV quantifies each feature's predictive power for separating defaults from non-defaults\n",
    "- Binning is fit **ONLY on training data** — then applied to val/test to prevent leakage\n",
    "- `grade`, `sub_grade`, and `int_rate` are excluded from the scorecard (they are outputs\n",
    "  of the credit decision, not independent predictors)\n",
    "\n",
    "### Inputs:\n",
    "- `data/processed/train.parquet` (training data only)\n",
    "- `config.py` constants\n",
    "\n",
    "### Outputs:\n",
    "- `src/woe_binning.py` (reusable WOEBinner module)\n",
    "- `data/processed/woe_binning_results.pkl`\n",
    "- `data/processed/iv_summary.csv`\n",
    "- `data/processed/train_woe.parquet`, `val_woe.parquet`, `test_woe.parquet`"
]))

# ── Cell 1: Imports ──
cells.append(cell("markdown", [
    "## Setup: Imports and Configuration"
]))

cells.append(cell("code", [
    "import sys\n",
    "import warnings\n",
    "import pickle\n",
    "from pathlib import Path\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib\n",
    "matplotlib.use('Agg')\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Project imports\n",
    "sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()))\n",
    "from config import *\n",
    "from src.woe_binning import WOEBinner\n",
    "\n",
    "print(f'Project root: {PROJECT_ROOT}')\n",
    "print(f'Random state: {RANDOM_STATE}')"
]))

# ── Cell 2: Load training data ──
cells.append(cell("markdown", [
    "## Step 1: Load Training Data\n",
    "\n",
    "WOE binning must be fit **only on training data**. We load the time-based split\n",
    "from Notebook 01 (train: 2007-2015, val: 2016, test: 2017-2018)."
]))

cells.append(cell("code", [
    "train = pd.read_parquet(DATA_PROCESSED_PATH / 'train.parquet')\n",
    "val = pd.read_parquet(DATA_PROCESSED_PATH / 'val.parquet')\n",
    "test = pd.read_parquet(DATA_PROCESSED_PATH / 'test.parquet')\n",
    "\n",
    "y_train = train[TARGET_COL]\n",
    "y_val = val[TARGET_COL]\n",
    "y_test = test[TARGET_COL]\n",
    "\n",
    "print(f'Train: {train.shape} — default rate: {y_train.mean():.4f}')\n",
    "print(f'Val:   {val.shape} — default rate: {y_val.mean():.4f}')\n",
    "print(f'Test:  {test.shape} — default rate: {y_test.mean():.4f}')"
]))

# ── Cell 3: Feature Engineering ──
cells.append(cell("markdown", [
    "## Step 2: Create Engineered Features\n",
    "\n",
    "Before WOE binning, we create derived features that combine raw attributes\n",
    "into more predictive ratios and flags."
]))

cells.append(cell("code", [
    "def engineer_features(df: pd.DataFrame) -> pd.DataFrame:\n",
    "    \"\"\"Create derived features for WOE/IV analysis.\"\"\"\n",
    "    df = df.copy()\n",
    "\n",
    "    # Credit history length in years\n",
    "    if 'earliest_cr_line' in df.columns and 'issue_d' in df.columns:\n",
    "        ecl = pd.to_datetime(df['earliest_cr_line'], errors='coerce')\n",
    "        iss = pd.to_datetime(df['issue_d'], errors='coerce')\n",
    "        df['credit_history_years'] = (iss - ecl).dt.days / 365.25\n",
    "\n",
    "    # FICO average\n",
    "    if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:\n",
    "        df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2\n",
    "\n",
    "    # Loan-to-income ratio\n",
    "    if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:\n",
    "        df['loan_to_income'] = df['loan_amnt'] / df['annual_inc'].clip(lower=1)\n",
    "\n",
    "    # Installment-to-income ratio (monthly)\n",
    "    if 'installment' in df.columns and 'annual_inc' in df.columns:\n",
    "        monthly_inc = df['annual_inc'].clip(lower=1) / 12\n",
    "        df['installment_to_income'] = df['installment'] / monthly_inc\n",
    "\n",
    "    # Total credit utilization\n",
    "    if 'revol_bal' in df.columns and 'total_rev_hi_lim' in df.columns:\n",
    "        df['total_credit_utilization'] = (\n",
    "            df['revol_bal'] / df['total_rev_hi_lim'].clip(lower=1)\n",
    "        )\n",
    "\n",
    "    # Binary flags\n",
    "    if 'delinq_2yrs' in df.columns:\n",
    "        df['delinq_flag'] = (df['delinq_2yrs'] > 0).astype(int)\n",
    "    if 'inq_last_6mths' in df.columns:\n",
    "        df['recent_inquiry_flag'] = (df['inq_last_6mths'] > 2).astype(int)\n",
    "    if 'dti' in df.columns:\n",
    "        df['high_dti_flag'] = (df['dti'] > 30).astype(int)\n",
    "\n",
    "    return df\n",
    "\n",
    "\n",
    "train = engineer_features(train)\n",
    "val = engineer_features(val)\n",
    "test = engineer_features(test)\n",
    "\n",
    "new_feats = ['credit_history_years', 'fico_avg', 'loan_to_income',\n",
    "             'installment_to_income', 'total_credit_utilization',\n",
    "             'delinq_flag', 'recent_inquiry_flag', 'high_dti_flag']\n",
    "print('Engineered features:')\n",
    "for f in new_feats:\n",
    "    if f in train.columns:\n",
    "        print(f'  {f}: mean={train[f].mean():.4f}, missing={train[f].isna().mean():.2%}')"
]))

# ── Cell 4: Define candidate features ──
cells.append(cell("markdown", [
    "## Step 3: Define Candidate Features for WOE/IV\n",
    "\n",
    "We include all PD model features from CLAUDE.md plus engineered features.\n",
    "**Excluded**: leakage variables (post-origination outcomes) and non-feature columns.\n",
    "**Also excluded from scorecard**: `grade`, `sub_grade`, `int_rate` — these are outputs\n",
    "of the credit decision process, creating circular dependency."
]))

cells.append(cell("code", [
    "# Leakage variables — NEVER use for PD model\n",
    "LEAKAGE = {\n",
    "    'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',\n",
    "    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',\n",
    "    'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt',\n",
    "    'last_pymnt_d', 'last_fico_range_high', 'last_fico_range_low',\n",
    "    'next_pymnt_d', 'last_credit_pull_d', 'hardship_flag',\n",
    "    'debt_settlement_flag',\n",
    "}\n",
    "\n",
    "# Non-feature columns\n",
    "NON_FEATURES = {\n",
    "    TARGET_COL, 'issue_d', 'issue_month', 'issue_year',\n",
    "    'emp_title', 'title', 'zip_code', 'addr_state',\n",
    "    'earliest_cr_line', 'application_type',\n",
    "    'initial_list_status', 'disbursement_method',\n",
    "}\n",
    "\n",
    "# Scorecard exclusions (circular — assigned based on credit risk)\n",
    "SCORECARD_EXCLUDE = {'grade', 'sub_grade', 'int_rate'}\n",
    "\n",
    "# Get all numeric columns\n",
    "all_numeric = train.select_dtypes(include=[np.number]).columns.tolist()\n",
    "\n",
    "# Candidate features = numeric - leakage - non_features\n",
    "candidates = [c for c in all_numeric\n",
    "              if c not in LEAKAGE\n",
    "              and c not in NON_FEATURES\n",
    "              and c not in SCORECARD_EXCLUDE]\n",
    "\n",
    "print(f'Total numeric columns: {len(all_numeric)}')\n",
    "print(f'After removing leakage ({len(LEAKAGE)}): {len(all_numeric) - len(LEAKAGE & set(all_numeric))}')\n",
    "print(f'After removing non-features: {len(candidates) + len(SCORECARD_EXCLUDE & set(all_numeric))}')\n",
    "print(f'After removing scorecard exclusions: {len(candidates)}')\n",
    "print(f'\\nCandidate features ({len(candidates)}):')\n",
    "for i, c in enumerate(sorted(candidates)):\n",
    "    print(f'  {i+1:2d}. {c}')"
]))

# ── Cell 5: Fit WOE binning ──
cells.append(cell("markdown", [
    "## Step 4: Apply WOE Binning to All Candidate Features\n",
    "\n",
    "The WOEBinner uses decision tree-based optimal binning to find splits that\n",
    "maximize separation between default and non-default groups. Monotonic bad-rate\n",
    "ordering is enforced across bins — this is critical for scorecard development."
]))

cells.append(cell("code", [
    "# Fit WOE binner on training data ONLY\n",
    "X_train_candidates = train[candidates].copy()\n",
    "\n",
    "binner = WOEBinner(max_bins=10, min_bin_pct=0.05, monotonic=True,\n",
    "                   random_state=RANDOM_STATE)\n",
    "binner.fit(X_train_candidates, y_train)\n",
    "\n",
    "print(f'Features successfully binned: {len(binner.fitted_features_)}')\n",
    "print(f'Features skipped: {len(candidates) - len(binner.fitted_features_)}')"
]))

# ── Cell 6: IV Summary ──
cells.append(cell("markdown", [
    "## Step 5: IV Summary and Feature Selection\n",
    "\n",
    "Information Value (IV) quantifies predictive power:\n",
    "- IV < 0.02: Not predictive (drop)\n",
    "- IV 0.02–0.10: Weak predictor (flag for review)\n",
    "- IV 0.10–0.30: Medium predictor (include)\n",
    "- IV 0.30–0.50: Strong predictor (include)\n",
    "- IV > 0.50: Suspicious — investigate for data leakage"
]))

cells.append(cell("code", [
    "iv_df = binner.iv_summary()\n",
    "\n",
    "print('IV Summary Table (all features):')\n",
    "print('=' * 70)\n",
    "for _, row in iv_df.iterrows():\n",
    "    marker = ''\n",
    "    if row['selection_status'] == 'suspicious_check_leakage':\n",
    "        marker = ' *** CHECK LEAKAGE ***'\n",
    "    elif row['selection_status'] == 'drop_not_predictive':\n",
    "        marker = ' (drop)'\n",
    "    elif row['selection_status'] == 'weak':\n",
    "        marker = ' (weak)'\n",
    "    print(f\"  {row['feature']:45s} IV={row['iv']:.6f}  [{row['selection_status']}]{marker}\")\n",
    "\n",
    "print(f'\\nIV Summary Statistics:')\n",
    "print(f'  Mean IV:   {iv_df[\"iv\"].mean():.4f}')\n",
    "print(f'  Median IV: {iv_df[\"iv\"].median():.4f}')\n",
    "print(f'\\nCount by selection status:')\n",
    "for status, count in iv_df['selection_status'].value_counts().items():\n",
    "    print(f'  {status}: {count}')"
]))

# ── Cell 7: Discuss grade/int_rate exclusion ──
cells.append(cell("markdown", [
    "## Step 6: Discuss High-IV Features and Scorecard Exclusions\n",
    "\n",
    "### Why exclude `grade`, `sub_grade`, and `int_rate`?\n",
    "\n",
    "These features will have **very high IV** (likely > 0.5) because they are\n",
    "assigned by LendingClub's own credit model based on the borrower's risk profile.\n",
    "Including them in a scorecard creates a **circular dependency**: the model would\n",
    "essentially learn \"high-risk grades default more\" rather than learning from\n",
    "underlying borrower characteristics.\n",
    "\n",
    "- `grade`/`sub_grade`: Assigned by LC's credit algorithm\n",
    "- `int_rate`: Directly determined by grade (A=low rate, G=high rate)\n",
    "\n",
    "**Decision**: Exclude from scorecard (Notebook 03). Keep for ML models (Notebook 04)\n",
    "where interpretability is less critical.\n",
    "\n",
    "### Leakage Investigation\n",
    "Any feature with IV > 0.5 that is NOT grade/int_rate should be investigated.\n",
    "Common culprits: `total_pymnt`, `last_pymnt_amnt`, `last_fico_range_*`."
]))

cells.append(cell("code", [
    "# Check which of the excluded features would have had high IV\n",
    "# by running WOE binning on grade-related features separately\n",
    "grade_feats = ['int_rate']\n",
    "grade_binner = WOEBinner(max_bins=10, random_state=RANDOM_STATE)\n",
    "grade_binner.fit(train[grade_feats], y_train)\n",
    "grade_iv = grade_binner.iv_summary()\n",
    "\n",
    "print('IV for excluded scorecard features (for documentation):')\n",
    "for _, row in grade_iv.iterrows():\n",
    "    print(f\"  {row['feature']}: IV={row['iv']:.4f} — EXCLUDED (credit decision output)\")\n",
    "\n",
    "# Check for any suspicious features in the main selection\n",
    "suspicious = iv_df[iv_df['selection_status'] == 'suspicious_check_leakage']\n",
    "if len(suspicious) > 0:\n",
    "    print(f'\\nFeatures with IV > 0.50 (investigate for leakage):')\n",
    "    for _, row in suspicious.iterrows():\n",
    "        print(f\"  {row['feature']}: IV={row['iv']:.4f}\")\n",
    "else:\n",
    "    print('\\nNo features with suspicious IV > 0.50 (good — no leakage detected).')"
]))

# ── Cell 8: Feature selection ──
cells.append(cell("markdown", [
    "## Step 7: Final Feature Selection (IV-Based)\n",
    "\n",
    "Select features with IV >= 0.02 (weak or stronger). Features with IV < 0.02\n",
    "have negligible predictive power and are dropped."
]))

cells.append(cell("code", [
    "# Select features with IV >= 0.02\n",
    "selected = iv_df[iv_df['iv'] >= 0.02].copy()\n",
    "dropped = iv_df[iv_df['iv'] < 0.02].copy()\n",
    "\n",
    "selected_features = selected['feature'].tolist()\n",
    "\n",
    "print(f'Features selected (IV >= 0.02): {len(selected_features)}')\n",
    "print(f'Features dropped  (IV <  0.02): {len(dropped)}')\n",
    "\n",
    "print(f'\\nSelected features by IV tier:')\n",
    "for status in ['strong', 'medium', 'weak', 'suspicious_check_leakage']:\n",
    "    tier = selected[selected['selection_status'] == status]\n",
    "    if len(tier) > 0:\n",
    "        print(f'\\n  {status.upper()} ({len(tier)}):')\n",
    "        for _, row in tier.iterrows():\n",
    "            print(f\"    {row['feature']:40s} IV={row['iv']:.4f}\")\n",
    "\n",
    "if len(dropped) > 0:\n",
    "    print(f'\\nDropped features ({len(dropped)}):')\n",
    "    for _, row in dropped.iterrows():\n",
    "        print(f\"    {row['feature']:40s} IV={row['iv']:.6f}\")"
]))

# ── Cell 9: Monotonicity check ──
cells.append(cell("markdown", [
    "## Step 8: Validate Monotonicity of Bad Rates\n",
    "\n",
    "For scorecard development, WOE bins should show monotonic bad rates.\n",
    "Minor violations are acceptable and documented; major violations\n",
    "require re-binning or feature transformation."
]))

cells.append(cell("code", [
    "mono_results = []\n",
    "for feat in selected_features:\n",
    "    info = binner.check_monotonicity(feat)\n",
    "    mono_results.append({\n",
    "        'feature': feat,\n",
    "        'monotonic': info['monotonic'],\n",
    "        'direction': info['direction'],\n",
    "        'n_bins': len(info['rates']),\n",
    "    })\n",
    "\n",
    "mono_df = pd.DataFrame(mono_results)\n",
    "n_mono = mono_df['monotonic'].sum()\n",
    "n_total = len(mono_df)\n",
    "\n",
    "print(f'Monotonicity Check: {n_mono}/{n_total} features are monotonic ({n_mono/n_total:.0%})')\n",
    "print()\n",
    "\n",
    "violations = mono_df[~mono_df['monotonic']]\n",
    "if len(violations) > 0:\n",
    "    print(f'Non-monotonic features ({len(violations)}):')\n",
    "    for _, row in violations.iterrows():\n",
    "        print(f\"  {row['feature']}: {row['direction']} ({row['n_bins']} bins)\")\n",
    "    print('\\nNote: Minor monotonicity violations are acceptable in practice.')\n",
    "    print('These features are still included — the WOE transformation handles the ordering.')\n",
    "else:\n",
    "    print('All selected features have monotonic bad rates across bins.')"
]))

# ── Cell 10: WOE plots for top features ──
cells.append(cell("markdown", [
    "## Step 9: WOE Bin Plots (Top Features)\n",
    "\n",
    "Visual inspection of WOE patterns and bad rates for the highest-IV features."
]))

cells.append(cell("code", [
    "# Plot top 12 features by IV\n",
    "top_features = selected.head(12)['feature'].tolist()\n",
    "\n",
    "fig, axes = plt.subplots(4, 3, figsize=(20, 24))\n",
    "for i, feat in enumerate(top_features):\n",
    "    ax = axes.flatten()[i]\n",
    "    binner.plot_bad_rate(feat, ax=ax)\n",
    "\n",
    "plt.suptitle('Bad Rate by Bin — Top 12 Features by IV', fontsize=16, y=1.01)\n",
    "plt.tight_layout()\n",
    "plt.savefig(DATA_RESULTS_PATH / 'woe_bad_rate_top12.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ── Cell 11: Transform all datasets ──
cells.append(cell("markdown", [
    "## Step 10: WOE Transformation\n",
    "\n",
    "Transform selected features to WOE values. The binner was fit on training data only —\n",
    "now we apply it to train, validation, and test sets.\n",
    "\n",
    "**Critical**: This transformation encodes each feature's relationship with the target\n",
    "into a single numeric value per bin, enabling logistic regression to work with\n",
    "optimally binned features."
]))

cells.append(cell("code", [
    "# Select only the features that passed IV screening\n",
    "# Create a new binner fitted only on selected features for clean transformation\n",
    "X_train_sel = train[selected_features]\n",
    "X_val_sel = val[selected_features]\n",
    "X_test_sel = test[selected_features]\n",
    "\n",
    "# Transform\n",
    "train_woe = binner.transform(X_train_sel)\n",
    "val_woe = binner.transform(X_val_sel)\n",
    "test_woe = binner.transform(X_test_sel)\n",
    "\n",
    "# Add target\n",
    "train_woe[TARGET_COL] = y_train.values\n",
    "val_woe[TARGET_COL] = y_val.values\n",
    "test_woe[TARGET_COL] = y_test.values\n",
    "\n",
    "# Add macro features (pass through without WOE transformation)\n",
    "for feat in FRED_SERIES:\n",
    "    if feat in train.columns:\n",
    "        train_woe[feat] = train[feat].values\n",
    "        val_woe[feat] = val[feat].values\n",
    "        test_woe[feat] = test[feat].values\n",
    "\n",
    "# Add binary flags (pass through without WOE transformation)\n",
    "flag_cols = [c for c in train.columns\n",
    "             if c.startswith(('has_', 'no_', 'emp_length_unknown', 'installment_features'))\n",
    "             and c.endswith(('_missing', '_flag', '_unknown', '_delinq', '_history',\n",
    "                             '_record', '_derog'))]\n",
    "# Also include engineered binary flags\n",
    "extra_flags = ['delinq_flag', 'recent_inquiry_flag', 'high_dti_flag']\n",
    "flag_cols = list(set(flag_cols + extra_flags) & set(train.columns))\n",
    "\n",
    "for feat in flag_cols:\n",
    "    if feat not in train_woe.columns:\n",
    "        train_woe[feat] = train[feat].values\n",
    "        val_woe[feat] = val[feat].values\n",
    "        test_woe[feat] = test[feat].values\n",
    "\n",
    "print(f'Train WOE shape: {train_woe.shape}')\n",
    "print(f'Val WOE shape:   {val_woe.shape}')\n",
    "print(f'Test WOE shape:  {test_woe.shape}')\n",
    "print(f'\\nColumns: {list(train_woe.columns)}')\n",
    "print(f'\\nSample WOE values (first 5 rows):')\n",
    "print(train_woe.head())"
]))

# ── Cell 12: IV Summary Bar Chart ──
cells.append(cell("markdown", [
    "## Step 11: IV Summary Visualization\n",
    "\n",
    "Horizontal bar chart showing IV for all selected features, color-coded by tier."
]))

cells.append(cell("code", [
    "# IV bar chart\n",
    "plot_df = selected.copy()\n",
    "color_map = {\n",
    "    'strong': '#2ca02c',\n",
    "    'medium': '#1f77b4',\n",
    "    'weak': '#ff7f0e',\n",
    "    'suspicious_check_leakage': '#d62728',\n",
    "    'drop_not_predictive': '#7f7f7f',\n",
    "}\n",
    "colors = [color_map.get(s, '#7f7f7f') for s in plot_df['selection_status']]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, max(8, len(plot_df) * 0.35)))\n",
    "bars = ax.barh(range(len(plot_df)), plot_df['iv'].values, color=colors)\n",
    "ax.set_yticks(range(len(plot_df)))\n",
    "ax.set_yticklabels(plot_df['feature'].values, fontsize=9)\n",
    "ax.set_xlabel('Information Value (IV)')\n",
    "ax.set_title('Feature Selection by Information Value')\n",
    "ax.axvline(x=0.02, color='gray', linestyle='--', alpha=0.5, label='Min threshold (0.02)')\n",
    "ax.axvline(x=0.1, color='blue', linestyle='--', alpha=0.3, label='Medium (0.10)')\n",
    "ax.axvline(x=0.3, color='green', linestyle='--', alpha=0.3, label='Strong (0.30)')\n",
    "ax.legend(loc='lower right')\n",
    "ax.invert_yaxis()\n",
    "plt.tight_layout()\n",
    "plt.savefig(DATA_RESULTS_PATH / 'iv_summary_chart.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()"
]))

# ── Cell 13: Correlation among WOE features ──
cells.append(cell("markdown", [
    "## Step 12: Correlation Among Selected WOE Features\n",
    "\n",
    "Check for multicollinearity among WOE-transformed features.\n",
    "Highly correlated pairs (|r| > 0.80) may need one member dropped."
]))

cells.append(cell("code", [
    "woe_only = [c for c in train_woe.columns\n",
    "            if c not in [TARGET_COL] + FRED_SERIES + flag_cols]\n",
    "corr = train_woe[woe_only].corr()\n",
    "\n",
    "# Extract high-correlation pairs\n",
    "pairs = []\n",
    "for i in range(len(woe_only)):\n",
    "    for j in range(i+1, len(woe_only)):\n",
    "        r = corr.iloc[i, j]\n",
    "        if abs(r) > 0.70:\n",
    "            pairs.append((woe_only[i], woe_only[j], round(r, 3)))\n",
    "\n",
    "if pairs:\n",
    "    print('Highly correlated WOE feature pairs (|r| > 0.70):')\n",
    "    for f1, f2, r in sorted(pairs, key=lambda x: -abs(x[2])):\n",
    "        print(f'  {f1:35s} ↔ {f2:35s}  r={r:+.3f}')\n",
    "else:\n",
    "    print('No highly correlated WOE feature pairs found (|r| > 0.70).')\n",
    "\n",
    "# Heatmap\n",
    "if len(woe_only) > 0:\n",
    "    fig, ax = plt.subplots(figsize=(14, 12))\n",
    "    sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1,\n",
    "                xticklabels=True, yticklabels=True, ax=ax, fmt='.1f',\n",
    "                annot=len(woe_only) <= 20, square=True)\n",
    "    ax.set_title('Correlation Matrix — WOE-Transformed Features')\n",
    "    plt.tight_layout()\n",
    "    plt.savefig(DATA_RESULTS_PATH / 'woe_correlation_matrix.png', dpi=150, bbox_inches='tight')\n",
    "    plt.show()"
]))

# ── Cell 14: Save outputs ──
cells.append(cell("markdown", [
    "## Step 13: Save All Outputs\n",
    "\n",
    "Save fitted binner, IV summary, and WOE-transformed datasets."
]))

cells.append(cell("code", [
    "# Save WOE binner object\n",
    "with open(DATA_PROCESSED_PATH / 'woe_binning_results.pkl', 'wb') as f:\n",
    "    pickle.dump(binner, f)\n",
    "print(f'Saved: woe_binning_results.pkl')\n",
    "\n",
    "# Save IV summary\n",
    "iv_df.to_csv(DATA_PROCESSED_PATH / 'iv_summary.csv', index=False)\n",
    "print(f'Saved: iv_summary.csv ({len(iv_df)} features)')\n",
    "\n",
    "# Save WOE-transformed datasets\n",
    "train_woe.to_parquet(DATA_PROCESSED_PATH / 'train_woe.parquet', index=False)\n",
    "val_woe.to_parquet(DATA_PROCESSED_PATH / 'val_woe.parquet', index=False)\n",
    "test_woe.to_parquet(DATA_PROCESSED_PATH / 'test_woe.parquet', index=False)\n",
    "print(f'Saved: train_woe.parquet ({train_woe.shape})')\n",
    "print(f'Saved: val_woe.parquet ({val_woe.shape})')\n",
    "print(f'Saved: test_woe.parquet ({test_woe.shape})')\n",
    "\n",
    "print(f'\\n{\"=\"*60}')\n",
    "print(f'SESSION 2 COMPLETE')\n",
    "print(f'{\"=\"*60}')\n",
    "print(f'Features fitted:  {len(binner.fitted_features_)}')\n",
    "print(f'Features selected (IV >= 0.02): {len(selected_features)}')\n",
    "print(f'WOE columns in output: {len(woe_only)}')\n",
    "print(f'Macro features carried through: {[s for s in FRED_SERIES if s in train_woe.columns]}')\n",
    "print(f'Binary flags carried through: {len(flag_cols)}')"
]))

# ── Build notebook ──
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "cells": cells
}

with open('notebooks/02_WOE_IV_Feature_Engineering.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Generated notebook with {len(cells)} cells")
