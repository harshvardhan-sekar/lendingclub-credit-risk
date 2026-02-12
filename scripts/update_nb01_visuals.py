"""
Update Notebook 01 with:
1. New loan_status documentation cells before Step 4
2. Updated portfolio_composition.png with diverging palette + percentage annotations
3. Updated default_rate_by_grade.png with matching palette
4. Updated distributions_by_default.png with larger figure size
"""
import json
import uuid

NB_PATH = 'notebooks/01_EDA_and_Data_Cleaning.ipynb'

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb['cells']

# ── Helper ──────────────────────────────────────────────────────────────────
def make_cell(cell_type, source_lines):
    return {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source_lines,
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {}),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1) INSERT two cells BEFORE Step 4 (currently at index 9)
# ═══════════════════════════════════════════════════════════════════════════
step4_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'markdown' and '## Step 4' in ''.join(c['source']):
        step4_idx = i
        break
assert step4_idx is not None, "Could not find Step 4 markdown cell"

md_cell = make_cell("markdown", [
    "## Step 3.5: Loan Status Distribution Before Filtering\n",
    "\n",
    "Before filtering to terminal statuses, we document the **full loan_status distribution**\n",
    "grouped by origination year. This captures:\n",
    "\n",
    "- The 9 distinct loan_status values including 2 \"Does not meet the credit policy\" variants\n",
    "- **2007-2010 vintages** contain \"Does not meet the credit policy\" loans — these were\n",
    "  originated under LendingClub's earlier underwriting standards before the platform\n",
    "  tightened its credit policy. They are excluded from modeling but documented here\n",
    "  for completeness.\n",
    "- The ratio of terminal-status loans to total originations per vintage year\n",
    "\n",
    "This is a **documentation-only** step — no data is modified."
])

code_cell = make_cell("code", [
    "# ── Step 3.5: Full loan_status distribution by vintage year ──\n",
    "print('='*80)\n",
    "print('LOAN STATUS DISTRIBUTION BY ORIGINATION YEAR')\n",
    "print('='*80)\n",
    "\n",
    "# Group by issue year and loan_status\n",
    "df['_issue_year'] = df['issue_d'].dt.year\n",
    "status_by_year = df.groupby(['_issue_year', 'loan_status']).size().unstack(fill_value=0)\n",
    "\n",
    "# Total originations vs terminal-status loans per vintage\n",
    "terminal = DEFAULT_STATUSES + NON_DEFAULT_STATUSES\n",
    "status_by_year['_Total'] = status_by_year.sum(axis=1)\n",
    "status_by_year['_Terminal'] = status_by_year[[s for s in terminal if s in status_by_year.columns]].sum(axis=1)\n",
    "status_by_year['_Terminal_Pct'] = (status_by_year['_Terminal'] / status_by_year['_Total'] * 100).round(1)\n",
    "\n",
    "print('\\nOriginations vs Terminal-Status Loans per Vintage:')\n",
    "summary = status_by_year[['_Total', '_Terminal', '_Terminal_Pct']].copy()\n",
    "summary.columns = ['Total Originations', 'Terminal Loans', 'Terminal %']\n",
    "print(summary.to_string())\n",
    "\n",
    "# \"Does not meet credit policy\" variants\n",
    "dnmcp_cols = [c for c in status_by_year.columns if 'credit policy' in str(c).lower()]\n",
    "if dnmcp_cols:\n",
    "    print(f'\\n\"Does not meet the credit policy\" variants ({len(dnmcp_cols)}):')\n",
    "    for col in dnmcp_cols:\n",
    "        total = status_by_year[col].sum()\n",
    "        years_present = status_by_year[status_by_year[col] > 0].index.tolist()\n",
    "        print(f'  {col}')\n",
    "        print(f'    Total count: {total:,}')\n",
    "        print(f'    Present in years: {years_present}')\n",
    "        # Show per-year breakdown\n",
    "        for yr in years_present:\n",
    "            ct = status_by_year.loc[yr, col]\n",
    "            print(f'      {yr}: {ct:,}')\n",
    "else:\n",
    "    print('\\nNo \"Does not meet credit policy\" variants found.')\n",
    "\n",
    "# Full cross-tab display\n",
    "print('\\nFull Status × Year Cross-Tabulation:')\n",
    "display_cols = [c for c in status_by_year.columns if not c.startswith('_')]\n",
    "print(status_by_year[display_cols].to_string())\n",
    "\n",
    "# Clean up temp column\n",
    "df.drop(columns=['_issue_year'], inplace=True)\n",
    "print('\\n(Documentation only — no rows dropped in this step.)')"
])

cells.insert(step4_idx, code_cell)
cells.insert(step4_idx, md_cell)
print(f"Inserted 2 cells at index {step4_idx} (before Step 4)")

# ═══════════════════════════════════════════════════════════════════════════
# 2) Define the shared grade color palette (tab10-based, 7 distinct colors)
# ═══════════════════════════════════════════════════════════════════════════

GRADE_PALETTE_DEF = (
    "# Consistent grade color palette (tab10-derived, visually separable A-G)\n"
    "GRADE_COLORS = {\n"
    "    'A': '#1f77b4',  # blue\n"
    "    'B': '#ff7f0e',  # orange\n"
    "    'C': '#2ca02c',  # green\n"
    "    'D': '#d62728',  # red\n"
    "    'E': '#9467bd',  # purple\n"
    "    'F': '#8c564b',  # brown\n"
    "    'G': '#e377c2',  # pink\n"
    "}\n"
    "grade_color_list = [GRADE_COLORS[g] for g in GRADE_ORDER]\n"
)

# ═══════════════════════════════════════════════════════════════════════════
# 3) UPDATE default_rate_by_grade (Cell 23 → now shifted by +2 due to inserts)
# ═══════════════════════════════════════════════════════════════════════════

grade_cell_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'default_rate_by_grade.png' in ''.join(c['source']):
        grade_cell_idx = i
        break
assert grade_cell_idx is not None, "Could not find default_rate_by_grade cell"

cells[grade_cell_idx]['source'] = [
    GRADE_PALETTE_DEF,
    "\n",
    "grade_stats = df.groupby('grade')['default'].agg(['mean', 'count']).reset_index()\n",
    "grade_stats.columns = ['grade', 'default_rate', 'count']\n",
    "grade_stats = grade_stats.set_index('grade').loc[GRADE_ORDER].reset_index()\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
    "bars = axes[0].bar(grade_stats['grade'], grade_stats['default_rate'] * 100,\n",
    "                   color=grade_color_list)\n",
    "axes[0].set_xlabel('Grade'); axes[0].set_ylabel('Default Rate (%)')\n",
    "axes[0].set_title('Default Rate by Grade')\n",
    "for bar, r in zip(bars, grade_stats['default_rate']):\n",
    "    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,\n",
    "                 f'{r:.1%}', ha='center', fontsize=10)\n",
    "\n",
    "axes[1].bar(grade_stats['grade'], grade_stats['count'], color=grade_color_list)\n",
    "axes[1].set_xlabel('Grade'); axes[1].set_ylabel('Loans')\n",
    "axes[1].set_title('Loan Volume by Grade')\n",
    "axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))\n",
    "plt.tight_layout()\n",
    "plt.savefig(DATA_RESULTS_PATH / 'default_rate_by_grade.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "# Verify monotonicity\n",
    "rates = grade_stats['default_rate'].values\n",
    "mono = all(rates[i] < rates[i+1] for i in range(len(rates)-1))\n",
    "print(f\"\\nGrade monotonicity: {'PASS' if mono else 'FAIL'}\")\n",
    "for _, row in grade_stats.iterrows():\n",
    "    print(f\"  {row['grade']}: {row['default_rate']:.2%} ({row['count']:,})\")\n",
]
cells[grade_cell_idx]['outputs'] = []
cells[grade_cell_idx]['execution_count'] = None
print(f"Updated default_rate_by_grade cell at index {grade_cell_idx}")

# ═══════════════════════════════════════════════════════════════════════════
# 4) UPDATE distributions_by_default (larger figure size: 20x16)
# ═══════════════════════════════════════════════════════════════════════════

dist_cell_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'distributions_by_default.png' in ''.join(c['source']):
        dist_cell_idx = i
        break
assert dist_cell_idx is not None, "Could not find distributions_by_default cell"

cells[dist_cell_idx]['source'] = [
    "kde_feats = ['fico_range_low', 'dti', 'annual_inc', 'int_rate',\n",
    "             'revol_util', 'open_acc', 'total_acc', 'inq_last_6mths']\n",
    "fig, axes = plt.subplots(2, 4, figsize=(20, 16))\n",
    "\n",
    "for i, feat in enumerate(kde_feats):\n",
    "    ax = axes.flatten()[i]\n",
    "    if feat in df.columns:\n",
    "        d0 = df.loc[df['default'] == 0, feat].dropna()\n",
    "        d1 = df.loc[df['default'] == 1, feat].dropna()\n",
    "        lo = min(d0.quantile(0.01), d1.quantile(0.01))\n",
    "        hi = max(d0.quantile(0.99), d1.quantile(0.99))\n",
    "        ax.hist(d0.clip(lo, hi), bins=50, density=True, alpha=0.5,\n",
    "                color='steelblue', label='Non-Default')\n",
    "        ax.hist(d1.clip(lo, hi), bins=50, density=True, alpha=0.5,\n",
    "                color='coral', label='Default')\n",
    "        ax.set_title(feat, fontsize=14)\n",
    "        ax.legend(fontsize=10)\n",
    "        ax.tick_params(labelsize=10)\n",
    "\n",
    "plt.suptitle('Feature Distributions by Default Status', fontsize=18, y=0.98)\n",
    "plt.tight_layout(rect=[0, 0, 1, 0.96])\n",
    "plt.savefig(DATA_RESULTS_PATH / 'distributions_by_default.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
]
cells[dist_cell_idx]['outputs'] = []
cells[dist_cell_idx]['execution_count'] = None
print(f"Updated distributions_by_default cell at index {dist_cell_idx}")

# ═══════════════════════════════════════════════════════════════════════════
# 5) UPDATE portfolio_composition (diverging palette + percentage annotations)
# ═══════════════════════════════════════════════════════════════════════════

comp_cell_idx = None
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'portfolio_composition.png' in ''.join(c['source']):
        comp_cell_idx = i
        break
assert comp_cell_idx is not None, "Could not find portfolio_composition cell"

cells[comp_cell_idx]['source'] = [
    "comp = df.groupby(['issue_year', 'grade']).size().unstack(fill_value=0)\n",
    "comp = comp[GRADE_ORDER]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(14, 7))\n",
    "comp.plot(kind='bar', stacked=True, ax=ax, color=grade_color_list)\n",
    "ax.set_xlabel('Origination Year'); ax.set_ylabel('Loans')\n",
    "ax.set_title('Portfolio Composition by Grade Over Time')\n",
    "ax.legend(title='Grade', bbox_to_anchor=(1.05, 1), loc='upper left')\n",
    "ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))\n",
    "\n",
    "# Add percentage annotations inside each segment\n",
    "comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100\n",
    "for i_bar, year in enumerate(comp.index):\n",
    "    cumulative = 0\n",
    "    for grade in GRADE_ORDER:\n",
    "        val = comp.loc[year, grade]\n",
    "        pct = comp_pct.loc[year, grade]\n",
    "        if pct >= 3:  # Only annotate segments >= 3% for readability\n",
    "            mid = cumulative + val / 2\n",
    "            ax.text(i_bar, mid, f'{pct:.0f}%', ha='center', va='center',\n",
    "                    fontsize=7, fontweight='bold', color='white')\n",
    "        cumulative += val\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig(DATA_RESULTS_PATH / 'portfolio_composition.png', dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
]
cells[comp_cell_idx]['outputs'] = []
cells[comp_cell_idx]['execution_count'] = None
print(f"Updated portfolio_composition cell at index {comp_cell_idx}")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════
nb['cells'] = cells
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"\nSaved notebook with {len(cells)} cells (was 53, now {len(cells)})")
