"""
Forward default flow rate analysis for LendingClub Credit Risk Analytics.

Flow rates track one-directional (worsening) delinquency progression from
synthetic monthly DPD reconstruction. Curing is unobservable in loan-level data.

CRITICAL DATA LIMITATION:
This module computes flow rates from synthetically reconstructed monthly DPD status
derived from loan-level terminal outcomes. In a production environment with monthly
payment tapes, curing rates and two-way transition matrices would be observable.

Flow Rate Definition:
  30+ Flow Rate = 30 DPD balance(t) / Current balance(t-1)
  60+ Flow Rate = 60 DPD balance(t) / 30 DPD balance(t-1)
  ...
  GCO Flow Rate = GCO balance(t) / 180+ DPD balance(t-1)

Flow Through Rate (FTR):
  Product of all intermediate flow rates:
  FTR = (Current→30+) × (30+→60+) × ... × (150+→180+) × (180+→GCO)
  Interpretation: For every $100 in Current, FTR% will eventually charge off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Synthetic Monthly Panel Construction ─────────────────────────────────────

def reconstruct_monthly_panel(
    df: pd.DataFrame,
    snapshot_date: str = "2018-12-31",
) -> pd.DataFrame:
    """Vectorized reconstruction of monthly DPD status from loan-level terminal outcomes.

    This function creates a synthetic monthly panel by back-calculating monthly
    delinquency status from terminal loan outcomes. Uses vectorized operations
    for performance with large datasets (500K+ loans).

    ASSUMPTIONS:
    - Fully Paid loans: Current every month until payoff
    - Charged Off loans: Current until delinquency onset (last_pymnt_d + 1 month),
      then progressive DPD buckets (30→60→90→120→150→180→GCO)
    - Performing balances: scheduled amortization formula
    - Delinquent balances: frozen at last performing balance (approximation)

    LIMITATIONS:
    - Curing is unobservable (loans that cured before final payoff are invisible)
    - Intermediate delinquencies for eventually-performing loans are invisible
    - Balances for delinquent months are approximate (no penalty interest)

    Args:
        df: Loan-level DataFrame with columns:
            - issue_d (datetime): Origination date
            - last_pymnt_d (datetime): Last payment date
            - loan_status (str): Terminal status
            - funded_amnt (float): Loan amount
            - int_rate (float): Interest rate (%)
            - term (int): Loan term in months
            - grade (str): Risk grade
        snapshot_date: Data snapshot cutoff (loans still active beyond this are right-censored).

    Returns:
        DataFrame with monthly panel rows:
            - loan_id, month_date, dpd_bucket, balance, grade, vintage_year
    """
    snapshot = pd.to_datetime(snapshot_date)

    # Prepare loan-level data
    df = df.copy()
    df['loan_id'] = df.index
    df['vintage_year'] = df['issue_d'].dt.year

    # Filter out loans with missing issue_d
    df = df[df['issue_d'].notna()].copy()

    # Compute end_date for each loan vectorized
    def compute_end_date_vectorized(df_input):
        end_dates = pd.Series(index=df_input.index, dtype='datetime64[ns]')

        # Charged Off loans
        charged_off_mask = df_input['loan_status'] == 'Charged Off'
        has_last_pymnt = df_input['last_pymnt_d'].notna()

        charged_off_with_pymnt = charged_off_mask & has_last_pymnt
        charged_off_no_pymnt = charged_off_mask & ~has_last_pymnt

        # Add 7 months using period arithmetic
        end_dates.loc[charged_off_with_pymnt] = (
            (df_input.loc[charged_off_with_pymnt, 'last_pymnt_d'].dt.to_period('M') + 7).dt.to_timestamp()
        )
        end_dates.loc[charged_off_no_pymnt] = (
            (df_input.loc[charged_off_no_pymnt, 'issue_d'].dt.to_period('M') +
             df_input.loc[charged_off_no_pymnt, 'term']).dt.to_timestamp()
        )

        # Fully Paid loans
        fully_paid_mask = df_input['loan_status'] == 'Fully Paid'
        fully_paid_with_pymnt = fully_paid_mask & has_last_pymnt
        fully_paid_no_pymnt = fully_paid_mask & ~has_last_pymnt

        end_dates.loc[fully_paid_with_pymnt] = df_input.loc[fully_paid_with_pymnt, 'last_pymnt_d']
        end_dates.loc[fully_paid_no_pymnt] = (
            (df_input.loc[fully_paid_no_pymnt, 'issue_d'].dt.to_period('M') +
             df_input.loc[fully_paid_no_pymnt, 'term']).dt.to_timestamp()
        )

        # Other statuses (Current, Late, etc.)
        other_mask = ~charged_off_mask & ~fully_paid_mask
        other_with_pymnt = other_mask & has_last_pymnt
        other_no_pymnt = other_mask & ~has_last_pymnt

        end_dates.loc[other_with_pymnt] = df_input.loc[other_with_pymnt, 'last_pymnt_d'].clip(upper=snapshot)
        end_dates.loc[other_no_pymnt] = snapshot

        return end_dates

    df['end_date'] = compute_end_date_vectorized(df)

    # Filter out loans with no valid end_date
    df = df[df['end_date'].notna()].copy()

    # Compute number of months for each loan
    df['n_months'] = (
        (df['end_date'].dt.year - df['issue_d'].dt.year) * 12 +
        (df['end_date'].dt.month - df['issue_d'].dt.month) + 1
    )
    df['n_months'] = df['n_months'].clip(lower=1)

    # Expand to monthly rows: each loan gets n_months rows
    monthly_df = df.loc[df.index.repeat(df['n_months'])].copy()

    # Add month offset (0, 1, 2, ...) for each loan
    monthly_df['month_offset'] = monthly_df.groupby('loan_id').cumcount()

    # Compute month_date vectorized using period arithmetic
    monthly_df['issue_period'] = monthly_df['issue_d'].dt.to_period('M')
    monthly_df['month_period'] = monthly_df['issue_period'] + monthly_df['month_offset']
    monthly_df['month_date'] = monthly_df['month_period'].dt.to_timestamp()

    # Compute months elapsed (for balance calculation)
    monthly_df['months_elapsed'] = monthly_df['month_offset']

    # Compute DPD buckets vectorized
    # For Charged Off loans, compute months since delinquency onset
    is_charged_off = monthly_df['loan_status'] == 'Charged Off'
    has_last_pymnt = monthly_df['last_pymnt_d'].notna()

    # Onset date = last_pymnt_d + 1 month
    monthly_df['onset_period'] = monthly_df['last_pymnt_d'].dt.to_period('M') + 1

    # Compute months since onset (can be negative if before onset)
    monthly_df['months_since_onset'] = (
        monthly_df['month_period'] - monthly_df['onset_period']
    ).apply(lambda x: x.n if pd.notna(x) else -999)

    # Assign DPD buckets using np.select
    conditions = [
        ~(is_charged_off & has_last_pymnt),  # Not charged off or no last payment → Current
        is_charged_off & has_last_pymnt & (monthly_df['months_since_onset'] < 0),  # Before onset → Current
        is_charged_off & has_last_pymnt & (monthly_df['months_since_onset'] == 0),  # 30+
        is_charged_off & has_last_pymnt & (monthly_df['months_since_onset'] == 1),  # 60+
        is_charged_off & has_last_pymnt & (monthly_df['months_since_onset'] == 2),  # 90+
        is_charged_off & has_last_pymnt & (monthly_df['months_since_onset'] == 3),  # 120+
        is_charged_off & has_last_pymnt & (monthly_df['months_since_onset'] == 4),  # 150+
        is_charged_off & has_last_pymnt & (monthly_df['months_since_onset'] == 5),  # 180+
    ]

    choices = ['Current', 'Current', '30+', '60+', '90+', '120+', '150+', '180+']

    monthly_df['dpd_bucket'] = np.select(conditions, choices, default='GCO')

    # Compute balance vectorized (amortization formula)
    r = monthly_df['int_rate'] / 100 / 12  # monthly rate
    n = monthly_df['term']
    t = monthly_df['months_elapsed']
    P = monthly_df['funded_amnt']

    # Handle r > 0 case
    mask_positive_rate = (r > 0) & (t < n)
    numerator = (1 + r) ** n - (1 + r) ** t
    denominator = (1 + r) ** n - 1

    monthly_df['balance'] = 0.0
    monthly_df.loc[mask_positive_rate, 'balance'] = (
        P[mask_positive_rate] * (numerator[mask_positive_rate] / denominator[mask_positive_rate])
    )

    # Handle r == 0 case (linear approximation)
    mask_zero_rate = (r == 0) & (t < n)
    monthly_df.loc[mask_zero_rate, 'balance'] = (
        P[mask_zero_rate] * np.maximum(0, 1 - t[mask_zero_rate] / n[mask_zero_rate])
    )

    # Clip negative balances
    monthly_df['balance'] = monthly_df['balance'].clip(lower=0)

    # Select output columns
    result = monthly_df[['loan_id', 'month_date', 'dpd_bucket', 'balance', 'grade', 'vintage_year']].copy()

    return result


# ── Receivables Tracker ───────────────────────────────────────────────────────

def build_receivables_tracker(
    monthly_panel: pd.DataFrame,
    loan_recoveries: pd.DataFrame | None = None,
    dpd_buckets: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate synthetic monthly panel into receivables tracker format.

    Args:
        monthly_panel: Output from reconstruct_monthly_panel().
        loan_recoveries: Optional DataFrame with columns [loan_id, recovery_amount].
        dpd_buckets: List of DPD bucket names in order.

    Returns:
        DataFrame with columns:
            - month_date, dpd_bucket, balance, count, grade (if available),
              gco_amount, recovery_amount, nco_amount, loss_rate
    """
    if dpd_buckets is None:
        dpd_buckets = ["Current", "30+", "60+", "90+", "120+", "150+", "180+", "GCO"]

    group_cols = ["month_date", "dpd_bucket"]
    if "grade" in monthly_panel.columns:
        group_cols.append("grade")

    tracker = (
        monthly_panel
        .groupby(group_cols, as_index=False)
        .agg(balance=("balance", "sum"), count=("loan_id", "count"))
    )

    # Ensure all buckets are represented
    tracker["dpd_bucket"] = pd.Categorical(
        tracker["dpd_bucket"], categories=dpd_buckets, ordered=True
    )
    tracker = tracker.sort_values(["month_date", "dpd_bucket"])

    # Compute GCO flows (new GCO entries per month)
    panel_sorted = monthly_panel.sort_values(['loan_id', 'month_date']).copy()
    panel_sorted['prev_bucket'] = panel_sorted.groupby('loan_id')['dpd_bucket'].shift(1)

    # New GCO = loans where dpd_bucket is 'GCO' but prev_bucket is not 'GCO'
    new_gco = panel_sorted[(panel_sorted['dpd_bucket'] == 'GCO') & (panel_sorted['prev_bucket'] != 'GCO')]

    gco_group_cols = ['month_date']
    if 'grade' in new_gco.columns:
        gco_group_cols.append('grade')

    gco_by_month = new_gco.groupby(gco_group_cols)['balance'].sum().reset_index()
    gco_by_month = gco_by_month.rename(columns={'balance': 'gco_amount'})

    # Compute recovery amounts (6 months after GCO)
    if loan_recoveries is not None:
        # Find month each loan first enters GCO
        first_gco = panel_sorted[panel_sorted['dpd_bucket'] == 'GCO'].groupby('loan_id')['month_date'].min().reset_index()
        first_gco.columns = ['loan_id', 'gco_month']

        # Recovery arrives 6 months after GCO
        first_gco['recovery_month'] = first_gco['gco_month'] + pd.DateOffset(months=6)

        # Merge with recovery amounts and grade
        recovery_df = first_gco.merge(loan_recoveries, on='loan_id', how='left')
        if 'grade' in panel_sorted.columns:
            grade_map = panel_sorted[['loan_id', 'grade']].drop_duplicates('loan_id')
            recovery_df = recovery_df.merge(grade_map, on='loan_id', how='left')

        recovery_group_cols = ['recovery_month']
        if 'grade' in recovery_df.columns:
            recovery_group_cols.append('grade')

        recovery_by_month = recovery_df.groupby(recovery_group_cols)['recovery_amount'].sum().reset_index()
        recovery_by_month = recovery_by_month.rename(columns={'recovery_month': 'month_date'})
    else:
        recovery_by_month = pd.DataFrame(columns=['month_date', 'recovery_amount'])
        if 'grade' in tracker.columns:
            recovery_by_month['grade'] = []

    # Merge GCO and recovery into tracker (broadcast across all DPD buckets)
    # First, get unique month_date × grade combinations
    merge_cols = ['month_date']
    if 'grade' in tracker.columns:
        merge_cols.append('grade')

    tracker = tracker.merge(gco_by_month, on=merge_cols, how='left')
    tracker = tracker.merge(recovery_by_month, on=merge_cols, how='left')

    # Fill missing values with 0
    tracker['gco_amount'] = tracker['gco_amount'].fillna(0)
    tracker['recovery_amount'] = tracker['recovery_amount'].fillna(0)

    # Compute NCO
    tracker['nco_amount'] = tracker['gco_amount'] - tracker['recovery_amount']

    # Compute loss rate = NCO / total non-GCO balance
    # For each month × grade, compute total balance excluding GCO
    non_gco_balance = tracker[tracker['dpd_bucket'] != 'GCO'].groupby(merge_cols)['balance'].sum().reset_index()
    non_gco_balance = non_gco_balance.rename(columns={'balance': 'total_non_gco_balance'})

    tracker = tracker.merge(non_gco_balance, on=merge_cols, how='left')
    tracker['loss_rate'] = np.where(
        tracker['total_non_gco_balance'] > 0,
        tracker['nco_amount'] / tracker['total_non_gco_balance'],
        0
    )

    # Drop temp column
    tracker = tracker.drop(columns=['total_non_gco_balance'])

    return tracker


# ── Flow Rate Computation ─────────────────────────────────────────────────────

def compute_flow_rates(
    receivables_tracker: pd.DataFrame,
    dpd_buckets: list[str] | None = None,
    min_balance: float = 1000,
) -> pd.DataFrame:
    """Compute forward-only flow rates from receivables tracker.

    Flow Rate = DPD_bucket(t) / Previous_bucket(t-1)

    Args:
        receivables_tracker: Output from build_receivables_tracker().
        dpd_buckets: Ordered list of DPD buckets.
        min_balance: Minimum denominator balance to compute rate (avoid division by tiny balances).

    Returns:
        DataFrame with columns:
            - month_date, flow_rate_30, flow_rate_60, ..., flow_rate_gco
    """
    if dpd_buckets is None:
        dpd_buckets = ["Current", "30+", "60+", "90+", "120+", "150+", "180+", "GCO"]

    # Pivot to wide format: month_date × dpd_bucket
    pivot = receivables_tracker.pivot_table(
        index="month_date",
        columns="dpd_bucket",
        values="balance",
        aggfunc="sum",
        fill_value=0,
    )

    flow_rates = pd.DataFrame(index=pivot.index[1:])  # start from month 2

    for i in range(len(dpd_buckets) - 1):
        from_bucket = dpd_buckets[i]
        to_bucket = dpd_buckets[i + 1]

        if from_bucket in pivot.columns and to_bucket in pivot.columns:
            # Flow rate = to_bucket(t) / from_bucket(t-1)
            numerator = pivot[to_bucket].shift(-1).iloc[:-1].values
            denominator = pivot[from_bucket].iloc[:-1].values

            rate = np.where(denominator > min_balance, numerator / denominator, np.nan)
            flow_rates[f"flow_rate_{to_bucket.replace('+', '').lower()}"] = rate

    flow_rates = flow_rates.reset_index()
    return flow_rates


def compute_flow_through_rate(
    flow_rates: pd.DataFrame,
    method: str = "diagonal",
) -> pd.Series:
    """Compute cumulative Flow Through Rate (FTR).

    FTR = Product of all intermediate flow rates along a cohort's path.
    Interpretation: For every $100 in Current at month t, FTR% will eventually charge off.

    Two methods:
      - "diagonal" (institutional standard): Each successive flow rate is lagged by
        one month, tracing the actual path a cohort would take.
        FTR(t) = flow_rate_30(t) × flow_rate_60(t+1) × flow_rate_90(t+2) × ...
      - "same_month" (point-in-time snapshot): All flow rates from the same month.
        FTR(t) = flow_rate_30(t) × flow_rate_60(t) × flow_rate_90(t) × ...

    Args:
        flow_rates: Output from compute_flow_rates() with columns flow_rate_*.
        method: "diagonal" (default, institutional) or "same_month" (snapshot).

    Returns:
        Series of FTR values indexed by the starting month_date.
    """
    rate_cols = sorted(
        [col for col in flow_rates.columns if col.startswith("flow_rate_")],
        key=lambda c: flow_rates.columns.tolist().index(c),
    )

    if not rate_cols:
        return pd.Series(dtype=float)

    if method == "same_month":
        # Point-in-time: multiply all rates from the same row
        ftr = flow_rates[rate_cols].prod(axis=1, skipna=False)
        ftr.index = flow_rates["month_date"].values
        return ftr

    # Diagonal (institutional standard):
    # For a cohort starting at month t, each transition step k uses the rate
    # from month (t + k). So flow_rate_30 is unshifted, flow_rate_60 is shifted
    # forward by 1, flow_rate_90 by 2, etc.
    n = len(flow_rates)
    ftr_values = np.ones(n)
    ftr_valid = np.ones(n, dtype=bool)

    for step, col in enumerate(rate_cols):
        rates = flow_rates[col].values
        # For cohort starting at row i, use rate from row (i + step)
        for i in range(n):
            j = i + step
            if j < n and np.isfinite(rates[j]):
                ftr_values[i] *= rates[j]
            else:
                # Incomplete diagonal — mark as NaN
                ftr_valid[i] = False

    ftr = pd.Series(
        np.where(ftr_valid, ftr_values, np.nan),
        index=flow_rates["month_date"].values,
    )
    return ftr


# ── Dual-Mode Flow Rate Forecasting ──────────────────────────────────────────

def forecast_flow_rates_extend(
    historical_rates: pd.DataFrame,
    lookback_months: int = 6,
    forecast_months: int = 120,
) -> pd.DataFrame:
    """Extend flow rates using rolling average (Operational mode).

    Args:
        historical_rates: Historical flow rates (output from compute_flow_rates()).
        lookback_months: Number of recent months to average.
        forecast_months: Forecast horizon.

    Returns:
        DataFrame with extended flow rates (flat extrapolation).
    """
    rate_cols = [col for col in historical_rates.columns if col.startswith("flow_rate_")]

    if len(historical_rates) < lookback_months:
        lookback_months = len(historical_rates)

    # Compute rolling average for last lookback_months
    recent = historical_rates.tail(lookback_months)
    avg_rates = recent[rate_cols].mean()

    # Generate future dates
    last_date = historical_rates["month_date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_months,
        freq="MS",
    )

    # Extend flat
    forecast = pd.DataFrame({"month_date": future_dates})
    for col in rate_cols:
        forecast[col] = avg_rates[col]

    # Combine historical + forecast
    extended = pd.concat([historical_rates, forecast], ignore_index=True)
    return extended


def forecast_flow_rates_cecl(
    historical_rates: pd.DataFrame,
    rs_period_months: int = 24,
    reversion_months: int = 12,
    forecast_months: int = 120,
    macro_adjustment: dict[str, float] | None = None,
) -> pd.DataFrame:
    """CECL-compliant flow rate forecast with R&S period + reversion.

    Phase 1 (R&S): Macro-adjusted rates for rs_period_months.
    Phase 2 (Reversion): Straight-line transition to long-run historical avg.
    Phase 3 (Remaining): Pure historical average.

    Args:
        historical_rates: Historical flow rates.
        rs_period_months: Reasonable & Supportable period (default 24 months).
        reversion_months: Reversion period (default 12 months).
        forecast_months: Total forecast horizon (default 120 months).
        macro_adjustment: Optional dict mapping rate column to multiplier (e.g., {"flow_rate_30": 1.15}).

    Returns:
        DataFrame with CECL-compliant flow rates.
    """
    rate_cols = [col for col in historical_rates.columns if col.startswith("flow_rate_")]

    # Long-run historical average
    long_run_avg = historical_rates[rate_cols].mean()

    # Phase 1: R&S period (macro-adjusted or current average)
    recent_avg = historical_rates.tail(6)[rate_cols].mean()
    if macro_adjustment is None:
        rs_rates = recent_avg
    else:
        rs_rates = recent_avg.copy()
        for col, multiplier in macro_adjustment.items():
            if col in rs_rates.index:
                rs_rates[col] *= multiplier

    # Generate future dates
    last_date = historical_rates["month_date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_months,
        freq="MS",
    )

    forecast = pd.DataFrame({"month_date": future_dates})

    for col in rate_cols:
        rates = []
        for i in range(forecast_months):
            if i < rs_period_months:
                # Phase 1: R&S
                rate = rs_rates[col]
            elif i < rs_period_months + reversion_months:
                # Phase 2: Straight-line reversion
                alpha = (i - rs_period_months) / reversion_months
                rate = (1 - alpha) * rs_rates[col] + alpha * long_run_avg[col]
            else:
                # Phase 3: Historical average
                rate = long_run_avg[col]
            rates.append(rate)

        forecast[col] = rates

    # Combine historical + forecast
    cecl_rates = pd.concat([historical_rates, forecast], ignore_index=True)
    return cecl_rates


# ── Roll Rate Computation ────────────────────────────────────────────────────

def compute_roll_counts(
    monthly_panel: pd.DataFrame,
    dpd_buckets: list[str] | None = None,
) -> pd.DataFrame:
    """Count accounts transitioning between DPD buckets month-over-month.

    For each loan, compare dpd_bucket(t) vs dpd_bucket(t-1).

    Args:
        monthly_panel: Output from reconstruct_monthly_panel().
        dpd_buckets: List of DPD bucket names (optional).

    Returns:
        DataFrame: month_date, from_bucket, to_bucket, account_count, balance_amount
                  (plus grade if available in panel)
    """
    if dpd_buckets is None:
        dpd_buckets = ["Current", "30+", "60+", "90+", "120+", "150+", "180+", "GCO"]

    panel = monthly_panel.sort_values(['loan_id', 'month_date']).copy()
    panel['prev_bucket'] = panel.groupby('loan_id')['dpd_bucket'].shift(1)

    # Drop first month per loan (no previous bucket)
    transitions = panel.dropna(subset=['prev_bucket'])

    # Include grade if available
    group_cols = ['month_date', 'prev_bucket', 'dpd_bucket']
    if 'grade' in transitions.columns:
        group_cols.append('grade')

    roll_counts = transitions.groupby(group_cols).agg(
        account_count=('loan_id', 'nunique'),
        balance_amount=('balance', 'sum'),
    ).reset_index()

    roll_counts = roll_counts.rename(columns={'prev_bucket': 'from_bucket', 'dpd_bucket': 'to_bucket'})
    return roll_counts


def compute_roll_rates(roll_counts: pd.DataFrame) -> pd.DataFrame:
    """Convert roll counts to roll rates (% of accounts that transitioned).

    Roll rate = accounts moving from A to B / total accounts in A

    Args:
        roll_counts: Output from compute_roll_counts().

    Returns:
        DataFrame with added column 'roll_rate' = account_count / total_in_bucket
    """
    # Total accounts in each from_bucket per month
    group_cols = ['month_date', 'from_bucket']
    if 'grade' in roll_counts.columns:
        group_cols.append('grade')

    totals = roll_counts.groupby(group_cols)['account_count'].sum().reset_index()
    totals = totals.rename(columns={'account_count': 'total_in_bucket'})

    result = roll_counts.merge(totals, on=group_cols)
    result['roll_rate'] = result['account_count'] / result['total_in_bucket']

    return result
