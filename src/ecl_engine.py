"""
ECL (Expected Credit Loss) computation for LendingClub Credit Risk Analytics.

Implements both Simple ECL (PD × EAD × LGD) and DCF-based ECL with competing risks.

DCF-ECL mirrors LendingClub's 10-K methodology:
  1. Project monthly cash flows over remaining loan life
  2. Apply three competing outcomes per month:
     - P(stay current) × scheduled_payment
     - P(default) × recovery_value
     - P(prepay) × remaining_balance
  3. Discount at effective interest rate
  4. ECL = Contractual CF (NPV) - Expected CF (NPV)

Three ECL Views (FEG Framework):
  - Pre-FEG: Pure model output, no macro overlay
  - Central: Baseline macro scenario applied
  - Post-FEG: Weighted across scenarios + qualitative adjustments
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable


# ── Simple ECL ────────────────────────────────────────────────────────────────

def compute_simple_ecl(
    pd_values: pd.Series,
    ead_values: pd.Series,
    lgd_values: pd.Series,
) -> pd.Series:
    """Simple ECL = PD × EAD × LGD (point-in-time, loan-level).

    Args:
        pd_values: Probability of default per loan.
        ead_values: Exposure at default per loan.
        lgd_values: Loss given default per loan.

    Returns:
        ECL per loan.
    """
    return pd_values * ead_values * lgd_values


def ecl_by_segment(
    ecl_values: pd.Series,
    segment_col: pd.Series,
    segment_order: list | None = None,
) -> pd.DataFrame:
    """Aggregate ECL by segment (grade, vintage, purpose, etc.).

    Args:
        ecl_values: ECL per loan.
        segment_col: Segment identifier per loan.
        segment_order: Optional order for display.

    Returns:
        DataFrame with columns [segment, total_ecl, count, mean_ecl].
    """
    result = (
        pd.DataFrame({"ecl": ecl_values, "segment": segment_col})
        .groupby("segment")["ecl"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "total_ecl", "mean": "mean_ecl"})
        .reset_index()
    )

    if segment_order is not None:
        result["segment"] = pd.Categorical(result["segment"], categories=segment_order, ordered=True)
        result = result.sort_values("segment")

    return result


def compute_alll_ratio(
    total_ecl: float,
    total_balance: float,
) -> float:
    """Compute ALLL ratio = ECL / Outstanding Balance.

    Target: Compare to LendingClub 10-K ALLL ratio of ~5.7%.

    Args:
        total_ecl: Total expected credit loss.
        total_balance: Total outstanding balance.

    Returns:
        ALLL ratio (percentage).
    """
    if total_balance == 0:
        return 0.0
    return total_ecl / total_balance


# ── DCF-Based ECL with Competing Risks ────────────────────────────────────────

def compute_scheduled_payment(
    principal: float,
    rate: float,
    term: int,
) -> float:
    """Compute monthly scheduled payment via amortization formula.

    Payment = P × [r(1+r)^n] / [(1+r)^n - 1]

    Args:
        principal: Loan principal.
        rate: Monthly interest rate (e.g., 0.01 for 1%).
        term: Loan term in months.

    Returns:
        Monthly payment amount.
    """
    if rate == 0:
        return principal / term if term > 0 else 0

    numerator = rate * (1 + rate) ** term
    denominator = (1 + rate) ** term - 1
    return principal * (numerator / denominator)


def compute_remaining_balance(
    principal: float,
    rate: float,
    term: int,
    months_elapsed: int,
) -> float:
    """Compute remaining principal via amortization formula.

    Balance(t) = P × [(1+r)^n - (1+r)^t] / [(1+r)^n - 1]

    Args:
        principal: Original loan principal.
        rate: Monthly interest rate.
        term: Loan term in months.
        months_elapsed: Months since origination.

    Returns:
        Remaining principal balance.
    """
    if months_elapsed >= term:
        return 0.0

    if rate == 0:
        return principal * max(0, 1 - months_elapsed / term)

    numerator = (1 + rate) ** term - (1 + rate) ** months_elapsed
    denominator = (1 + rate) ** term - 1
    return principal * (numerator / denominator)


def dcf_ecl_single_loan(
    principal: float,
    int_rate: float,
    term: int,
    months_elapsed: int,
    monthly_pd: float | np.ndarray,
    lgd: float,
    prepay_rate: float,
) -> dict:
    """DCF-based ECL for a single loan with competing risks.

    Three competing outcomes per month:
      1. Stay current: P(current) × payment
      2. Default: P(default) × (1 - LGD) × balance
      3. Prepay: P(prepay) × balance

    Args:
        principal: Original loan principal.
        int_rate: Annual interest rate (%).
        term: Loan term in months.
        months_elapsed: Months already elapsed.
        monthly_pd: Monthly marginal PD (scalar or array of length remaining_months).
        lgd: Loss given default.
        prepay_rate: Monthly prepayment rate (CPR / 12 for SMM).

    Returns:
        Dict with keys:
            - contractual_npv: NPV of contractual cash flows
            - expected_npv: NPV of expected cash flows
            - ecl: Expected credit loss
            - loss_timing: Array of monthly loss amounts
    """
    r = int_rate / 100 / 12  # monthly rate
    remaining_months = term - months_elapsed

    if remaining_months <= 0:
        return {"contractual_npv": 0, "expected_npv": 0, "ecl": 0, "loss_timing": np.array([])}

    # Scheduled payment
    payment = compute_scheduled_payment(principal, r, term)

    # If monthly_pd is scalar, broadcast to array
    if isinstance(monthly_pd, (int, float)):
        monthly_pd = np.full(remaining_months, monthly_pd)
    else:
        monthly_pd = np.array(monthly_pd[:remaining_months])

    # Initialize
    contractual_cf = []
    expected_cf = []
    balance = compute_remaining_balance(principal, r, term, months_elapsed)
    survival_prob = 1.0  # cumulative probability of not yet defaulted/prepaid

    for t in range(remaining_months):
        # Discount factor
        discount = 1 / (1 + r) ** (t + 1)

        # Contractual cash flow
        contractual_cf.append(payment * discount)

        # Competing risks
        p_default = monthly_pd[t]
        p_prepay = prepay_rate
        p_current = 1 - p_default - p_prepay

        # Ensure probabilities are valid
        p_current = max(0, min(1, p_current))
        p_default = max(0, min(1, p_default))
        p_prepay = max(0, min(1, p_prepay))

        # Expected cash flow
        cf_current = survival_prob * p_current * payment
        cf_default = survival_prob * p_default * (1 - lgd) * balance
        cf_prepay = survival_prob * p_prepay * balance

        expected_cf.append((cf_current + cf_default + cf_prepay) * discount)

        # Update survival probability
        survival_prob *= p_current

        # Update balance
        if t < remaining_months - 1:
            balance = compute_remaining_balance(principal, r, term, months_elapsed + t + 1)

        if survival_prob < 1e-6:
            break

    contractual_npv = sum(contractual_cf)
    expected_npv = sum(expected_cf)
    ecl = max(0, contractual_npv - expected_npv)

    loss_timing = contractual_npv - np.array(expected_cf) if expected_cf else np.array([])

    return {
        "contractual_npv": contractual_npv,
        "expected_npv": expected_npv,
        "ecl": ecl,
        "loss_timing": loss_timing,
    }


def dcf_ecl_portfolio(
    df: pd.DataFrame,
    pd_col: str = "pd",
    lgd_col: str = "lgd",
    prepay_col: str = "prepay_rate",
) -> pd.DataFrame:
    """DCF-based ECL for entire portfolio.

    Args:
        df: DataFrame with columns:
            - funded_amnt, int_rate, term, months_elapsed
            - pd (or pd_col): Probability of default
            - lgd (or lgd_col): Loss given default
            - prepay_rate (or prepay_col): Monthly prepayment rate

    Returns:
        DataFrame with added columns:
            - contractual_npv, expected_npv, ecl_dcf
    """
    results = []

    for idx, row in df.iterrows():
        result = dcf_ecl_single_loan(
            principal=row["funded_amnt"],
            int_rate=row["int_rate"],
            term=row["term"],
            months_elapsed=row.get("months_elapsed", 0),
            monthly_pd=row[pd_col],
            lgd=row[lgd_col],
            prepay_rate=row[prepay_col],
        )
        results.append(result)

    df = df.copy()
    df["contractual_npv"] = [r["contractual_npv"] for r in results]
    df["expected_npv"] = [r["expected_npv"] for r in results]
    df["ecl_dcf"] = [r["ecl"] for r in results]

    return df


def dcf_ecl_batch(
    df: pd.DataFrame,
    pd_col: str = 'pd_pred',
    lgd_col: str = 'lgd_pred',
    prepay_col: str = 'prepay_rate',
    principal_col: str = 'funded_amnt',
    rate_col: str = 'int_rate',
    term_col: str = 'term',
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute DCF-ECL for a batch of loans (vectorized).

    Vectorized outer loop processes all loans simultaneously per month.
    ~100x faster than loan-by-loan iteration.

    Args:
        df: DataFrame with loan features.
        pd_col: Column name for probability of default (lifetime PD).
        lgd_col: Column name for loss given default.
        prepay_col: Column name for monthly prepayment rate.
        principal_col: Column name for loan principal.
        rate_col: Column name for annual interest rate (%).
        term_col: Column name for loan term (months).

    Returns:
        Tuple of (result_df, monthly_loss_matrix):
            - result_df: DataFrame with contractual_npv, expected_npv, ecl_dcf
            - monthly_loss_matrix: ndarray (n_loans × max_term) with monthly loss amounts
    """
    n = len(df)
    principals = df[principal_col].values.astype(float)
    rates = df[rate_col].values.astype(float) / 100 / 12  # monthly rate
    terms = df[term_col].values.astype(int)

    # Convert lifetime PD to monthly PD (simple approximation)
    lifetime_pds = df[pd_col].values.astype(float)
    monthly_pds = lifetime_pds / terms.astype(float)

    lgds = df[lgd_col].values.astype(float)
    prepay_rates = df[prepay_col].values.astype(float)

    max_term = int(terms.max())

    # Arrays for all loans
    balance = principals.copy()
    survival = np.ones(n)
    contractual_npv = np.zeros(n)
    expected_npv = np.zeros(n)
    monthly_losses = np.zeros((n, max_term))

    # Monthly payment (vectorized amortization formula)
    # For r == 0, use linear approximation
    payment = np.zeros(n)
    mask_positive_rate = rates > 0
    mask_zero_rate = rates == 0

    # Positive rate: P × [r(1+r)^n] / [(1+r)^n - 1]
    payment[mask_positive_rate] = principals[mask_positive_rate] * rates[mask_positive_rate] * (1 + rates[mask_positive_rate]) ** terms[mask_positive_rate] / ((1 + rates[mask_positive_rate]) ** terms[mask_positive_rate] - 1)

    # Zero rate: P / n
    payment[mask_zero_rate] = principals[mask_zero_rate] / terms[mask_zero_rate]

    for t in range(max_term):
        active = t < terms  # mask for loans still active
        discount = 1.0 / (1.0 + rates) ** (t + 1)

        # Contractual CF
        contractual_cf = np.where(active, payment * discount, 0)
        contractual_npv += contractual_cf

        # Competing probabilities
        p_default = np.where(active, monthly_pds, 0)
        p_prepay = np.where(active, prepay_rates, 0)
        p_current = np.where(active, np.maximum(1 - p_default - p_prepay, 0), 0)

        # Expected CFs
        cf_current = survival * p_current * payment * discount
        cf_default = survival * p_default * (1 - lgds) * balance * discount
        cf_prepay = survival * p_prepay * balance * discount

        expected_cf = np.where(active, cf_current + cf_default + cf_prepay, 0)
        expected_npv += expected_cf

        # Monthly loss = contractual - expected (already discounted)
        monthly_losses[:, t] = np.where(active, contractual_cf - expected_cf, 0)

        # Update survival and balance
        survival *= np.where(active, p_current, 1)

        # Update balance (amortization)
        interest = balance * rates
        principal_payment = np.where(active, payment - interest, 0)
        balance = np.where(active, np.maximum(balance - principal_payment, 0), 0)

    ecl = contractual_npv - expected_npv

    result = pd.DataFrame({
        'contractual_npv': contractual_npv,
        'expected_npv': expected_npv,
        'ecl_dcf': ecl,
    }, index=df.index)

    return result, monthly_losses


# ── ALLL Tracker ──────────────────────────────────────────────────────────────

def build_alll_tracker(
    monthly_ecl: pd.DataFrame,
    monthly_nco: pd.DataFrame,
) -> pd.DataFrame:
    """Build ALLL reserve tracker with provision expense.

    Args:
        monthly_ecl: DataFrame with columns [month_date, total_ecl].
        monthly_nco: DataFrame with columns [month_date, nco_amount].

    Returns:
        DataFrame with columns:
            - month_date, total_ecl, nco, provision, alll_reserve, nco_coverage_ratio
    """
    merged = pd.merge(monthly_ecl, monthly_nco, on="month_date", how="outer").fillna(0)
    merged = merged.sort_values("month_date")

    # ALLL reserve = prior ALLL + provision - NCO
    # Provision = ΔECL + NCO
    merged["provision"] = merged["total_ecl"].diff().fillna(0) + merged["nco_amount"]
    merged["alll_reserve"] = merged["total_ecl"]  # simplified: ALLL ≈ ECL

    # NCO coverage ratio = ALLL / annualized NCO
    annualized_nco = merged["nco_amount"].rolling(12, min_periods=1).sum()
    merged["nco_coverage_ratio"] = np.where(
        annualized_nco > 0,
        merged["alll_reserve"] / annualized_nco,
        np.nan,
    )

    return merged[["month_date", "total_ecl", "nco_amount", "provision", "alll_reserve", "nco_coverage_ratio"]]


# ── Three ECL Views (FEG Framework) ───────────────────────────────────────────

def apply_macro_adjustment(
    flow_rates: pd.DataFrame,
    scenario: str,
    macro_multipliers: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Apply macro scenario adjustment to flow rates.

    Args:
        flow_rates: DataFrame with flow rate columns.
        scenario: Scenario name (e.g., "baseline", "downturn", "stress").
        macro_multipliers: Dict mapping scenario → {rate_col: multiplier}.
            Example: {"downturn": {"flow_rate_30": 1.15, "flow_rate_60": 1.20}}

    Returns:
        Adjusted flow rates DataFrame.
    """
    adjusted = flow_rates.copy()

    if scenario in macro_multipliers:
        multipliers = macro_multipliers[scenario]
        for col, mult in multipliers.items():
            if col in adjusted.columns:
                adjusted[col] *= mult

    return adjusted


def compute_post_feg_ecl(
    ecl_by_scenario: dict[str, float],
    scenario_weights: dict[str, float],
    qualitative_adjustment: float = 0.0,
) -> float:
    """Compute Post-FEG ECL as weighted average + qualitative overlay.

    Args:
        ecl_by_scenario: Dict mapping scenario → ECL value.
        scenario_weights: Dict mapping scenario → weight (must sum to 1).
        qualitative_adjustment: Additional management overlay (e.g., +0.10 = +10%).

    Returns:
        Post-FEG ECL.
    """
    weighted_ecl = sum(ecl_by_scenario[s] * scenario_weights[s] for s in scenario_weights)
    post_feg = weighted_ecl * (1 + qualitative_adjustment)
    return post_feg
