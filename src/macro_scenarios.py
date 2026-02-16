"""
Macro Scenario Analysis and Credit Strategy Module
====================================================

V6.1: Multi-factor, time-varying stress testing.

Three macro drivers (UNRATE, CSUSHPINSA/HPI, DFF) with 8-quarter forward paths
replace the single-factor static-delta approach. Stress is applied at the flow
rate level, preserving non-linear compounding dynamics:
  15% stress per flow rate → ~75% increase in cumulative flow-through rate.

Three ECL views (FEG Framework):
  - Pre-FEG: Pure model output, rolling 6-month average flow rates, no macro overlay
  - Central: Actual 2019 FRED path applied → improving economy → Central < Pre-FEG
  - Post-FEG: Probability-weighted across scenarios + qualitative adjustments

Key V6.1 fix: Central ≠ Pre-FEG because baseline delta is no longer zero.
Under the improving 2019 economy, Central < Pre-FEG < Post-FEG.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── Macro Regime Classification ──────────────────────────────────────────────

MACRO_REGIMES: dict[str, list[int]] = {
    "Crisis (2008-09)": [2008, 2009],
    "Recovery (2010-12)": [2010, 2011, 2012],
    "Expansion (2013-15)": [2013, 2014, 2015],
    "Late Cycle (2016-18)": [2016, 2017, 2018],
}

SCENARIO_DEFINITIONS: dict[str, dict[str, Any]] = {
    "baseline": {
        "label": "Baseline",
        "weight": 0.60,
        "unrate_delta": 0.0,
        "gdp_delta": 0.0,
        "description": "Current trajectory (UNRATE ~4%, GDP growth ~2%)",
    },
    "mild": {
        "label": "Mild Downturn",
        "weight": 0.25,
        "unrate_delta": 1.5,
        "gdp_delta": -0.5,
        "description": "Unemployment +1.5pp, GDP -0.5%",
    },
    "stress": {
        "label": "Stress",
        "weight": 0.15,
        "unrate_delta": 3.0,
        "gdp_delta": -3.0,
        "description": "Unemployment +3pp, GDP -3%",
    },
}

# ── V6.1: Multi-Factor Forward Path Constants ──────────────────────────────

# Baseline macro levels as of Q4 2018 (end of our data window).
# These serve as the "starting point" for forward projection.
# Source: FRED public data — UNRATE Dec 2018, HPI Dec 2018, DFF Dec 2018.
BASELINE_LEVELS: dict[str, float] = {
    "UNRATE": 4.0,        # 3.9% rounded — US unemployment Dec 2018
    "CSUSHPINSA": 203.0,  # Case-Shiller National HPI, Dec 2018
    "DFF": 1.8,           # Effective Federal Funds Rate, Dec 2018
}

# 8-quarter forward paths for 3 scenarios × 3 variables.
# Central: actual 2019-2020 FRED values (economy was improving in 2019,
#   then COVID hit in 2020 — but we use actual observed values).
# Mild Downturn: +1.5pp UNRATE shock, -5% HPI, moderately elevated DFF.
# Stress: +3.0pp UNRATE shock, -10% HPI, sharply elevated then easing DFF.
#
# DATA LIMITATION: No 2019 loan data in our dataset. Forward paths use
# external FRED values, not model-predicted. Production systems would use
# ARIMA/VAR models; we hardcode for transparency and reproducibility.
FORWARD_PATHS: dict[str, dict[str, list[float]]] = {
    "central": {
        # Actual 2019–2020 FRED: economy improving, HPI rising, Fed cutting rates
        # Source: FRED CSUSHPINSA ~208–230 through 2019–2020
        "UNRATE":     [3.8, 3.7, 3.6, 3.5, 3.5, 3.5, 3.5, 3.5],
        "CSUSHPINSA": [208.0, 211.0, 214.0, 217.0, 220.0, 219.0, 224.0, 230.0],
        "DFF":        [2.40, 2.40, 2.19, 1.55, 1.58, 0.65, 0.09, 0.09],
    },
    "mild": {
        # +1.5pp UNRATE shock from baseline, -5% HPI (~193), elevated DFF
        "UNRATE":     [5.5, 5.3, 5.2, 5.0, 4.8, 4.7, 4.6, 4.5],
        "CSUSHPINSA": [193.0, 192.0, 191.0, 190.0, 191.0, 192.0, 193.0, 194.0],
        "DFF":        [2.50, 2.60, 2.50, 2.20, 2.00, 1.80, 1.50, 1.20],
    },
    "stress": {
        # +5.5pp UNRATE shock from baseline (4.0), -15% HPI (~173), sharply elevated DFF
        # Roadmap: 5.5→7.0→8.5→9.5→10.0→9.5→9.0→8.5 with mean-reversion
        "UNRATE":     [5.5, 7.0, 8.5, 9.5, 10.0, 9.5, 9.0, 8.5],
        "CSUSHPINSA": [173.0, 170.0, 168.0, 167.0, 168.0, 170.0, 172.0, 175.0],
        "DFF":        [3.50, 3.80, 3.50, 3.00, 2.50, 2.00, 1.50, 0.80],
    },
}

# Scenario weights (same as V6.0 — unchanged)
SCENARIO_WEIGHTS: dict[str, float] = {
    "central": 0.60,
    "mild": 0.25,
    "stress": 0.15,
}


# ── Macro–Flow Rate Elasticity ──────────────────────────────────────────────

def compute_macro_flow_rate_relationship(
    flow_rates: pd.DataFrame,
    loans: pd.DataFrame,
    macro_col: str = "UNRATE",
) -> pd.DataFrame:
    """Compute relationship between macro variable and flow rates by period.

    Aggregates monthly flow rates to yearly averages, then merges with
    the average macro variable for that year from the loan dataset.

    Args:
        flow_rates: Historical flow rates (from compute_flow_rates with grade column).
        loans: Loan-level data with macro columns and issue_d.
        macro_col: Macro variable column name (default UNRATE).

    Returns:
        DataFrame with year, macro_avg, and average flow rates.
    """
    fr = flow_rates.copy()
    fr["month_date"] = pd.to_datetime(fr["month_date"])
    fr["year"] = fr["month_date"].dt.year

    rate_cols = [c for c in fr.columns if c.startswith("flow_rate_")]

    # Yearly average flow rates (across all grades)
    yearly_fr = fr.groupby("year")[rate_cols].mean().reset_index()

    # Yearly average macro from loan origination data
    loans = loans.copy()
    loans["issue_year"] = loans["issue_d"].dt.year
    yearly_macro = (
        loans.groupby("issue_year")[macro_col]
        .mean()
        .reset_index()
        .rename(columns={"issue_year": "year", macro_col: "macro_avg"})
    )

    merged = yearly_fr.merge(yearly_macro, on="year", how="inner")
    return merged


def estimate_flow_rate_elasticity(
    macro_flow_df: pd.DataFrame,
    rate_col: str = "flow_rate_30",
) -> dict[str, float]:
    """Estimate elasticity of flow rate to macro variable via OLS.

    elasticity = %Δflow_rate / %Δmacro_variable

    Args:
        macro_flow_df: Output from compute_macro_flow_rate_relationship.
        rate_col: Flow rate column to regress.

    Returns:
        Dict with slope, intercept, r_squared, elasticity.
    """
    df = macro_flow_df.dropna(subset=["macro_avg", rate_col])
    if len(df) < 3:
        return {"slope": np.nan, "intercept": np.nan, "r_squared": np.nan, "elasticity": np.nan}

    x = df["macro_avg"].values
    y = df[rate_col].values

    # Simple OLS
    n = len(x)
    x_mean, y_mean = x.mean(), y.mean()
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx == 0:
        return {"slope": 0.0, "intercept": y_mean, "r_squared": 0.0, "elasticity": 0.0}

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Point elasticity at mean
    elasticity = slope * x_mean / y_mean if y_mean != 0 else 0.0

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "elasticity": float(elasticity),
    }


# ── V6.1: Multi-Factor Elasticity & Forward Path Functions ──────────────────

def estimate_multi_factor_elasticity(
    macro_flow_df: pd.DataFrame,
    macro_cols: list[str] | None = None,
    rate_col: str = "flow_rate_30",
    alpha: float = 0.5,
    n_folds: int = 5,
) -> dict[str, dict[str, float]]:
    """Estimate separate elasticities for multiple macro variables via Ridge regression.

    Why Ridge (not OLS): With only 11 yearly observations and 3 predictors,
    OLS overfits (8 degrees of freedom). Ridge regularization (α=0.5) shrinks
    coefficients toward zero, improving out-of-sample stability. We report
    5-fold cross-validated MSE alongside point estimates for transparency.

    Formula:
        elasticity_i = slope_i × mean(X_i) / mean(y)

    This gives the percentage change in flow rate for a 1% change in the macro
    variable, evaluated at the sample means.

    Args:
        macro_flow_df: Output from compute_macro_flow_rate_relationship,
            with columns 'macro_avg' replaced by individual macro columns.
            Must have yearly data with macro variables and flow rates.
        macro_cols: Macro variable column names (default: UNRATE, CSUSHPINSA, DFF).
        rate_col: Flow rate column to regress (default: flow_rate_30, the entry rate).
        alpha: Ridge regularization strength (default: 0.5).
        n_folds: Number of cross-validation folds (default: 5).

    Returns:
        Dict mapping variable name → {slope, elasticity, r_squared, cv_mse, cv_std}.

    Data Limitation:
        11 yearly observations (2008-2018) is a very small sample.
        Ridge mitigates overfitting, but elasticity estimates should be
        interpreted as directional indicators, not precise point estimates.
    """
    if macro_cols is None:
        macro_cols = ["UNRATE", "CSUSHPINSA", "DFF"]

    df = macro_flow_df.dropna(subset=macro_cols + [rate_col])
    if len(df) < 5:
        return {col: {"slope": np.nan, "elasticity": np.nan, "r_squared": np.nan,
                       "cv_mse": np.nan, "cv_std": np.nan} for col in macro_cols}

    X = df[macro_cols].values
    y = df[rate_col].values
    n, p = X.shape

    # Standardize for Ridge (mean-center, unit variance)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=1)
    X_std[X_std == 0] = 1.0  # prevent division by zero
    X_scaled = (X - X_mean) / X_std
    y_mean = y.mean()
    y_centered = y - y_mean

    # Ridge closed-form: β = (X'X + αI)^{-1} X'y
    XtX = X_scaled.T @ X_scaled
    Xty = X_scaled.T @ y_centered
    beta_scaled = np.linalg.solve(XtX + alpha * np.eye(p), Xty)

    # Convert back to original scale
    beta_original = beta_scaled / X_std
    intercept = y_mean - (beta_original @ X_mean)

    # R-squared
    y_pred = X @ beta_original + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # k-fold Cross-validation MSE
    cv_mses = []
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    folds = np.array_split(indices, min(n_folds, n))

    for fold_idx in range(len(folds)):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != fold_idx])

        if len(train_idx) < 2:
            continue

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        # Standardize on train fold
        tr_mean = X_tr.mean(axis=0)
        tr_std = X_tr.std(axis=0, ddof=1)
        tr_std[tr_std == 0] = 1.0
        X_tr_s = (X_tr - tr_mean) / tr_std
        y_tr_mean = y_tr.mean()
        y_tr_c = y_tr - y_tr_mean

        try:
            b_s = np.linalg.solve(X_tr_s.T @ X_tr_s + alpha * np.eye(p), X_tr_s.T @ y_tr_c)
            b_o = b_s / tr_std
            intercept_cv = y_tr_mean - (b_o @ tr_mean)
            y_pred_cv = X_te @ b_o + intercept_cv
            cv_mses.append(np.mean((y_te - y_pred_cv) ** 2))
        except np.linalg.LinAlgError:
            continue

    cv_mse = float(np.mean(cv_mses)) if cv_mses else np.nan
    cv_std = float(np.std(cv_mses)) if len(cv_mses) > 1 else np.nan

    # Build per-variable results
    results = {}
    for i, col in enumerate(macro_cols):
        elasticity = beta_original[i] * X_mean[i] / y_mean if y_mean != 0 else 0.0
        results[col] = {
            "slope": float(beta_original[i]),
            "elasticity": float(elasticity),
            "r_squared": float(r_squared),
            "cv_mse": cv_mse,
            "cv_std": cv_std,
        }

    return results


def compute_multi_factor_macro_flow_relationship(
    flow_rates: pd.DataFrame,
    loans: pd.DataFrame,
    macro_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute yearly macro averages merged with yearly flow rate averages.

    Similar to compute_macro_flow_rate_relationship but for multiple macro
    variables at once.

    Args:
        flow_rates: Historical flow rates with grade column.
        loans: Loan-level data with macro columns and issue_d.
        macro_cols: List of macro column names.

    Returns:
        DataFrame with year, macro variables, and average flow rates.
    """
    if macro_cols is None:
        macro_cols = ["UNRATE", "CSUSHPINSA", "DFF"]

    fr = flow_rates.copy()
    fr["month_date"] = pd.to_datetime(fr["month_date"])
    fr["year"] = fr["month_date"].dt.year

    rate_cols = [c for c in fr.columns if c.startswith("flow_rate_")]
    yearly_fr = fr.groupby("year")[rate_cols].mean().reset_index()

    loans = loans.copy()
    loans["issue_year"] = loans["issue_d"].dt.year
    yearly_macro = (
        loans.groupby("issue_year")[macro_cols]
        .mean()
        .reset_index()
        .rename(columns={"issue_year": "year"})
    )

    merged = yearly_fr.merge(yearly_macro, on="year", how="inner")
    return merged


def generate_all_forward_paths(
    forward_paths: dict[str, dict[str, list[float]]] | None = None,
) -> pd.DataFrame:
    """Generate DataFrame of 8-quarter forward macro paths for all scenarios.

    Returns 24 rows (8 quarters × 3 scenarios) with columns:
    [quarter, scenario, UNRATE, CSUSHPINSA, DFF].

    Central scenario uses actual 2019-2020 FRED values. The economy was
    improving through 2019 (falling unemployment, rising HPI, Fed rate cuts),
    so Central macro conditions are BETTER than the Q4 2018 baseline.

    Mild and Stress scenarios apply constructed shocks with gradual
    mean-reversion (unemployment peaks early and slowly declines).

    Args:
        forward_paths: Dict of scenario → {variable → [8 quarterly values]}.
            Defaults to FORWARD_PATHS constant.

    Returns:
        DataFrame with 24 rows, columns: quarter, quarter_label, scenario,
        UNRATE, CSUSHPINSA, DFF.
    """
    if forward_paths is None:
        forward_paths = FORWARD_PATHS

    quarter_labels = [
        "2019-Q1", "2019-Q2", "2019-Q3", "2019-Q4",
        "2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4",
    ]

    rows = []
    for scenario, paths in forward_paths.items():
        n_quarters = len(next(iter(paths.values())))
        for q in range(n_quarters):
            row = {
                "quarter": q + 1,
                "quarter_label": quarter_labels[q] if q < len(quarter_labels) else f"Q{q+1}",
                "scenario": scenario,
            }
            for var_name, values in paths.items():
                row[var_name] = values[q]
            rows.append(row)

    return pd.DataFrame(rows)


def compute_quarterly_stress_multipliers(
    forward_paths_df: pd.DataFrame,
    elasticities: dict[str, dict[str, float]],
    baseline_levels: dict[str, float] | None = None,
    macro_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Convert forward paths + elasticities into per-quarter composite stress multipliers.

    For each quarter q and scenario s, the composite multiplier is:

        multiplier(q,s) = Σ_i [ β_i × (macro_i(q,s) - baseline_i) / baseline_i ]

    where β_i is the elasticity of flow_rate_30 to macro variable i,
    and (macro_i - baseline_i)/baseline_i is the proportional deviation from baseline.

    The composite multiplier can be NEGATIVE when macro conditions improve
    (e.g., Central scenario: unemployment falls, HPI rises → negative deviation
    for UNRATE, positive deviation for HPI but HPI elasticity is typically
    negative → both contribute to lower stress).

    Args:
        forward_paths_df: Output from generate_all_forward_paths().
        elasticities: Dict from estimate_multi_factor_elasticity().
        baseline_levels: Baseline macro levels (defaults to BASELINE_LEVELS).
        macro_cols: Macro variables to use (defaults to UNRATE, CSUSHPINSA, DFF).

    Returns:
        DataFrame with 24 rows: quarter, scenario, individual contributions,
        composite_multiplier.
    """
    if baseline_levels is None:
        baseline_levels = BASELINE_LEVELS
    if macro_cols is None:
        macro_cols = ["UNRATE", "CSUSHPINSA", "DFF"]

    rows = []
    for _, frow in forward_paths_df.iterrows():
        row = {
            "quarter": frow["quarter"],
            "quarter_label": frow["quarter_label"],
            "scenario": frow["scenario"],
        }

        composite = 0.0
        for col in macro_cols:
            bl = baseline_levels.get(col, 1.0)
            level = frow[col]
            deviation = (level - bl) / bl if bl != 0 else 0.0
            elast = elasticities.get(col, {}).get("elasticity", 0.0)
            contribution = elast * deviation

            row[f"{col}_level"] = level
            row[f"{col}_deviation"] = deviation
            row[f"{col}_contribution"] = contribution
            composite += contribution

        row["composite_multiplier"] = composite
        rows.append(row)

    return pd.DataFrame(rows)


def stress_flow_rates_by_quarter(
    flow_rates: pd.DataFrame,
    quarterly_multipliers: pd.DataFrame,
    lookback_months: int = 6,
    grades: list[str] | None = None,
) -> pd.DataFrame:
    """Apply time-varying quarterly multipliers to baseline flow rates.

    For each scenario and grade:
    1. Compute 6-month rolling average baseline rates
    2. For each of 8 quarters, apply that quarter's composite multiplier:
       stressed_rate(q) = base_rate × (1 + multiplier(q))
    3. Average the 8 quarterly snapshots to get a single stressed rate per transition
    4. Compute FTR from the averaged stressed rates

    This produces the same output structure as build_stress_comparison_by_grade(),
    so downstream functions (compute_ecl_by_scenario, compute_weighted_ecl) work
    unchanged.

    The key V6.1 improvement: Central scenario has NEGATIVE multipliers
    (improving economy → lower flow rates → Central ECL < Pre-FEG ECL).

    Args:
        flow_rates: Historical flow rates with 'grade' column.
        quarterly_multipliers: Output from compute_quarterly_stress_multipliers().
        lookback_months: Months for rolling average base rates.
        grades: Grade list (defaults to A-G).

    Returns:
        DataFrame with grade, scenario, stressed flow rates, FTR.
        Same structure as build_stress_comparison_by_grade().
    """
    if grades is None:
        grades = ["A", "B", "C", "D", "E", "F", "G"]

    rate_cols = [c for c in flow_rates.columns if c.startswith("flow_rate_")]
    scenarios = quarterly_multipliers["scenario"].unique()

    rows = []
    for grade in grades:
        grade_data = flow_rates[flow_rates["grade"] == grade] if "grade" in flow_rates.columns else flow_rates
        if len(grade_data) == 0:
            continue

        # Filter to active period
        active = grade_data[grade_data["flow_rate_30"] > 0]
        if len(active) == 0:
            active = grade_data.dropna(subset=["flow_rate_30"])
        if len(active) == 0:
            continue

        recent = active.tail(lookback_months)
        base_avg = recent[rate_cols].mean()

        for scenario in scenarios:
            sc_quarters = quarterly_multipliers[
                quarterly_multipliers["scenario"] == scenario
            ].sort_values("quarter")

            # Average stressed rates across all 8 quarters
            # KEY: Apply full multiplier to flow_rate_30 (entry rate) only.
            # Downstream rates (30→60, 60→90, etc.) are ~95% and driven by
            # workout/servicer behavior, not macro conditions. Applying the
            # full macro multiplier to 95% rates produces unrealistic results
            # (e.g., -40% × 95% = 57%). We use 10% dampening for downstream.
            DOWNSTREAM_DAMPENING = 0.10  # downstream gets 10% of macro effect
            quarterly_stressed = {col: [] for col in rate_cols}
            for _, qrow in sc_quarters.iterrows():
                mult = qrow["composite_multiplier"]
                for col in rate_cols:
                    if col == "flow_rate_30":
                        # Full macro effect on entry rate
                        effective_mult = max(mult, -0.5)
                    else:
                        # Dampened effect on downstream rates
                        effective_mult = max(mult * DOWNSTREAM_DAMPENING, -0.5)
                    stressed_val = max(0.0, min(base_avg[col] * (1 + effective_mult), 1.0))
                    quarterly_stressed[col].append(stressed_val)

            # Average across quarters
            row = {"grade": grade, "scenario": scenario}
            avg_mult = sc_quarters["composite_multiplier"].mean()
            row["stress_multiplier"] = avg_mult

            ftr_product = 1.0
            for col in rate_cols:
                avg_stressed = np.mean(quarterly_stressed[col])
                row[col] = avg_stressed
                if avg_stressed > 0:
                    ftr_product *= avg_stressed
            row["flow_through_rate"] = ftr_product
            rows.append(row)

    return pd.DataFrame(rows)


# ── V6.0 (Legacy) Flow Rate Stress ─────────────────────────────────────────

def compute_stress_multipliers(
    elasticity: float,
    scenario_defs: dict[str, dict[str, Any]] | None = None,
    baseline_macro: float = 4.0,
) -> dict[str, float]:
    """Convert macro scenario deltas to flow rate stress multipliers.

    stress_multiplier = elasticity × (macro_delta / baseline_macro)

    For UNRATE: if elasticity=1.5 and UNRATE goes from 4% to 5.5% (+1.5pp),
    then stress_multiplier = 1.5 × (1.5 / 4.0) = 0.5625 → 56% increase.

    Args:
        elasticity: Flow rate elasticity to macro variable.
        scenario_defs: Scenario definitions (defaults to SCENARIO_DEFINITIONS).
        baseline_macro: Baseline macro level (e.g., 4.0 for UNRATE).

    Returns:
        Dict mapping scenario name to stress multiplier (0 = no stress).
    """
    if scenario_defs is None:
        scenario_defs = SCENARIO_DEFINITIONS

    multipliers = {}
    for name, defn in scenario_defs.items():
        delta = defn["unrate_delta"]
        mult = elasticity * (delta / baseline_macro) if baseline_macro > 0 else 0.0
        multipliers[name] = max(mult, 0.0)  # no negative stress

    return multipliers


def stress_flow_rates(
    flow_rates: pd.DataFrame,
    stress_multiplier: float,
    rate_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply multiplicative stress to flow rates.

    stressed_rate = base_rate × (1 + stress_multiplier)

    Cap at 1.0 to prevent flow rates exceeding 100%.

    Args:
        flow_rates: Historical flow rates DataFrame.
        stress_multiplier: Multiplicative stress (e.g., 0.15 = 15% increase).
        rate_cols: Flow rate columns to stress (defaults to all flow_rate_* cols).

    Returns:
        Stressed flow rates DataFrame.
    """
    stressed = flow_rates.copy()

    if rate_cols is None:
        rate_cols = [c for c in stressed.columns if c.startswith("flow_rate_")]

    for col in rate_cols:
        if col in stressed.columns:
            stressed[col] = (stressed[col] * (1 + stress_multiplier)).clip(upper=1.0)

    return stressed


def compute_flow_through_from_rates(
    flow_rates_row: pd.Series | dict,
    rate_cols: list[str] | None = None,
) -> float:
    """Compute flow-through rate from a single row of flow rates.

    FTR = product of all intermediate flow rates.
    Demonstrates non-linear compounding of stress.

    Args:
        flow_rates_row: Series or dict with flow rate values.
        rate_cols: Column names for flow rates (in order).

    Returns:
        Flow-through rate (cumulative product).
    """
    if rate_cols is None:
        rate_cols = [
            "flow_rate_30", "flow_rate_60", "flow_rate_90",
            "flow_rate_120", "flow_rate_150", "flow_rate_180", "flow_rate_gco",
        ]

    product = 1.0
    for col in rate_cols:
        val = flow_rates_row.get(col, np.nan) if isinstance(flow_rates_row, dict) else flow_rates_row.get(col, np.nan)
        if pd.notna(val) and val > 0:
            product *= val

    return product


def build_stress_comparison(
    base_flow_rates: pd.DataFrame,
    scenario_multipliers: dict[str, float],
    lookback_months: int = 6,
) -> pd.DataFrame:
    """Build side-by-side flow rate stress comparison across scenarios.

    For each scenario, computes stressed average flow rates and the resulting FTR.
    Demonstrates non-linear compounding: 15% stress per rate → ~75% FTR increase.

    Args:
        base_flow_rates: Historical flow rates.
        scenario_multipliers: Dict mapping scenario name to stress multiplier.
        lookback_months: Months for rolling average base rates.

    Returns:
        DataFrame with scenarios as rows, flow rates + FTR as columns.
    """
    rate_cols = [c for c in base_flow_rates.columns if c.startswith("flow_rate_")]

    # Filter to active period (where flow_rate_30 has meaningful values)
    active = base_flow_rates[base_flow_rates["flow_rate_30"] > 0]
    if len(active) == 0:
        active = base_flow_rates.dropna(subset=["flow_rate_30"])

    # Compute base rates (rolling average of recent active months)
    recent = active.tail(lookback_months)
    base_avg = recent[rate_cols].mean()

    rows = []
    for scenario, mult in scenario_multipliers.items():
        row = {"scenario": scenario, "stress_multiplier": mult}
        ftr_product = 1.0
        for col in rate_cols:
            stressed_val = min(base_avg[col] * (1 + mult), 1.0)
            row[col] = stressed_val
            if pd.notna(stressed_val) and stressed_val > 0:
                ftr_product *= stressed_val
        row["flow_through_rate"] = ftr_product
        rows.append(row)

    return pd.DataFrame(rows)


def build_stress_comparison_by_grade(
    flow_rates: pd.DataFrame,
    scenario_multipliers: dict[str, float],
    lookback_months: int = 6,
    grades: list[str] | None = None,
) -> pd.DataFrame:
    """Build stress comparison by grade and scenario.

    Args:
        flow_rates: Historical flow rates with 'grade' column.
        scenario_multipliers: Dict mapping scenario to multiplier.
        lookback_months: Months for rolling average.
        grades: Grade list (defaults to A-G).

    Returns:
        DataFrame with grade, scenario, stressed flow rates, FTR.
    """
    if grades is None:
        grades = ["A", "B", "C", "D", "E", "F", "G"]

    rate_cols = [c for c in flow_rates.columns if c.startswith("flow_rate_")]
    rows = []

    for grade in grades:
        grade_data = flow_rates[flow_rates["grade"] == grade] if "grade" in flow_rates.columns else flow_rates
        if len(grade_data) == 0:
            continue

        # Filter to active period (where flow_rate_30 has meaningful values)
        active = grade_data[grade_data["flow_rate_30"] > 0]
        if len(active) == 0:
            active = grade_data.dropna(subset=["flow_rate_30"])
        if len(active) == 0:
            continue

        recent = active.tail(lookback_months)
        base_avg = recent[rate_cols].mean()

        for scenario, mult in scenario_multipliers.items():
            row = {"grade": grade, "scenario": scenario, "stress_multiplier": mult}
            ftr_product = 1.0
            for col in rate_cols:
                stressed_val = min(base_avg[col] * (1 + mult), 1.0)
                row[col] = stressed_val
                if pd.notna(stressed_val) and stressed_val > 0:
                    ftr_product *= stressed_val
            row["flow_through_rate"] = ftr_product
            rows.append(row)

    return pd.DataFrame(rows)


# ── Scenario ECL Computation ────────────────────────────────────────────────

def compute_ecl_by_scenario(
    ecl_by_grade: pd.DataFrame,
    stress_comparison: pd.DataFrame,
    baseline_ftr_col: str = "flow_through_rate",
) -> pd.DataFrame:
    """Compute ECL under each scenario using stressed flow-through rates.

    Scales Pre-FEG ECL by the ratio of stressed FTR to baseline FTR.
    ECL_scenario = ECL_baseline × (FTR_scenario / FTR_baseline)

    Args:
        ecl_by_grade: Pre-FEG ECL by grade (from Session 6).
        stress_comparison: Output from build_stress_comparison_by_grade.
        baseline_ftr_col: Column for FTR in stress_comparison.

    Returns:
        DataFrame with grade, scenario, ecl, ead, alll_ratio.
    """
    rows = []

    grades = ecl_by_grade["segment"].unique()
    scenarios = stress_comparison["scenario"].unique()

    # Get baseline FTR per grade
    baseline_ftr = (
        stress_comparison[stress_comparison["scenario"] == "baseline"]
        .set_index("grade")[baseline_ftr_col]
    )

    for grade in grades:
        ecl_row = ecl_by_grade[ecl_by_grade["segment"] == grade].iloc[0]
        base_ecl = ecl_row["total_ecl"]
        total_ead = ecl_row["total_ead"]
        count = ecl_row["count"]

        for scenario in scenarios:
            sc_row = stress_comparison[
                (stress_comparison["grade"] == grade) &
                (stress_comparison["scenario"] == scenario)
            ]
            if len(sc_row) == 0:
                continue

            sc_ftr = sc_row[baseline_ftr_col].iloc[0]
            base_ftr = baseline_ftr.get(grade, sc_ftr)

            # Scale ECL by FTR ratio
            if base_ftr > 0:
                scaling = sc_ftr / base_ftr
            else:
                scaling = 1.0

            scenario_ecl = base_ecl * scaling
            alll = scenario_ecl / total_ead if total_ead > 0 else 0.0

            rows.append({
                "grade": grade,
                "scenario": scenario,
                "total_ecl": scenario_ecl,
                "total_ead": total_ead,
                "count": count,
                "alll_ratio": alll,
                "ftr": sc_ftr,
                "ecl_scaling": scaling,
            })

    return pd.DataFrame(rows)


def compute_weighted_ecl(
    ecl_by_scenario: pd.DataFrame,
    scenario_weights: dict[str, float] | None = None,
    qualitative_adjustment: float = 0.0,
) -> pd.DataFrame:
    """Compute Post-FEG weighted ECL across scenarios.

    ECL_weighted = Σ(weight_i × ECL_i) × (1 + qualitative_adjustment)

    Args:
        ecl_by_scenario: Output from compute_ecl_by_scenario.
        scenario_weights: Weights per scenario (default: 0.60/0.25/0.15).
        qualitative_adjustment: Management overlay (e.g., 0.05 = +5%).

    Returns:
        DataFrame with grade-level weighted ECL.
    """
    if scenario_weights is None:
        scenario_weights = {
            s: d["weight"] for s, d in SCENARIO_DEFINITIONS.items()
        }

    rows = []
    for grade, gdf in ecl_by_scenario.groupby("grade"):
        weighted_ecl = 0.0
        total_ead = gdf["total_ead"].iloc[0]
        count = gdf["count"].iloc[0]

        for _, row in gdf.iterrows():
            w = scenario_weights.get(row["scenario"], 0.0)
            weighted_ecl += w * row["total_ecl"]

        # Apply qualitative overlay
        post_feg_ecl = weighted_ecl * (1 + qualitative_adjustment)
        alll = post_feg_ecl / total_ead if total_ead > 0 else 0.0

        rows.append({
            "grade": grade,
            "total_ecl": post_feg_ecl,
            "total_ead": total_ead,
            "count": count,
            "alll_ratio": alll,
            "qualitative_adjustment": qualitative_adjustment,
        })

    return pd.DataFrame(rows)


# ── Non-Linear Stress Demonstration ────────────────────────────────────────

def demonstrate_nonlinear_compounding(
    base_rates: dict[str, float],
    stress_pcts: list[float] | None = None,
) -> pd.DataFrame:
    """Demonstrate how stress compounds non-linearly through the waterfall.

    Key insight: 15% stress per individual flow rate results in ~75% increase
    in the cumulative flow-through rate due to multiplicative compounding.

    Args:
        base_rates: Dict of base flow rates (e.g., {"flow_rate_30": 0.005, ...}).
        stress_pcts: List of stress percentages to test (default: 0 to 50%).

    Returns:
        DataFrame with stress_pct, stressed_ftr, ftr_increase_pct.
    """
    if stress_pcts is None:
        stress_pcts = [0, 5, 10, 15, 20, 25, 30, 40, 50]

    rate_keys = sorted(base_rates.keys())
    base_ftr = 1.0
    for k in rate_keys:
        base_ftr *= base_rates[k]

    rows = []
    for pct in stress_pcts:
        mult = pct / 100.0
        stressed_ftr = 1.0
        for k in rate_keys:
            stressed_ftr *= min(base_rates[k] * (1 + mult), 1.0)
        increase = (stressed_ftr / base_ftr - 1) * 100 if base_ftr > 0 else 0.0
        rows.append({
            "stress_pct": pct,
            "base_ftr": base_ftr,
            "stressed_ftr": stressed_ftr,
            "ftr_increase_pct": increase,
        })

    return pd.DataFrame(rows)


# ── Sensitivity Analysis ────────────────────────────────────────────────────

def unemployment_sensitivity(
    base_flow_rates: pd.DataFrame,
    elasticity: float,
    baseline_unrate: float = 4.0,
    delta_range: list[float] | None = None,
    lookback_months: int = 6,
) -> pd.DataFrame:
    """Compute flow rate sensitivity to unemployment changes.

    Args:
        base_flow_rates: Historical flow rates.
        elasticity: UNRATE-to-flow-rate elasticity.
        baseline_unrate: Baseline unemployment rate.
        delta_range: Range of UNRATE deltas to test (default: [-2, -1, 0, +1, +2]).
        lookback_months: Months for base rate computation.

    Returns:
        DataFrame with unrate_delta, unrate_level, flow_rate_30, ftr.
    """
    if delta_range is None:
        delta_range = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    rate_cols = [c for c in base_flow_rates.columns if c.startswith("flow_rate_")]

    # Filter to active period (where flow_rate_30 has meaningful values)
    active = base_flow_rates[base_flow_rates["flow_rate_30"] > 0]
    if len(active) == 0:
        active = base_flow_rates.dropna(subset=["flow_rate_30"])

    recent = active.tail(lookback_months)
    base_avg = recent[rate_cols].mean()

    rows = []
    for delta in delta_range:
        mult = max(elasticity * (delta / baseline_unrate), -0.5) if baseline_unrate > 0 else 0.0

        row = {"unrate_delta": delta, "unrate_level": baseline_unrate + delta, "stress_multiplier": mult}
        ftr_product = 1.0
        for col in rate_cols:
            stressed_val = max(0, min(base_avg[col] * (1 + mult), 1.0))
            row[col] = stressed_val
            if stressed_val > 0:
                ftr_product *= stressed_val
        row["flow_through_rate"] = ftr_product
        rows.append(row)

    return pd.DataFrame(rows)


def recovery_rate_sensitivity(
    ecl_by_grade: pd.DataFrame,
    recovery_range: list[float] | None = None,
    base_recovery: float = 0.17,
) -> pd.DataFrame:
    """Compute ECL sensitivity to recovery rate assumptions.

    LGD = 1 - recovery_rate, so ECL scales roughly linearly with LGD.

    Args:
        ecl_by_grade: Pre-FEG ECL by grade.
        recovery_range: Range of recovery rates to test.
        base_recovery: Base recovery rate (LGD model output).

    Returns:
        DataFrame with recovery_rate, total_ecl, alll_ratio, pct_change.
    """
    if recovery_range is None:
        recovery_range = [0.10, 0.12, 0.15, 0.17, 0.20, 0.22, 0.25]

    base_ecl = ecl_by_grade["total_ecl"].sum()
    total_ead = ecl_by_grade["total_ead"].sum()
    base_lgd = 1 - base_recovery

    rows = []
    for rr in recovery_range:
        lgd = 1 - rr
        scaling = lgd / base_lgd if base_lgd > 0 else 1.0
        scenario_ecl = base_ecl * scaling
        alll = scenario_ecl / total_ead if total_ead > 0 else 0.0
        pct_change = (scenario_ecl / base_ecl - 1) * 100 if base_ecl > 0 else 0.0
        rows.append({
            "recovery_rate": rr,
            "lgd": lgd,
            "total_ecl": scenario_ecl,
            "alll_ratio": alll,
            "pct_change": pct_change,
        })

    return pd.DataFrame(rows)


def scorecard_cutoff_sensitivity(
    loans: pd.DataFrame,
    pd_col: str = "pd_pred",
    cutoffs: list[float] | None = None,
) -> pd.DataFrame:
    """Compute approval rate vs expected loss at various PD cutoffs.

    Args:
        loans: Loan-level data with PD predictions, int_rate, default flag.
        pd_col: Column for predicted PD.
        cutoffs: List of PD cutoff thresholds.

    Returns:
        DataFrame with cutoff, approval_rate, expected_loss_rate, avg_int_rate.
    """
    if cutoffs is None:
        cutoffs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0]

    total = len(loans)
    rows = []
    for cut in cutoffs:
        approved = loans[loans[pd_col] <= cut]
        n_approved = len(approved)
        approval_rate = n_approved / total if total > 0 else 0.0
        expected_loss = approved["default"].mean() if n_approved > 0 else 0.0
        avg_rate = approved["int_rate"].mean() if n_approved > 0 else 0.0
        rows.append({
            "pd_cutoff": cut,
            "n_approved": n_approved,
            "approval_rate": approval_rate,
            "expected_loss_rate": expected_loss,
            "avg_int_rate": avg_rate,
        })

    return pd.DataFrame(rows)


def build_tornado_chart_data(
    base_ecl: float,
    sensitivities: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Build tornado chart data from sensitivity results.

    Args:
        base_ecl: Base ECL value.
        sensitivities: Dict mapping factor name to (low_ecl, high_ecl) tuple.

    Returns:
        DataFrame sorted by impact range, with columns for plotting.
    """
    rows = []
    for factor, (low, high) in sensitivities.items():
        low_pct = (low / base_ecl - 1) * 100 if base_ecl > 0 else 0.0
        high_pct = (high / base_ecl - 1) * 100 if base_ecl > 0 else 0.0
        rows.append({
            "factor": factor,
            "low_ecl": low,
            "high_ecl": high,
            "low_pct_change": low_pct,
            "high_pct_change": high_pct,
            "range": abs(high_pct - low_pct),
        })

    df = pd.DataFrame(rows).sort_values("range", ascending=True)
    return df


# ── Credit Strategy Analysis ────────────────────────────────────────────────

def grade_profitability_analysis(
    loans: pd.DataFrame,
    ecl_by_grade: pd.DataFrame,
) -> pd.DataFrame:
    """Compute grade-level profitability: interest income minus expected loss.

    Args:
        loans: Loan-level data with grade, int_rate, funded_amnt, term.
        ecl_by_grade: Pre-FEG ECL by grade.

    Returns:
        DataFrame with grade, avg_int_rate, annual_income, expected_loss,
                 net_margin, spread_over_ecl.
    """
    grade_stats = (
        loans.groupby("grade")
        .agg(
            n_loans=("funded_amnt", "count"),
            total_balance=("funded_amnt", "sum"),
            avg_int_rate=("int_rate", "mean"),
            avg_loan_size=("funded_amnt", "mean"),
            default_rate=("default", "mean"),
        )
        .reset_index()
    )

    # Merge ECL data
    ecl_merge = ecl_by_grade[["segment", "total_ecl", "total_ead", "alll_ratio"]].copy()
    ecl_merge = ecl_merge.rename(columns={"segment": "grade"})
    grade_stats = grade_stats.merge(ecl_merge, on="grade", how="left")

    # Annualized interest income (simplified)
    grade_stats["annual_interest_income"] = (
        grade_stats["total_balance"] * grade_stats["avg_int_rate"] / 100
    )

    # Expected annual loss (ECL spread over average weighted term)
    avg_term_years = loans.groupby("grade")["term"].mean() / 12
    grade_stats = grade_stats.merge(
        avg_term_years.reset_index().rename(columns={"term": "avg_term_years"}),
        on="grade",
        how="left",
    )
    grade_stats["annual_expected_loss"] = (
        grade_stats["total_ecl"] / grade_stats["avg_term_years"]
    )

    # Net margin
    grade_stats["net_margin"] = (
        grade_stats["annual_interest_income"] - grade_stats["annual_expected_loss"]
    )
    grade_stats["net_margin_pct"] = (
        grade_stats["net_margin"] / grade_stats["total_balance"] * 100
    )

    # Spread over ECL rate
    grade_stats["spread_over_ecl"] = (
        grade_stats["avg_int_rate"] - grade_stats["alll_ratio"] * 100
    )

    return grade_stats


def vintage_comparison(
    loans: pd.DataFrame,
    vintage_a: int,
    vintage_b: int,
) -> pd.DataFrame:
    """Compare two vintages to identify performance differences.

    Args:
        loans: Loan-level data with issue_d, default, grade, int_rate, etc.
        vintage_a: First vintage year.
        vintage_b: Second vintage year.

    Returns:
        DataFrame comparing key metrics between vintages.
    """
    loans = loans.copy()
    loans["issue_year"] = loans["issue_d"].dt.year

    metrics = []
    for year in [vintage_a, vintage_b]:
        v = loans[loans["issue_year"] == year]
        metrics.append({
            "vintage": year,
            "n_loans": len(v),
            "total_balance": v["funded_amnt"].sum(),
            "avg_loan_size": v["funded_amnt"].mean(),
            "avg_int_rate": v["int_rate"].mean(),
            "default_rate": v["default"].mean(),
            "avg_dti": v["dti"].mean() if "dti" in v.columns else np.nan,
            "avg_fico": ((v["fico_range_low"] + v["fico_range_high"]) / 2).mean()
            if "fico_range_low" in v.columns else np.nan,
            "pct_grade_A": (v["grade"] == "A").mean() * 100,
            "pct_grade_D_G": v["grade"].isin(["D", "E", "F", "G"]).mean() * 100,
            "avg_annual_inc": v["annual_inc"].mean() if "annual_inc" in v.columns else np.nan,
        })

    df = pd.DataFrame(metrics)

    # Add deltas
    delta = {}
    for col in df.columns:
        if col == "vintage":
            continue
        delta[col] = df[col].iloc[1] - df[col].iloc[0]
    df = pd.concat([df, pd.DataFrame([{"vintage": "delta", **delta}])], ignore_index=True)

    return df


def pricing_adequacy_analysis(
    grade_profitability: pd.DataFrame,
    cost_of_funds: float = 2.0,
    operating_cost_bps: float = 200,
) -> pd.DataFrame:
    """Assess whether each grade is priced to cover its expected loss.

    Args:
        grade_profitability: Output from grade_profitability_analysis.
        cost_of_funds: Funding cost (% annual, default 2.0%).
        operating_cost_bps: Operating cost in basis points (default 200bps = 2%).

    Returns:
        DataFrame with pricing adequacy assessment per grade.
    """
    df = grade_profitability[["grade", "avg_int_rate", "alll_ratio", "default_rate"]].copy()

    # Required rate = cost of funds + operating cost + ECL rate
    operating_pct = operating_cost_bps / 100
    df["ecl_rate_pct"] = df["alll_ratio"] * 100
    df["required_rate"] = cost_of_funds + operating_pct + df["ecl_rate_pct"]
    df["excess_spread"] = df["avg_int_rate"] - df["required_rate"]
    df["adequately_priced"] = df["excess_spread"] > 0

    return df


# ── Output Helpers ──────────────────────────────────────────────────────────

def save_sensitivity_results(
    results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save sensitivity analysis results to JSON.

    Args:
        results: Dict of sensitivity results (DataFrames converted to records).
        output_path: Output file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
