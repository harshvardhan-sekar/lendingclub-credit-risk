"""
EAD and LGD model utilities for LendingClub Credit Risk Analytics.

EAD (Exposure at Default): Reconstructed via amortization schedule since
out_prncp is zeroed for charged-off loans in the data snapshot.

LGD (Loss Given Default): Two-stage approach —
  Stage 1: LogisticRegression for P(any recovery)
  Stage 2: GradientBoostingRegressor for E[recovery_rate | recovery]
  Combined: LGD = 1 - P(recovery) × E[recovery_rate | recovery]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


# ── EAD Utilities ─────────────────────────────────────────────────────────────

def compute_ead_amortization(
    funded_amnt: pd.Series,
    int_rate: pd.Series,
    term: pd.Series,
    months_elapsed: pd.Series,
) -> pd.Series:
    """Reconstruct outstanding principal at default via amortization schedule.

    The LendingClub data snapshot zeros out_prncp after charge-off.
    This reconstructs the remaining balance using the standard amortization
    formula: B(t) = P × [(1+r)^n - (1+r)^t] / [(1+r)^n - 1].

    Args:
        funded_amnt: Original funded loan amount.
        int_rate: Annual interest rate (percentage, e.g. 12.5 for 12.5%).
        term: Loan term in months (36 or 60).
        months_elapsed: Months from issue to last payment.

    Returns:
        Remaining principal balance (EAD estimate).
    """
    r = int_rate / 100 / 12  # monthly rate
    n = term
    t = months_elapsed.clip(lower=0)

    numerator = (1 + r) ** n - (1 + r) ** t
    denominator = (1 + r) ** n - 1

    ead = funded_amnt * (numerator / denominator)
    # Handle edge cases: zero rate, t >= n, invalid values
    ead = ead.where((r > 0) & (t < n) & ead.notna(), other=np.nan)
    return ead.clip(lower=0)


def compute_ccf(ead: pd.Series, funded_amnt: pd.Series) -> pd.Series:
    """Compute Credit Conversion Factor = EAD / funded_amnt.

    Args:
        ead: Exposure at default (remaining balance).
        funded_amnt: Original funded loan amount.

    Returns:
        CCF values clipped to [0, 1].
    """
    ccf = ead / funded_amnt
    return ccf.clip(0, 1)


def fit_ead_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    **kwargs,
) -> RandomForestRegressor:
    """Fit a Random Forest regressor for EAD prediction.

    Args:
        X_train: Feature matrix (loan_amnt, term, grade_enc, annual_inc, dti, fico_avg).
        y_train: Target — reconstructed EAD (remaining principal at default).
        random_state: Reproducibility seed.
        **kwargs: Additional RF hyperparameters.

    Returns:
        Fitted RandomForestRegressor.
    """
    params = dict(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=50,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )
    params.update(kwargs)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model


# ── LGD Utilities ─────────────────────────────────────────────────────────────

def compute_lgd_primary(
    recoveries: pd.Series,
    collection_recovery_fee: pd.Series,
    ead: pd.Series,
) -> pd.Series:
    """Primary LGD formula (V4 update).

    LGD = 1 - (net_recovery / EAD)
    where net_recovery = recoveries - collection_recovery_fee.

    Args:
        recoveries: Post-charge-off gross recovery.
        collection_recovery_fee: Fee paid to recovery agent.
        ead: Exposure at default (reconstructed remaining balance).

    Returns:
        LGD values clipped to [0, 1].
    """
    net_recovery = (recoveries - collection_recovery_fee).clip(lower=0)
    lgd = 1 - (net_recovery / ead)
    lgd = lgd.where(ead > 0, other=1.0)  # no exposure → full loss
    return lgd.clip(0, 1)


def compute_lgd_simple(
    total_rec_prncp: pd.Series,
    ead: pd.Series,
) -> pd.Series:
    """Cross-check LGD formula using total principal received.

    LGD_simple = 1 - (total_rec_prncp / EAD)
    Note: total_rec_prncp includes pre-default principal payments,
    so this measure has different economic meaning from the primary formula.

    Args:
        total_rec_prncp: Total principal received to date.
        ead: Exposure at default (reconstructed remaining balance).

    Returns:
        LGD values (may be negative if pre-default payments exceed EAD).
    """
    return 1 - (total_rec_prncp / ead)


def compute_recovery_targets(
    recoveries: pd.Series,
    collection_recovery_fee: pd.Series,
    ead: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Compute targets for the two-stage LGD model.

    Args:
        recoveries: Post-charge-off gross recovery.
        collection_recovery_fee: Fee paid to recovery agent.
        ead: Exposure at default.

    Returns:
        Tuple of (recovery_flag, recovery_rate):
          - recovery_flag: 1 if net recovery > 0, else 0
          - recovery_rate: net_recovery / EAD, clipped to [0, 1]
    """
    net_recovery = (recoveries - collection_recovery_fee).clip(lower=0)
    recovery_flag = (net_recovery > 0).astype(int)
    recovery_rate = (net_recovery / ead).clip(0, 1)
    recovery_rate = recovery_rate.where(ead > 0, other=0.0)
    return recovery_flag, recovery_rate


def fit_lgd_stage1(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    **kwargs,
) -> LogisticRegression:
    """Fit Stage 1 (classification): P(any recovery).

    Args:
        X_train: Feature matrix.
        y_train: Binary target — 1 if any net recovery, 0 otherwise.
        random_state: Reproducibility seed.
        **kwargs: Additional LogisticRegression params.

    Returns:
        Fitted LogisticRegression.
    """
    params = dict(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
    )
    params.update(kwargs)
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


def fit_lgd_stage2(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    **kwargs,
) -> GradientBoostingRegressor:
    """Fit Stage 2 (regression): E[recovery_rate | recovery].

    Trained only on loans with positive net recovery.

    Args:
        X_train: Feature matrix (recovery loans only).
        y_train: Recovery rate = net_recovery / EAD, clipped to [0, 1].
        random_state: Reproducibility seed.
        **kwargs: Additional GBR params.

    Returns:
        Fitted GradientBoostingRegressor.
    """
    params = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=50,
        subsample=0.8,
        random_state=random_state,
    )
    params.update(kwargs)
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model


def predict_lgd(
    stage1_model: LogisticRegression,
    stage2_model: GradientBoostingRegressor,
    X: pd.DataFrame,
) -> np.ndarray:
    """Combined two-stage LGD prediction.

    LGD = 1 - P(recovery) × E[recovery_rate | recovery]

    Args:
        stage1_model: Fitted classification model for P(recovery).
        stage2_model: Fitted regression model for recovery_rate.
        X: Feature matrix.

    Returns:
        LGD predictions clipped to [0, 1].
    """
    p_recovery = stage1_model.predict_proba(X)[:, 1]
    e_recovery_rate = np.clip(stage2_model.predict(X), 0, 1)
    lgd = 1 - p_recovery * e_recovery_rate
    return np.clip(lgd, 0, 1)


def lgd_by_grade(
    lgd_values: pd.Series,
    grades: pd.Series,
    grade_order: list[str] | None = None,
) -> pd.DataFrame:
    """Compute portfolio-level LGD by letter grade.

    Args:
        lgd_values: LGD per loan.
        grades: Letter grade (A-G).
        grade_order: Ordered grade list for display.

    Returns:
        DataFrame with columns [grade, mean_lgd, count].
    """
    if grade_order is None:
        grade_order = ["A", "B", "C", "D", "E", "F", "G"]

    result = (
        pd.DataFrame({"lgd": lgd_values, "grade": grades})
        .groupby("grade")["lgd"]
        .agg(["mean", "count"])
        .reindex(grade_order)
        .reset_index()
        .rename(columns={"mean": "mean_lgd"})
    )
    return result
