"""
Model Validation and Monitoring Module
=======================================

Institutional-grade validation metrics for credit risk models:
- Discrimination: Gini, KS, AUC with bootstrap CI, CAP curve
- Calibration: Hosmer-Lemeshow, Brier score, decile calibration
- Stability: PSI, CSI, VDI
- RAG framework: Bank-standard quarterly monitoring thresholds

Mirrors quarterly model monitoring report format from prior institution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    auc,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# RAG thresholds — mirrors institutional quarterly monitoring
# ---------------------------------------------------------------------------

RAG_THRESHOLDS: dict[str, dict[str, tuple[float, float]]] = {
    # PD Scorecard (V5.1 corrected): Green >= 42%, Amber 36-42%, Red < 36%
    "gini_scorecard": {"green": (0.42, float("inf")), "amber": (0.36, 0.42)},
    # ML Models: Green >= 46%, Amber 42-46%, Red < 42%
    "gini_ml": {"green": (0.46, float("inf")), "amber": (0.42, 0.46)},
    # PSI: Green < 0.10, Amber 0.10-0.25, Red >= 0.25
    "psi": {"green": (0.0, 0.10), "amber": (0.10, 0.25)},
    # CSI (same thresholds as PSI)
    "csi": {"green": (0.0, 0.10), "amber": (0.10, 0.25)},
    # VDI (same thresholds as PSI)
    "vdi": {"green": (0.0, 0.10), "amber": (0.10, 0.25)},
    # AUC Scorecard: Green >= 0.71, Amber 0.68-0.71, Red < 0.68
    "auc_scorecard": {"green": (0.71, float("inf")), "amber": (0.68, 0.71)},
    # AUC ML: Green >= 0.73, Amber 0.71-0.73, Red < 0.71
    "auc_ml": {"green": (0.73, float("inf")), "amber": (0.71, 0.73)},
    # KS Scorecard: Green >= 30%, Amber 26-30%, Red < 26%
    "ks_scorecard": {"green": (0.30, float("inf")), "amber": (0.26, 0.30)},
    # Brier Skill Score: Green >= 0.20, Amber 0.10-0.20, Red < 0.10 (higher is better)
    # BSS = 1 - brier_model / brier_naive; scale-invariant relative to prevalence baseline
    "brier_skill_score": {"green": (0.20, float("inf")), "amber": (0.10, 0.20)},
    # Overfit gap (train-test AUC): Green < 0.03, Amber 0.03-0.05, Red >= 0.05
    "overfit_gap": {"green": (0.0, 0.03), "amber": (0.03, 0.05)},
    # Hosmer-Lemeshow p-value: Green > 0.05, Amber 0.01-0.05, Red < 0.01
    "hosmer_lemeshow_p": {"green": (0.05, float("inf")), "amber": (0.01, 0.05)},
    # EAD MAPE: Green < 0.15, Amber 0.15-0.25, Red >= 0.25
    "ead_mape": {"green": (0.0, 0.15), "amber": (0.15, 0.25)},
    # LGD MAE: Green < 0.10, Amber 0.10-0.15, Red >= 0.15
    "lgd_mae": {"green": (0.0, 0.10), "amber": (0.10, 0.15)},
    # LGD Stage 1 AUC: Green >= 0.65, Amber 0.55-0.65, Red < 0.55
    "lgd_stage1_auc": {"green": (0.65, float("inf")), "amber": (0.55, 0.65)},
}


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def compute_brier_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier Skill Score (BSS) relative to naive prevalence baseline.

    BSS = 1 - brier_model / brier_naive, where brier_naive = p*(1-p) for the
    prevalence p. BSS is scale-invariant and higher-is-better (unlike raw Brier).
    BSS > 0 means model beats the naive baseline; BSS = 1 is perfect.

    Parameters
    ----------
    y_true : array-like
        Binary labels (0/1).
    y_pred : array-like
        Predicted probabilities.

    Returns
    -------
    float
        Brier Skill Score in (−∞, 1].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    brier_model = brier_score_loss(y_true, y_pred)
    prevalence = float(y_true.mean())
    brier_naive = prevalence * (1.0 - prevalence)
    if brier_naive == 0.0:
        return 0.0
    return float(1.0 - brier_model / brier_naive)


# ---------------------------------------------------------------------------
# Discrimination metrics
# ---------------------------------------------------------------------------


def compute_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Gini coefficient = 2 * AUC - 1.

    Parameters
    ----------
    y_true : array-like
        Binary labels (0/1).
    y_pred : array-like
        Predicted probabilities or scores (higher = more likely default).

    Returns
    -------
    float
        Gini coefficient in [−1, 1].
    """
    auc_val = roc_auc_score(y_true, y_pred)
    return 2 * auc_val - 1


def compute_ks(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute KS statistic and cumulative distribution data for plotting.

    Parameters
    ----------
    y_true : array-like
        Binary labels (0/1).
    y_pred : array-like
        Predicted probabilities (higher = more likely default).

    Returns
    -------
    dict
        ks_statistic : float
            Maximum separation between cumulative good/bad distributions.
        ks_threshold : float
            Score threshold at which KS is achieved.
        plot_data : DataFrame
            Columns: threshold, cum_bad_rate, cum_good_rate, ks_diff
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Sort by predicted probability descending
    order = np.argsort(-y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]

    n_bad = y_true.sum()
    n_good = len(y_true) - n_bad

    cum_bad = np.cumsum(y_true_sorted) / n_bad
    cum_good = np.cumsum(1 - y_true_sorted) / n_good
    ks_diff = np.abs(cum_bad - cum_good)

    ks_idx = np.argmax(ks_diff)
    ks_statistic = float(ks_diff[ks_idx])
    ks_threshold = float(y_pred_sorted[ks_idx])

    # Subsample for plotting (every nth point)
    n = len(y_pred_sorted)
    step = max(1, n // 200)
    idx = np.arange(0, n, step)

    plot_data = pd.DataFrame(
        {
            "threshold": y_pred_sorted[idx],
            "cum_bad_rate": cum_bad[idx],
            "cum_good_rate": cum_good[idx],
            "ks_diff": ks_diff[idx],
        }
    )

    return {
        "ks_statistic": ks_statistic,
        "ks_threshold": ks_threshold,
        "plot_data": plot_data,
    }


def compute_auc_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute AUC with bootstrap confidence interval.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_pred : array-like
        Predicted probabilities.
    n_bootstrap : int
        Number of bootstrap iterations.
    confidence : float
        Confidence level (default 0.95 for 95% CI).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        auc : float, ci_lower : float, ci_upper : float, std : float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(random_state)

    auc_point = roc_auc_score(y_true, y_pred)
    aucs = np.empty(n_bootstrap)
    n = len(y_true)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        # Ensure both classes present in bootstrap sample
        if len(np.unique(y_true[idx])) < 2:
            aucs[i] = np.nan
            continue
        aucs[i] = roc_auc_score(y_true[idx], y_pred[idx])

    aucs = aucs[~np.isnan(aucs)]
    alpha = (1 - confidence) / 2

    return {
        "auc": float(auc_point),
        "ci_lower": float(np.percentile(aucs, 100 * alpha)),
        "ci_upper": float(np.percentile(aucs, 100 * (1 - alpha))),
        "std": float(np.std(aucs)),
    }


def compute_cap_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Compute Cumulative Accuracy Profile (CAP) curve data.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_pred : array-like
        Predicted probabilities (higher = more likely default).

    Returns
    -------
    dict
        ar : float
            Accuracy Ratio (= Gini).
        plot_data : DataFrame
            Columns: pct_population, pct_defaults_model, pct_defaults_perfect, pct_defaults_random
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    n_defaults = y_true.sum()

    # Sort by predicted probability descending
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]

    cum_defaults = np.cumsum(y_sorted)
    pct_pop = np.arange(1, n + 1) / n
    pct_defaults_model = cum_defaults / n_defaults

    # Perfect model: all defaults first
    perfect_sorted = np.sort(y_true)[::-1]
    pct_defaults_perfect = np.cumsum(perfect_sorted) / n_defaults

    # Random model: diagonal
    pct_defaults_random = pct_pop

    # Accuracy Ratio = area between model and random / area between perfect and random
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area_model = _trapz(pct_defaults_model, pct_pop)
    area_random = _trapz(pct_defaults_random, pct_pop)
    area_perfect = _trapz(pct_defaults_perfect, pct_pop)
    ar = (area_model - area_random) / (area_perfect - area_random)

    # Subsample for plotting
    step = max(1, n // 200)
    idx = np.arange(0, n, step)

    plot_data = pd.DataFrame(
        {
            "pct_population": pct_pop[idx],
            "pct_defaults_model": pct_defaults_model[idx],
            "pct_defaults_perfect": pct_defaults_perfect[idx],
            "pct_defaults_random": pct_defaults_random[idx],
        }
    )

    return {"ar": float(ar), "plot_data": plot_data}


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def compute_hosmer_lemeshow(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_groups: int = 10,
) -> dict[str, Any]:
    """Hosmer-Lemeshow goodness-of-fit test.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_pred : array-like
        Predicted probabilities.
    n_groups : int
        Number of decile groups.

    Returns
    -------
    dict
        statistic : float, p_value : float, table : DataFrame
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Create decile groups based on predicted probability
    try:
        groups = pd.qcut(y_pred, q=n_groups, duplicates="drop")
    except ValueError:
        groups = pd.cut(y_pred, bins=n_groups)

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups})
    table = (
        df.groupby("group", observed=True)
        .agg(
            n=("y_true", "count"),
            observed_events=("y_true", "sum"),
            predicted_prob=("y_pred", "mean"),
        )
        .reset_index()
    )
    table["expected_events"] = table["n"] * table["predicted_prob"]
    table["observed_non_events"] = table["n"] - table["observed_events"]
    table["expected_non_events"] = table["n"] * (1 - table["predicted_prob"])

    # HL statistic
    hl_stat = 0.0
    for _, row in table.iterrows():
        if row["expected_events"] > 0:
            hl_stat += (row["observed_events"] - row["expected_events"]) ** 2 / row[
                "expected_events"
            ]
        if row["expected_non_events"] > 0:
            hl_stat += (
                row["observed_non_events"] - row["expected_non_events"]
            ) ** 2 / row["expected_non_events"]

    dof = len(table) - 2
    p_value = 1 - stats.chi2.cdf(hl_stat, dof) if dof > 0 else 1.0

    return {
        "statistic": float(hl_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": dof,
        "table": table,
    }


def compute_calibration_by_decile(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute calibration table by score decile with bootstrap CIs.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_pred : array-like
        Predicted probabilities.
    n_bins : int
        Number of bins.
    n_bootstrap : int
        Bootstrap iterations for 95% CI on observed_default_rate.
    confidence : float
        Confidence level (default 0.95).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    DataFrame
        Columns: decile, n, observed_default_rate, predicted_default_rate, ratio,
                 obs_ci_lower, obs_ci_upper
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    try:
        deciles = pd.qcut(y_pred, q=n_bins, duplicates="drop", labels=False)
    except ValueError:
        deciles = pd.cut(y_pred, bins=n_bins, labels=False)

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "decile": deciles})
    result = (
        df.groupby("decile")
        .agg(
            n=("y_true", "count"),
            observed_default_rate=("y_true", "mean"),
            predicted_default_rate=("y_pred", "mean"),
        )
        .reset_index()
    )
    result["ratio"] = result["observed_default_rate"] / result[
        "predicted_default_rate"
    ].replace(0, np.nan)

    # Bootstrap 95% CIs for observed_default_rate per decile.
    # Bin boundaries are fixed from the point estimate; only rows are resampled.
    rng = np.random.RandomState(random_state)
    alpha = (1.0 - confidence) / 2.0
    decile_labels = result["decile"].tolist()
    boot_means: dict = {d: [] for d in decile_labels}

    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(df), size=len(df))
        samp = df.iloc[idx]
        grp_means = samp.groupby("decile")["y_true"].mean()
        for d in decile_labels:
            if d in grp_means.index:
                boot_means[d].append(float(grp_means[d]))

    result["obs_ci_lower"] = [
        float(np.percentile(boot_means[d], 100.0 * alpha)) if boot_means[d] else np.nan
        for d in decile_labels
    ]
    result["obs_ci_upper"] = [
        float(np.percentile(boot_means[d], 100.0 * (1.0 - alpha))) if boot_means[d] else np.nan
        for d in decile_labels
    ]

    return result


# ---------------------------------------------------------------------------
# Stability metrics: PSI, CSI, VDI
# ---------------------------------------------------------------------------


def _safe_psi_term(p: float, q: float) -> float:
    """Single PSI term with safe log handling."""
    if p == 0 and q == 0:
        return 0.0
    if p == 0:
        p = 1e-6
    if q == 0:
        q = 1e-6
    return (p - q) * np.log(p / q)


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    bins: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute Population Stability Index.

    Parameters
    ----------
    expected : array-like
        Training / baseline distribution (scores or feature values).
    actual : array-like
        Test / monitoring distribution.
    n_bins : int
        Number of equal-width bins.
    bins : array-like, optional
        Pre-defined bin edges. If provided, n_bins is ignored.

    Returns
    -------
    dict
        psi : float, bin_table : DataFrame with per-bin PSI
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Remove NaN
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if bins is None:
        # Use expected distribution to define bins
        bins = np.linspace(
            min(expected.min(), actual.min()) - 1e-6,
            max(expected.max(), actual.max()) + 1e-6,
            n_bins + 1,
        )

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()

    psi_bins = np.array(
        [_safe_psi_term(a, e) for a, e in zip(act_pct, exp_pct)]
    )
    psi_total = float(psi_bins.sum())

    bin_table = pd.DataFrame(
        {
            "bin_lower": bins[:-1],
            "bin_upper": bins[1:],
            "expected_pct": exp_pct,
            "actual_pct": act_pct,
            "psi_bin": psi_bins,
        }
    )

    return {"psi": psi_total, "bin_table": bin_table}


def compute_csi(
    train_feature: np.ndarray,
    test_feature: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute Characteristic Stability Index for a single feature.

    Same methodology as PSI but applied to input feature distributions.

    Parameters
    ----------
    train_feature : array-like
        Training distribution of a feature.
    test_feature : array-like
        Test distribution of the same feature.
    n_bins : int
        Number of bins.

    Returns
    -------
    dict
        csi : float, bin_table : DataFrame
    """
    result = compute_psi(train_feature, test_feature, n_bins=n_bins)
    return {"csi": result["psi"], "bin_table": result["bin_table"]}


def compute_vdi(
    train_feature: np.ndarray,
    test_feature: np.ndarray,
) -> dict[str, Any]:
    """Compute Variable Deviation Index.

    Measures shift in distribution via summary statistics (mean, std, skewness).

    Parameters
    ----------
    train_feature : array-like
        Training distribution of a feature.
    test_feature : array-like
        Test distribution of the same feature.

    Returns
    -------
    dict
        vdi : float, components : dict with mean_shift, std_shift, skew_shift
    """
    train_feature = np.asarray(train_feature, dtype=float)
    test_feature = np.asarray(test_feature, dtype=float)

    train_feature = train_feature[~np.isnan(train_feature)]
    test_feature = test_feature[~np.isnan(test_feature)]

    train_mean = np.mean(train_feature)
    test_mean = np.mean(test_feature)
    train_std = np.std(train_feature)
    test_std = np.std(test_feature)

    # Mean shift as fraction of training std
    mean_shift = (
        abs(test_mean - train_mean) / train_std if train_std > 0 else 0.0
    )

    # Std ratio
    std_shift = abs(test_std / train_std - 1.0) if train_std > 0 else 0.0

    # Skewness change
    train_skew = float(stats.skew(train_feature))
    test_skew = float(stats.skew(test_feature))
    skew_shift = abs(test_skew - train_skew)

    # VDI as weighted combination (industry standard weighting)
    vdi = 0.50 * mean_shift + 0.30 * std_shift + 0.20 * skew_shift

    return {
        "vdi": float(vdi),
        "components": {
            "mean_shift": float(mean_shift),
            "std_shift": float(std_shift),
            "skew_shift": float(skew_shift),
            "train_mean": float(train_mean),
            "test_mean": float(test_mean),
            "train_std": float(train_std),
            "test_std": float(test_std),
        },
    }


# ---------------------------------------------------------------------------
# RAG status framework
# ---------------------------------------------------------------------------

# RAG symbols for display
RAG_SYMBOLS = {"GREEN": "\u2713", "AMBER": "\u25B3", "RED": "\u2717"}


def rag_status(
    value: float,
    metric_type: str,
) -> Literal["GREEN", "AMBER", "RED"]:
    """Determine RAG status for a metric value.

    Parameters
    ----------
    value : float
        Metric value to evaluate.
    metric_type : str
        Key into RAG_THRESHOLDS (e.g. 'gini_scorecard', 'psi', 'brier').

    Returns
    -------
    str
        'GREEN', 'AMBER', or 'RED'.
    """
    if metric_type not in RAG_THRESHOLDS:
        raise ValueError(
            f"Unknown metric_type '{metric_type}'. "
            f"Valid types: {list(RAG_THRESHOLDS.keys())}"
        )

    thresholds = RAG_THRESHOLDS[metric_type]
    green_lo, green_hi = thresholds["green"]
    amber_lo, amber_hi = thresholds["amber"]

    if green_lo <= value < green_hi:
        return "GREEN"
    if amber_lo <= value < amber_hi:
        return "AMBER"
    return "RED"


def rag_status_table(
    metrics: dict[str, tuple[float, str]],
) -> pd.DataFrame:
    """Build a RAG status table from a dictionary of metrics.

    Parameters
    ----------
    metrics : dict
        Keys are metric display names. Values are (value, metric_type) tuples
        where metric_type is a key in RAG_THRESHOLDS.

    Returns
    -------
    DataFrame
        Columns: Metric, Value, RAG_Status, Symbol, Threshold_Green, Threshold_Amber
    """
    rows = []
    for name, (value, metric_type) in metrics.items():
        status = rag_status(value, metric_type)
        thresholds = RAG_THRESHOLDS[metric_type]
        green_lo, green_hi = thresholds["green"]
        amber_lo, amber_hi = thresholds["amber"]

        # Format threshold description
        if green_hi == float("inf"):
            green_desc = f">= {green_lo}"
            amber_desc = f"{amber_lo} - {amber_hi}"
            red_desc = f"< {amber_lo}"
        else:
            green_desc = f"< {green_hi}"
            amber_desc = f"{amber_lo} - {amber_hi}"
            red_desc = f">= {amber_hi}"

        rows.append(
            {
                "Metric": name,
                "Value": round(value, 4),
                "RAG_Status": status,
                "Symbol": RAG_SYMBOLS[status],
                "Threshold_Green": green_desc,
                "Threshold_Amber": amber_desc,
                "Threshold_Red": red_desc,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Monitoring report generator
# ---------------------------------------------------------------------------


def generate_monitoring_report(
    results: dict[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a formatted monitoring report summary.

    Parameters
    ----------
    results : dict
        Nested dict of validation results. Expected keys:
        - 'discrimination': {auc, gini, ks, cap_ar}
        - 'calibration': {brier, hosmer_lemeshow_p, hosmer_lemeshow_stat}
        - 'stability': {psi_by_period: dict, avg_csi, max_csi_feature}
        - 'ead': {mape, r2}
        - 'lgd': {mae, stage1_auc, portfolio_avg}
    output_path : str or Path, optional
        If provided, save JSON to this path.

    Returns
    -------
    dict
        Structured report with RAG assessments.
    """
    report: dict[str, Any] = {
        "report_type": "Quarterly Model Monitoring Report",
        "sections": {},
    }

    # Discrimination
    if "discrimination" in results:
        d = results["discrimination"]
        report["sections"]["discrimination"] = {
            "metrics": d,
            "assessment": (
                "Model discrimination within acceptable range."
                if d.get("gini", 0) >= 0.36
                else "Model discrimination below minimum threshold."
            ),
        }

    # Calibration
    if "calibration" in results:
        c = results["calibration"]
        report["sections"]["calibration"] = {
            "metrics": c,
            "assessment": (
                "Model calibration adequate."
                if c.get("brier", 1) < 0.20
                else "Model calibration requires review."
            ),
        }

    # Stability
    if "stability" in results:
        s = results["stability"]
        report["sections"]["stability"] = {
            "metrics": s,
            "assessment": (
                "Population stable across periods."
                if all(
                    v < 0.25
                    for v in s.get("psi_by_period", {}).values()
                )
                else "Population shift detected in one or more periods."
            ),
        }

    # EAD / LGD
    if "ead" in results:
        report["sections"]["ead"] = {"metrics": results["ead"]}
    if "lgd" in results:
        report["sections"]["lgd"] = {"metrics": results["lgd"]}

    # Overall
    all_rag = []
    for section in report["sections"].values():
        for k, v in section.get("metrics", {}).items():
            if isinstance(v, str) and v in ("GREEN", "AMBER", "RED"):
                all_rag.append(v)

    if "RED" in all_rag:
        report["overall_status"] = "ACTION REQUIRED"
    elif "AMBER" in all_rag:
        report["overall_status"] = "MONITOR"
    else:
        report["overall_status"] = "SATISFACTORY"

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert non-serializable types
        def _convert(obj: Any) -> Any:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            return obj

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=_convert)

    return report


# ---------------------------------------------------------------------------
# Out-of-time validation helpers
# ---------------------------------------------------------------------------


def compute_oot_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    period_label: str = "",
) -> dict[str, float]:
    """Compute standard discrimination metrics for a single OOT period.

    Parameters
    ----------
    y_true : array-like
        Binary labels.
    y_pred : array-like
        Predicted probabilities.
    period_label : str
        Label for the period (e.g. '2016', '2017').

    Returns
    -------
    dict
        period, auc, gini, ks, brier, n, default_rate
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    auc_val = roc_auc_score(y_true, y_pred)
    gini_val = 2 * auc_val - 1
    ks_result = compute_ks(y_true, y_pred)
    brier_val = brier_score_loss(y_true, y_pred)

    return {
        "period": period_label,
        "auc": float(auc_val),
        "gini": float(gini_val),
        "ks": float(ks_result["ks_statistic"]),
        "brier": float(brier_val),
        "n": len(y_true),
        "default_rate": float(y_true.mean()),
    }


# ---------------------------------------------------------------------------
# Backtesting helper
# ---------------------------------------------------------------------------


def compute_backtesting(
    df: pd.DataFrame,
    vintage_col: str = "issue_year",
    y_true_col: str = "default",
    y_pred_col: str = "pd_pred",
) -> pd.DataFrame:
    """Compare predicted vs actual cumulative default rates by vintage.

    Parameters
    ----------
    df : DataFrame
        Must contain vintage_col, y_true_col, y_pred_col columns.
    vintage_col : str
        Column with vintage year.
    y_true_col : str
        Actual default indicator (0/1).
    y_pred_col : str
        Model-predicted PD.

    Returns
    -------
    DataFrame
        Columns: vintage, n_loans, actual_default_rate, predicted_default_rate,
                 ratio, difference
    """
    result = (
        df.groupby(vintage_col)
        .agg(
            n_loans=(y_true_col, "count"),
            actual_default_rate=(y_true_col, "mean"),
            predicted_default_rate=(y_pred_col, "mean"),
        )
        .reset_index()
    )
    result.rename(columns={vintage_col: "vintage"}, inplace=True)
    result["ratio"] = result["predicted_default_rate"] / result[
        "actual_default_rate"
    ].replace(0, np.nan)
    result["difference"] = (
        result["predicted_default_rate"] - result["actual_default_rate"]
    )

    return result
