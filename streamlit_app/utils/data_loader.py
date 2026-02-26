"""Cached data loaders for all CSV, JSON, and Parquet files used by the Streamlit app."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Base paths ──
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = APP_DIR.parent
RESULTS = PROJECT_DIR / "data" / "results"
PROCESSED = PROJECT_DIR / "data" / "processed"
MODELS = PROJECT_DIR / "models"


def _resolve(filename: str, subdir: str | None = None) -> Path:
    """Find a results file, checking subdirectory first then top-level fallback.

    This handles cases where notebooks save outputs into subdirectories
    (e.g. 07_ecl_flow_rates/) but top-level copies also exist.
    """
    if subdir:
        primary = RESULTS / subdir / filename
        if primary.exists():
            return primary
    fallback = RESULTS / filename
    if fallback.exists():
        return fallback
    # Return primary path for better error messages
    return (RESULTS / subdir / filename) if subdir else fallback


def _read_csv_safe(filename: str, subdir: str | None = None, **kwargs) -> pd.DataFrame:
    """Read CSV with subdirectory/top-level fallback and OSError handling."""
    path = _resolve(filename, subdir)
    try:
        return pd.read_csv(path, **kwargs)
    except OSError:
        # Try fallback if primary failed
        alt = RESULTS / filename if subdir else None
        if alt and alt.exists():
            return pd.read_csv(alt, **kwargs)
        raise


def _read_json_safe(filename: str, subdir: str | None = None) -> dict:
    """Read JSON with subdirectory/top-level fallback."""
    path = _resolve(filename, subdir)
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        alt = RESULTS / filename if subdir else None
        if alt and alt.exists():
            with open(alt) as f:
                return json.load(f)
        raise


# ── Strategy & ECL ──

@st.cache_data
def load_strategy_analysis() -> pd.DataFrame:
    return _read_csv_safe("strategy_analysis.csv")

@st.cache_data
def load_ecl_postfeg() -> pd.DataFrame:
    return _read_csv_safe("ecl_postfeg.csv")

@st.cache_data
def load_ecl_central() -> pd.DataFrame:
    return _read_csv_safe("ecl_central.csv")

@st.cache_data
def load_ecl_prefeg() -> pd.DataFrame:
    try:
        return _read_csv_safe("ecl_prefeg.csv", "07_ecl_flow_rates")
    except (OSError, FileNotFoundError):
        return _read_csv_safe("ecl_prefeg_v2.csv", "07_ecl_flow_rates")

@st.cache_data
def load_ecl_by_scenario() -> pd.DataFrame:
    return _read_csv_safe("ecl_by_scenario.csv")


# ── Flow Rates & Receivables ──

@st.cache_data
def load_receivables_tracker() -> pd.DataFrame:
    df = _read_csv_safe("receivables_tracker.csv", "07_ecl_flow_rates")
    df["month_date"] = pd.to_datetime(df["month_date"])
    return df

@st.cache_data
def load_roll_rates() -> pd.DataFrame:
    df = _read_csv_safe("roll_rates.csv", "07_ecl_flow_rates")
    df["month_date"] = pd.to_datetime(df["month_date"])
    return df

@st.cache_data
def load_flow_through_rate() -> pd.DataFrame:
    df = _read_csv_safe("flow_through_rate.csv", "07_ecl_flow_rates")
    df["month_date"] = pd.to_datetime(df["month_date"])
    return df

@st.cache_data
def load_flow_rates_extend() -> pd.DataFrame:
    df = _read_csv_safe("flow_rates_extend.csv", "07_ecl_flow_rates")
    df["month_date"] = pd.to_datetime(df["month_date"])
    return df

@st.cache_data
def load_flow_rates_cecl() -> pd.DataFrame:
    df = _read_csv_safe("flow_rates_cecl.csv", "07_ecl_flow_rates")
    df["month_date"] = pd.to_datetime(df["month_date"])
    return df

@st.cache_data
def load_flow_rates_by_scenario() -> pd.DataFrame:
    return _read_csv_safe("flow_rates_by_scenario.csv")


# ── Vintage ──
_VINTAGE_CACHE_V = "v3"

@st.cache_data
def load_vintage_curves(_v=_VINTAGE_CACHE_V) -> pd.DataFrame:
    return _read_csv_safe("vintage_curves.csv", "07_ecl_flow_rates")

@st.cache_data
def load_vintage_curves_mob(_v=_VINTAGE_CACHE_V) -> pd.DataFrame:
    """Load MOB-indexed vintage curves. If file is missing, generate from loan data."""
    try:
        return _read_csv_safe("vintage_curves_mob.csv", "07_ecl_flow_rates")
    except (OSError, FileNotFoundError):
        return _generate_vintage_mob_data()

@st.cache_data
def load_marginal_pd_by_grade(_v=_VINTAGE_CACHE_V) -> pd.DataFrame:
    """Load marginal PD by grade. If file is missing, generate from loan data."""
    try:
        return _read_csv_safe("marginal_pd_by_grade.csv", "07_ecl_flow_rates")
    except (OSError, FileNotFoundError):
        return _generate_marginal_pd_data()

@st.cache_data
def load_seasoning_pattern(_v=_VINTAGE_CACHE_V) -> pd.DataFrame:
    """Load seasoning pattern. If file is missing, generate from vintage MOB data."""
    try:
        return _read_csv_safe("seasoning_pattern.csv", "07_ecl_flow_rates")
    except (OSError, FileNotFoundError):
        return _generate_seasoning_data()


def _generate_vintage_mob_data() -> pd.DataFrame:
    """Generate vintage curves by MOB from the cleaned loan data."""
    import numpy as np
    try:
        loans = load_loans_sample()
    except Exception:
        return pd.DataFrame(columns=["vintage_year", "mob", "cum_default_rate", "cum_loss_rate"])

    if "issue_d" not in loans.columns:
        return pd.DataFrame(columns=["vintage_year", "mob", "cum_default_rate", "cum_loss_rate"])

    default_col = "default" if "default" in loans.columns else "default_flag"
    loans = loans.dropna(subset=["issue_d"])
    loans["vintage_year"] = loans["issue_d"].dt.year

    rows = []
    for vy, grp in loans.groupby("vintage_year"):
        n_total = len(grp)
        n_defaults = grp[default_col].sum()
        total_funded = grp["funded_amnt"].sum()
        default_funded = grp.loc[grp[default_col] == 1, "funded_amnt"].sum()

        # Distribute defaults across MOBs using a typical seasoning curve
        # (beta distribution peaking around MOB 18-24 for 36-month loans)
        max_mob = 60
        mob_weights = np.array([
            (m / 24.0) ** 1.5 * np.exp(-0.03 * m) for m in range(1, max_mob + 1)
        ])
        mob_weights = mob_weights / mob_weights.sum()

        cum_defaults = 0.0
        cum_loss = 0.0
        for mob in range(1, max_mob + 1):
            cum_defaults += n_defaults * mob_weights[mob - 1]
            cum_loss += default_funded * mob_weights[mob - 1]
            rows.append({
                "vintage_year": int(vy),
                "mob": mob,
                "cum_default_rate": cum_defaults / max(n_total, 1),
                "cum_loss_rate": cum_loss / max(total_funded, 1),
            })

    return pd.DataFrame(rows)


def _generate_marginal_pd_data() -> pd.DataFrame:
    """Generate marginal PD by grade from loan data."""
    import numpy as np
    try:
        loans = load_loans_sample()
    except Exception:
        return pd.DataFrame(columns=["vintage_year", "grade", "mob", "marginal_pd"])

    if "issue_d" not in loans.columns or "grade" not in loans.columns:
        return pd.DataFrame(columns=["vintage_year", "grade", "mob", "marginal_pd"])

    default_col = "default" if "default" in loans.columns else "default_flag"
    loans = loans.dropna(subset=["issue_d", "grade"])
    loans["vintage_year"] = loans["issue_d"].dt.year

    rows = []
    for (vy, grade), grp in loans.groupby(["vintage_year", "grade"]):
        n_total = len(grp)
        n_defaults = grp[default_col].sum()
        lifetime_pd = n_defaults / max(n_total, 1)

        max_mob = 60
        # Hazard rate shape varies by grade (riskier grades peak earlier)
        grade_shift = {"A": 0.8, "B": 0.9, "C": 1.0, "D": 1.1, "E": 1.2, "F": 1.3, "G": 1.4}
        shift = grade_shift.get(grade, 1.0)
        mob_weights = np.array([
            (m / (24.0 / shift)) ** 1.5 * np.exp(-0.03 * shift * m)
            for m in range(1, max_mob + 1)
        ])
        mob_weights = mob_weights / mob_weights.sum()

        for mob in range(1, max_mob + 1):
            rows.append({
                "vintage_year": int(vy),
                "grade": grade,
                "mob": mob,
                "marginal_pd": lifetime_pd * mob_weights[mob - 1],
            })

    return pd.DataFrame(rows)


def _generate_seasoning_data() -> pd.DataFrame:
    """Generate seasoning pattern from vintage MOB data."""
    try:
        mob_data = load_vintage_curves_mob()
    except Exception:
        return pd.DataFrame(columns=["mob", "avg_cum_default_rate"])

    if len(mob_data) == 0:
        return pd.DataFrame(columns=["mob", "avg_cum_default_rate"])

    seasoning = mob_data.groupby("mob")["cum_default_rate"].mean().reset_index()
    seasoning.columns = ["mob", "avg_cum_default_rate"]
    return seasoning


# ── Monthly ECL ──

@st.cache_data
def load_monthly_ecl_simple() -> pd.DataFrame:
    df = _read_csv_safe("monthly_ecl_simple.csv", "07_ecl_flow_rates")
    df["month_date"] = pd.to_datetime(df["month_date"])
    return df

@st.cache_data
def load_monthly_ecl_dcf() -> pd.DataFrame:
    df = _read_csv_safe("monthly_ecl_dcf.csv", "07_ecl_flow_rates")
    df["month_date"] = pd.to_datetime(df["month_date"])
    return df

@st.cache_data
def load_alll_tracker() -> pd.DataFrame:
    df = _read_csv_safe("alll_tracker_simple_ecl.csv", "07_ecl_flow_rates")
    df["month_date"] = pd.to_datetime(df["month_date"])
    return df


# ── Macro Scenarios ──

@st.cache_data
def load_macro_scenarios() -> dict:
    return _read_json_safe("macro_scenarios.json")

@st.cache_data
def load_forward_macro_paths() -> pd.DataFrame:
    return _read_csv_safe("forward_macro_paths.csv")

@st.cache_data
def load_quarterly_stress_multipliers() -> pd.DataFrame:
    return _read_csv_safe("quarterly_stress_multipliers.csv")

@st.cache_data
def load_sensitivity_results() -> dict:
    return _read_json_safe("sensitivity_results.json")

@st.cache_data
def load_multi_factor_elasticities() -> dict:
    return _read_json_safe("multi_factor_elasticities.json")


# ── Validation / RAG ──

@st.cache_data
def load_rag_status() -> pd.DataFrame:
    return _read_csv_safe("rag_status_table.csv", "08_validation")

@st.cache_data
def load_pd_metrics() -> dict:
    return _read_json_safe("pd_scorecard_metrics.json", "08_validation")

@st.cache_data
def load_psi_by_period() -> pd.DataFrame:
    return _read_csv_safe("psi_by_period.csv", "08_validation")

@st.cache_data
def load_csi_by_feature() -> pd.DataFrame:
    return _read_csv_safe("csi_by_feature.csv", "08_validation")


# ── Loan-level data (full cleaned dataset, cached for performance) ──

@st.cache_data
def load_loans_sample(n: int = 0) -> pd.DataFrame:
    """Load full loan data for dashboard. Uses loans_cleaned.parquet (all 1.35M loans)."""
    wanted = [
        "loan_amnt", "funded_amnt", "int_rate", "grade", "sub_grade",
        "term", "purpose", "issue_d", "default", "default_flag",
        "annual_inc", "dti", "fico_range_low", "addr_state",
    ]
    source = PROCESSED / "loans_cleaned.parquet"
    df = None

    def _try_read_parquet(path: Path, select_cols: bool = True) -> "pd.DataFrame | None":
        """Try multiple strategies to read a parquet file."""
        if not path.exists():
            return None
        # Strategy 1: read with column selection (pyarrow)
        if select_cols:
            try:
                import pyarrow.parquet as pq
                schema = pq.read_schema(path)
                cols = [c for c in wanted if c in schema.names]
                return pd.read_parquet(path, columns=cols, engine="pyarrow")
            except Exception:
                pass
        # Strategy 2: read full file then select (pyarrow)
        try:
            tmp = pd.read_parquet(path, engine="pyarrow")
            cols = [c for c in wanted if c in tmp.columns]
            return tmp[cols]
        except Exception:
            pass
        # Strategy 3: try fastparquet engine
        try:
            tmp = pd.read_parquet(path, engine="fastparquet")
            cols = [c for c in wanted if c in tmp.columns]
            return tmp[cols]
        except Exception:
            pass
        # Strategy 4: read as plain pandas (auto engine)
        try:
            tmp = pd.read_parquet(path)
            cols = [c for c in wanted if c in tmp.columns]
            return tmp[cols]
        except Exception:
            pass
        return None

    # Try loans_cleaned.parquet first
    df = _try_read_parquet(source)

    # Fallback: concatenate train/val/test parquets
    if df is None:
        parts = []
        for name in ["train.parquet", "val.parquet", "test.parquet"]:
            part = _try_read_parquet(PROCESSED / name)
            if part is not None and len(part) > 0:
                parts.append(part)
        if parts:
            df = pd.concat(parts, ignore_index=True)

    # Final fallback: try CSV files
    if df is None:
        for csv_name in ["loans_dashboard.csv", "loans_cleaned.csv"]:
            csv_path = PROCESSED / csv_name
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, usecols=lambda c: c in wanted)
                    break
                except Exception:
                    continue

    if df is None:
        raise FileNotFoundError(f"No loan data found in {PROCESSED}")

    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
        df["vintage_year"] = df["issue_d"].dt.year
    return df


# ── Prepayment / Liquidation ──

@st.cache_data
def load_prepayment_rates() -> pd.DataFrame:
    """Load CPR rates by term × grade × vintage for liquidation factors."""
    try:
        return _read_csv_safe("prepayment_rates.csv", "055_prepayment")
    except (OSError, FileNotFoundError):
        return pd.DataFrame()

@st.cache_data
def load_liquidation_curves() -> pd.DataFrame:
    """Load survival/liquidation curves by vintage and grade."""
    try:
        return _read_csv_safe("liquidation_curves.csv", "055_prepayment")
    except (OSError, FileNotFoundError):
        return pd.DataFrame()
