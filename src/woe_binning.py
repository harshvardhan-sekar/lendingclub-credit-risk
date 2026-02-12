"""
WOE/IV Binning Engine for Credit Risk Scorecard Development.

Wraps optbinning.OptimalBinning to provide a scikit-learn-compatible
interface for Weight of Evidence (WoE) transformation and Information
Value (IV) computation across a full feature set.

Usage:
    from src.woe_binning import WOEBinner
    binner = WOEBinner()
    binner.fit(X_train, y_train)
    X_woe = binner.transform(X_train)
    iv_table = binner.iv_summary()
"""
from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optbinning import OptimalBinning

# ── IV interpretation thresholds ──────────────────────────────────────────────
IV_THRESHOLDS = {
    "not_predictive": 0.02,
    "weak": 0.10,
    "medium": 0.30,
    "strong": 0.50,
}


def _classify_iv(iv: float) -> str:
    """Classify IV value into selection category.

    Args:
        iv: Information Value for a feature.

    Returns:
        Category string: 'drop', 'weak', 'medium', 'strong', or 'suspicious'.
    """
    if iv < IV_THRESHOLDS["not_predictive"]:
        return "drop"
    elif iv < IV_THRESHOLDS["weak"]:
        return "weak"
    elif iv < IV_THRESHOLDS["medium"]:
        return "medium"
    elif iv < IV_THRESHOLDS["strong"]:
        return "strong"
    else:
        return "suspicious"


class WOEBinner:
    """Weight of Evidence binner for credit risk scorecard features.

    Fits optimal bins on training data using ``optbinning.OptimalBinning``
    and stores per-feature bin tables, WoE mappings, and IV values.

    Parameters:
        max_n_bins: Maximum number of bins for continuous features.
        min_bin_size: Minimum fraction of records per bin.
        monotonic_trend: Enforce monotonic WoE trend.  ``None`` lets
            optbinning decide automatically.  Use ``'ascending'``,
            ``'descending'``, or ``'auto'`` to constrain.
        cat_cutoff: Minimum frequency for a categorical level to get
            its own bin; rarer levels are grouped.
        random_state: Seed for reproducibility.

    Attributes:
        binners_: dict mapping feature name → fitted OptimalBinning object.
        iv_: dict mapping feature name → IV float.
        bin_tables_: dict mapping feature name → DataFrame of bin statistics.
        feature_dtypes_: dict mapping feature name → 'numerical' | 'categorical'.
    """

    def __init__(
        self,
        max_n_bins: int = 10,
        min_bin_size: float = 0.05,
        monotonic_trend: Optional[str] = "auto",
        cat_cutoff: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.monotonic_trend = monotonic_trend
        self.cat_cutoff = cat_cutoff
        self.random_state = random_state

        # Populated after fit()
        self.binners_: dict[str, OptimalBinning] = {}
        self.iv_: dict[str, float] = {}
        self.bin_tables_: dict[str, pd.DataFrame] = {}
        self.feature_dtypes_: dict[str, str] = {}
        self._fitted = False

    # ── Core API ──────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[list[str]] = None,
    ) -> "WOEBinner":
        """Fit optimal bins on training data.

        Args:
            X: Feature DataFrame (training set only).
            y: Binary target (0/1).
            feature_names: Subset of columns to bin.  If ``None``, all
                columns in *X* are used.

        Returns:
            self
        """
        if feature_names is None:
            feature_names = X.columns.tolist()

        self.binners_ = {}
        self.iv_ = {}
        self.bin_tables_ = {}
        self.feature_dtypes_ = {}

        failed: list[str] = []

        for feat in feature_names:
            if feat not in X.columns:
                warnings.warn(f"Feature '{feat}' not found in X — skipping.")
                continue

            col = X[feat].copy()
            dtype = self._infer_dtype(col)
            self.feature_dtypes_[feat] = dtype

            try:
                binner = self._create_binner(feat, dtype)
                x_clean, y_clean = self._prepare_column(col, y)

                if len(x_clean) < 100:
                    warnings.warn(
                        f"Feature '{feat}' has < 100 non-null values — skipping."
                    )
                    failed.append(feat)
                    continue

                binner.fit(x_clean.values, y_clean.values)

                if binner.status != "OPTIMAL":
                    # Accept sub-optimal solutions that still converged
                    if binner.status not in ("OPTIMAL", "FEASIBLE"):
                        warnings.warn(
                            f"Feature '{feat}' binning status: {binner.status}"
                        )

                self.binners_[feat] = binner
                table = self._build_bin_table(binner, feat)
                self.bin_tables_[feat] = table
                self.iv_[feat] = table["iv"].sum()

            except Exception as exc:
                warnings.warn(f"Feature '{feat}' binning failed: {exc}")
                failed.append(feat)

        self._fitted = True
        if failed:
            print(f"[WOEBinner] {len(failed)} features failed: {failed[:10]}{'...' if len(failed) > 10 else ''}")
        print(
            f"[WOEBinner] Successfully fitted {len(self.binners_)} / "
            f"{len(feature_names)} features."
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace feature values with their WoE scores.

        Args:
            X: DataFrame with original feature values.

        Returns:
            DataFrame with WoE-transformed values for fitted features.
        """
        self._check_fitted()
        result = pd.DataFrame(index=X.index)

        for feat, binner in self.binners_.items():
            if feat not in X.columns:
                warnings.warn(f"Feature '{feat}' not in X during transform — skipping.")
                continue
            col = X[feat].copy()
            x_clean = self._coerce_column(col, feat)
            result[feat] = binner.transform(x_clean.values, metric="woe")

        return result

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Fit on training data and return WoE-transformed features.

        Args:
            X: Feature DataFrame (training set only).
            y: Binary target (0/1).
            feature_names: Subset of columns to bin.

        Returns:
            WoE-transformed DataFrame.
        """
        self.fit(X, y, feature_names)
        return self.transform(X)

    # ── Summaries ─────────────────────────────────────────────────────────

    def iv_summary(self) -> pd.DataFrame:
        """Return a ranked IV summary table for all fitted features.

        Returns:
            DataFrame with columns: feature, iv, selection_status, dtype.
        """
        self._check_fitted()
        rows = []
        for feat in self.iv_:
            rows.append(
                {
                    "feature": feat,
                    "iv": round(self.iv_[feat], 6),
                    "selection_status": _classify_iv(self.iv_[feat]),
                    "dtype": self.feature_dtypes_.get(feat, "unknown"),
                }
            )
        df = (
            pd.DataFrame(rows)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )
        return df

    def bin_table(self, feature: str) -> pd.DataFrame:
        """Return the detailed bin table for a single feature.

        Args:
            feature: Name of the feature.

        Returns:
            DataFrame with bin, count, event_count, non_event_count,
            event_rate, woe, iv columns.
        """
        self._check_fitted()
        if feature not in self.bin_tables_:
            raise KeyError(f"Feature '{feature}' not found in fitted binners.")
        return self.bin_tables_[feature]

    # ── Diagnostics ───────────────────────────────────────────────────────

    def check_monotonicity(self, feature: str) -> dict:
        """Check whether event rate is monotonic across bins.

        Args:
            feature: Name of the feature.

        Returns:
            Dict with keys 'monotonic' (bool), 'direction' (str),
            and 'violations' (list of bin indices where breaks occur).
        """
        self._check_fitted()
        table = self.bin_table(feature)
        # Exclude Missing and Special bins
        mask = ~table["bin"].astype(str).isin(["Missing", "Special"])
        rates = table.loc[mask, "event_rate"].values

        if len(rates) < 2:
            return {"monotonic": True, "direction": "n/a", "violations": []}

        diffs = np.diff(rates)
        increasing_violations = np.where(diffs < -1e-9)[0].tolist()
        decreasing_violations = np.where(diffs > 1e-9)[0].tolist()

        if len(increasing_violations) == 0:
            return {"monotonic": True, "direction": "ascending", "violations": []}
        if len(decreasing_violations) == 0:
            return {"monotonic": True, "direction": "descending", "violations": []}

        # Determine primary direction by majority of diffs
        if np.sum(diffs > 0) >= np.sum(diffs < 0):
            return {
                "monotonic": False,
                "direction": "mostly_ascending",
                "violations": increasing_violations,
            }
        return {
            "monotonic": False,
            "direction": "mostly_descending",
            "violations": decreasing_violations,
        }

    def monotonicity_report(self) -> pd.DataFrame:
        """Generate monotonicity report for all fitted features.

        Returns:
            DataFrame with feature, monotonic, direction, n_violations.
        """
        self._check_fitted()
        rows = []
        for feat in self.binners_:
            result = self.check_monotonicity(feat)
            rows.append(
                {
                    "feature": feat,
                    "monotonic": result["monotonic"],
                    "direction": result["direction"],
                    "n_violations": len(result["violations"]),
                }
            )
        return pd.DataFrame(rows).sort_values("n_violations", ascending=False)

    # ── Plotting ──────────────────────────────────────────────────────────

    def plot_woe(
        self,
        feature: str,
        figsize: tuple[int, int] = (10, 5),
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot WoE values and event rate by bin for a feature.

        Args:
            feature: Feature name.
            figsize: Figure size if creating new figure.
            ax: Optional axes to plot on.

        Returns:
            matplotlib Figure.
        """
        self._check_fitted()
        table = self.bin_table(feature)
        iv_val = self.iv_[feature]

        if ax is None:
            fig, ax1 = plt.subplots(figsize=figsize)
        else:
            ax1 = ax
            fig = ax.get_figure()

        x_pos = np.arange(len(table))
        labels = table["bin"].astype(str).values

        # Bar: WoE values
        colors = ["#d62728" if w > 0 else "#2ca02c" for w in table["woe"]]
        ax1.bar(x_pos, table["woe"], color=colors, alpha=0.7, label="WoE")
        ax1.set_ylabel("WoE", fontsize=11)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax1.axhline(0, color="black", linewidth=0.5, linestyle="--")

        # Line: event rate on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(
            x_pos, table["event_rate"], "ko-", markersize=5, label="Event Rate"
        )
        ax2.set_ylabel("Event Rate (Bad Rate)", fontsize=11)

        ax1.set_title(f"{feature}  |  IV = {iv_val:.4f}", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Bin", fontsize=11)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

        fig.tight_layout()
        return fig

    def plot_top_features(
        self,
        n: int = 20,
        figsize: tuple[int, int] = (12, 8),
    ) -> plt.Figure:
        """Horizontal bar chart of top-N features by IV.

        Args:
            n: Number of features to display.
            figsize: Figure dimensions.

        Returns:
            matplotlib Figure.
        """
        self._check_fitted()
        summary = self.iv_summary().head(n).iloc[::-1]

        fig, ax = plt.subplots(figsize=figsize)
        colors = [
            {"drop": "#bdbdbd", "weak": "#fdae61", "medium": "#2ca02c",
             "strong": "#1f77b4", "suspicious": "#d62728"}
            .get(s, "#bdbdbd")
            for s in summary["selection_status"]
        ]
        ax.barh(summary["feature"], summary["iv"], color=colors)
        ax.set_xlabel("Information Value (IV)", fontsize=12)
        ax.set_title(f"Top {n} Features by IV", fontsize=14, fontweight="bold")

        # Threshold lines
        for thresh, label in [
            (0.02, "Not Predictive"),
            (0.10, "Weak"),
            (0.30, "Medium"),
            (0.50, "Strong"),
        ]:
            ax.axvline(thresh, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(thresh + 0.005, -0.5, label, fontsize=8, color="gray")

        fig.tight_layout()
        return fig

    # ── Private Helpers ───────────────────────────────────────────────────

    def _infer_dtype(self, col: pd.Series) -> str:
        """Infer whether a column should be treated as numerical or categorical."""
        if col.dtype == object or isinstance(col.dtype, pd.CategoricalDtype):
            return "categorical"
        if col.nunique() <= 2 and set(col.dropna().unique()).issubset({0, 1}):
            return "categorical"
        return "numerical"

    def _create_binner(self, name: str, dtype: str) -> OptimalBinning:
        """Create an OptimalBinning instance with appropriate settings."""
        if dtype == "categorical":
            return OptimalBinning(
                name=name,
                dtype="categorical",
                solver="cp",
                cat_cutoff=self.cat_cutoff,
                max_n_bins=self.max_n_bins,
                min_bin_size=self.min_bin_size,
            )
        return OptimalBinning(
            name=name,
            dtype="numerical",
            solver="cp",
            max_n_bins=self.max_n_bins,
            min_bin_size=self.min_bin_size,
            monotonic_trend=self.monotonic_trend,
        )

    def _prepare_column(
        self, col: pd.Series, y: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """Drop rows where both feature and target are not null."""
        mask = col.notna() & y.notna()
        return col[mask], y[mask]

    def _coerce_column(self, col: pd.Series, feat: str) -> pd.Series:
        """Coerce column dtype to match what the binner expects."""
        dtype = self.feature_dtypes_.get(feat, "numerical")
        if dtype == "categorical":
            return col.astype(str).where(col.notna(), other=np.nan)
        return pd.to_numeric(col, errors="coerce")

    def _build_bin_table(self, binner: OptimalBinning, feat: str) -> pd.DataFrame:
        """Extract bin-level statistics from a fitted OptimalBinning object."""
        table = binner.binning_table.build()

        # Standardize column names
        rename_map = {}
        for c in table.columns:
            cl = c.lower().replace(" ", "_")
            if "count" == cl:
                rename_map[c] = "count"
            elif cl in ("event", "event_count"):
                rename_map[c] = "event_count"
            elif cl in ("non_event", "non_event_count"):
                rename_map[c] = "non_event_count"
            elif cl in ("event_rate", "event rate"):
                rename_map[c] = "event_rate"
            elif cl == "woe":
                rename_map[c] = "woe"
            elif cl == "iv":
                rename_map[c] = "iv"
            elif cl == "bin":
                rename_map[c] = "bin"
        table = table.rename(columns=rename_map)

        # Rename index-based bin column if needed
        if "bin" not in table.columns:
            table = table.reset_index()
            if "Bin" in table.columns:
                table = table.rename(columns={"Bin": "bin"})

        # Drop the Totals row
        if "bin" in table.columns:
            table = table[table["bin"].astype(str) != "Totals"].copy()

        # Ensure required columns exist
        for needed in ["count", "event_count", "non_event_count", "event_rate", "woe", "iv"]:
            if needed not in table.columns:
                # Try case-insensitive fallback
                for c in table.columns:
                    if c.lower().replace(" ", "_") == needed:
                        table = table.rename(columns={c: needed})
                        break

        # Ensure event_rate is numeric
        if "event_rate" in table.columns:
            table["event_rate"] = pd.to_numeric(table["event_rate"], errors="coerce")
        if "woe" in table.columns:
            table["woe"] = pd.to_numeric(table["woe"], errors="coerce")
        if "iv" in table.columns:
            table["iv"] = pd.to_numeric(table["iv"], errors="coerce")

        table["feature"] = feat
        return table

    def _check_fitted(self) -> None:
        """Raise if fit() has not been called."""
        if not self._fitted:
            raise RuntimeError("WOEBinner has not been fitted. Call fit() first.")
