"""
WOE/IV Binning Module for Credit Risk Scorecard Development.

Implements Weight of Evidence (WOE) and Information Value (IV) calculations
using decision tree-based optimal binning. This approach mirrors institutional
scorecard methodology for VantageScore/FICO-style feature binning.

WOE = ln(Distribution_Good / Distribution_Bad)
IV  = Σ (Distribution_Good - Distribution_Bad) × WOE

Binning uses sklearn DecisionTreeClassifier to find optimal split points,
then enforces monotonic bad-rate ordering across bins.
"""

from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class WOEBinner:
    """WOE/IV binner using decision-tree-based optimal binning.

    Fits optimal bins on training data, computes WOE/IV per bin,
    and transforms features to WOE values.

    Args:
        max_bins: Maximum number of bins per feature.
        min_bin_pct: Minimum percentage of observations per bin.
        monotonic: If True, enforce monotonic bad rate across bins.
        random_state: Random seed for reproducibility.

    Example:
        >>> binner = WOEBinner(max_bins=10)
        >>> binner.fit(X_train, y_train)
        >>> X_woe = binner.transform(X_train)
        >>> iv_summary = binner.iv_summary()
    """

    def __init__(
        self,
        max_bins: int = 10,
        min_bin_pct: float = 0.05,
        monotonic: bool = True,
        random_state: int = 42,
    ) -> None:
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.monotonic = monotonic
        self.random_state = random_state

        self.bin_edges_: dict[str, np.ndarray] = {}
        self.woe_maps_: dict[str, dict[int, float]] = {}
        self.iv_values_: dict[str, float] = {}
        self.bin_tables_: dict[str, pd.DataFrame] = {}
        self.fitted_features_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOEBinner":
        """Fit WOE binning on training data.

        Args:
            X: Feature matrix (numeric columns only).
            y: Binary target (0/1).

        Returns:
            self
        """
        self.fitted_features_ = []

        for col in X.columns:
            try:
                series = X[col].copy()
                mask = series.notna()
                if mask.sum() < 100:
                    continue

                edges = self._find_optimal_bins(series[mask].values, y[mask].values)
                if edges is None:
                    continue

                bin_table = self._compute_woe_iv(series, y, edges, col)
                if bin_table is None:
                    continue

                self.bin_edges_[col] = edges
                self.bin_tables_[col] = bin_table
                self.woe_maps_[col] = dict(
                    zip(bin_table["bin_id"], bin_table["woe"])
                )
                self.iv_values_[col] = bin_table["iv_component"].sum()
                self.fitted_features_.append(col)

            except Exception as e:
                warnings.warn(f"Skipping {col}: {e}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features to WOE values.

        Args:
            X: Feature matrix (same columns as fit).

        Returns:
            DataFrame with WOE-transformed values.
        """
        result = pd.DataFrame(index=X.index)

        for col in self.fitted_features_:
            if col not in X.columns:
                continue
            edges = self.bin_edges_[col]
            woe_map = self.woe_maps_[col]
            bin_ids = np.digitize(X[col].fillna(-999999), edges)
            bin_ids = np.clip(bin_ids, 0, max(woe_map.keys()))
            result[col] = pd.Series(bin_ids, index=X.index).map(woe_map)
            # Fill any unmapped bins with 0 (neutral WOE)
            result[col] = result[col].fillna(0.0)

        return result

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Feature matrix.
            y: Binary target.

        Returns:
            WOE-transformed DataFrame.
        """
        self.fit(X, y)
        return self.transform(X)

    def iv_summary(self) -> pd.DataFrame:
        """Generate IV summary table with selection status.

        Returns:
            DataFrame with feature, IV, and selection status columns.
        """
        rows = []
        for feat in self.fitted_features_:
            iv = self.iv_values_[feat]
            if iv < 0.02:
                status = "drop_not_predictive"
            elif iv < 0.1:
                status = "weak"
            elif iv < 0.3:
                status = "medium"
            elif iv < 0.5:
                status = "strong"
            else:
                status = "suspicious_check_leakage"
            rows.append({"feature": feat, "iv": round(iv, 6), "selection_status": status})

        return (
            pd.DataFrame(rows)
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )

    def get_bin_table(self, feature: str) -> pd.DataFrame:
        """Get detailed bin table for a specific feature.

        Args:
            feature: Feature name.

        Returns:
            DataFrame with bin details: edges, count, event_count,
            non_event_count, event_rate, woe, iv_component.
        """
        if feature not in self.bin_tables_:
            raise KeyError(f"Feature '{feature}' not fitted.")
        return self.bin_tables_[feature].copy()

    def check_monotonicity(self, feature: str) -> dict:
        """Check if bad rates are monotonic across bins for a feature.

        Args:
            feature: Feature name.

        Returns:
            Dict with 'monotonic' (bool), 'direction' (str), 'rates' (list).
        """
        table = self.get_bin_table(feature)
        rates = table["event_rate"].values

        increasing = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
        decreasing = all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))

        if increasing:
            direction = "increasing"
        elif decreasing:
            direction = "decreasing"
        else:
            direction = "non_monotonic"

        return {
            "monotonic": increasing or decreasing,
            "direction": direction,
            "rates": rates.tolist(),
        }

    def plot_woe(self, feature: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot WOE values by bin for a feature.

        Args:
            feature: Feature name.
            ax: Optional matplotlib axes.

        Returns:
            Matplotlib axes object.
        """
        table = self.get_bin_table(feature)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        colors = ["#d62728" if w < 0 else "#2ca02c" for w in table["woe"]]
        ax.bar(range(len(table)), table["woe"], color=colors, edgecolor="black", alpha=0.8)
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels(table["bin_label"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("WOE")
        ax.set_title(f"WOE by Bin — {feature} (IV={self.iv_values_[feature]:.4f})")
        ax.axhline(y=0, color="black", linewidth=0.5)

        # Add event rate as secondary axis
        ax2 = ax.twinx()
        ax2.plot(range(len(table)), table["event_rate"] * 100, "ko-", markersize=4)
        ax2.set_ylabel("Bad Rate (%)")

        return ax

    def plot_bad_rate(self, feature: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot bad rate by bin for monotonicity check.

        Args:
            feature: Feature name.
            ax: Optional matplotlib axes.

        Returns:
            Matplotlib axes object.
        """
        table = self.get_bin_table(feature)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        mono_info = self.check_monotonicity(feature)
        color = "#2ca02c" if mono_info["monotonic"] else "#d62728"

        ax.bar(range(len(table)), table["event_rate"] * 100, color=color, alpha=0.7,
               edgecolor="black")
        ax.set_xticks(range(len(table)))
        ax.set_xticklabels(table["bin_label"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Bad Rate (%)")
        status = "MONOTONIC" if mono_info["monotonic"] else "NON-MONOTONIC"
        ax.set_title(f"Bad Rate — {feature} [{status}, {mono_info['direction']}]")

        return ax

    # ── Private helpers ──────────────────────────────────────────────────

    def _find_optimal_bins(
        self, x: np.ndarray, y: np.ndarray
    ) -> Optional[np.ndarray]:
        """Use decision tree to find optimal bin edges."""
        min_samples = max(int(len(x) * self.min_bin_pct), 50)

        tree = DecisionTreeClassifier(
            max_leaf_nodes=self.max_bins,
            min_samples_leaf=min_samples,
            random_state=self.random_state,
        )
        tree.fit(x.reshape(-1, 1), y)

        thresholds = sorted(tree.tree_.threshold[tree.tree_.threshold != -2])

        if len(thresholds) == 0:
            # Fallback: equal-frequency bins
            quantiles = np.linspace(0, 1, min(self.max_bins + 1, 6))[1:-1]
            thresholds = np.unique(np.quantile(x, quantiles))

        if len(thresholds) == 0:
            return None

        edges = np.array(thresholds)

        if self.monotonic:
            edges = self._enforce_monotonic_bins(x, y, edges)

        return edges

    def _enforce_monotonic_bins(
        self, x: np.ndarray, y: np.ndarray, edges: np.ndarray
    ) -> np.ndarray:
        """Merge adjacent bins to enforce monotonic bad rates."""
        max_iter = 20
        for _ in range(max_iter):
            bin_ids = np.digitize(x, edges)
            rates = []
            for b in range(len(edges) + 1):
                mask = bin_ids == b
                if mask.sum() > 0:
                    rates.append(y[mask].mean())
                else:
                    rates.append(np.nan)

            rates = np.array(rates)
            valid = ~np.isnan(rates)
            valid_rates = rates[valid]

            if len(valid_rates) <= 2:
                break

            # Check if already monotonic
            diffs = np.diff(valid_rates)
            if np.all(diffs >= -1e-10) or np.all(diffs <= 1e-10):
                break

            # Find the first violation and merge that pair
            increasing_violations = np.where(diffs < -1e-10)[0]
            decreasing_violations = np.where(diffs > 1e-10)[0]

            # Determine dominant direction
            if len(increasing_violations) <= len(decreasing_violations):
                # Trying to be increasing — merge at first decreasing violation
                if len(increasing_violations) > 0:
                    merge_idx = increasing_violations[0]
                else:
                    break
            else:
                if len(decreasing_violations) > 0:
                    merge_idx = decreasing_violations[0]
                else:
                    break

            # Map valid indices back to edge indices
            valid_idx_map = np.where(valid)[0]
            if merge_idx + 1 < len(valid_idx_map):
                edge_to_remove = valid_idx_map[merge_idx + 1] - 1
                if 0 <= edge_to_remove < len(edges):
                    edges = np.delete(edges, edge_to_remove)
                else:
                    break
            else:
                break

            if len(edges) == 0:
                break

        return edges

    def _compute_woe_iv(
        self,
        series: pd.Series,
        y: pd.Series,
        edges: np.ndarray,
        col_name: str,
    ) -> Optional[pd.DataFrame]:
        """Compute WOE and IV for each bin."""
        bin_ids = np.digitize(series.fillna(-999999).values, edges)

        total_events = y.sum()
        total_non_events = len(y) - total_events

        if total_events == 0 or total_non_events == 0:
            return None

        rows = []
        for b in sorted(np.unique(bin_ids)):
            mask = bin_ids == b
            count = mask.sum()
            events = y[mask].sum()
            non_events = count - events

            # Avoid division by zero with Laplace smoothing
            dist_event = max(events, 0.5) / total_events
            dist_non_event = max(non_events, 0.5) / total_non_events

            woe = np.log(dist_non_event / dist_event)
            iv_comp = (dist_non_event - dist_event) * woe

            # Create bin label
            if b == 0:
                label = f"(-inf, {edges[0]:.2f}]"
            elif b == len(edges):
                label = f"({edges[-1]:.2f}, inf)"
            else:
                label = f"({edges[b - 1]:.2f}, {edges[b]:.2f}]"

            rows.append(
                {
                    "bin_id": b,
                    "bin_label": label,
                    "count": int(count),
                    "event_count": int(events),
                    "non_event_count": int(non_events),
                    "event_rate": events / count if count > 0 else 0,
                    "woe": round(woe, 6),
                    "iv_component": round(iv_comp, 6),
                }
            )

        return pd.DataFrame(rows)
