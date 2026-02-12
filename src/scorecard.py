"""
Credit Risk Scorecard — points-based scoring from logistic regression.

Converts a fitted logistic regression model with WoE-transformed features
into a traditional credit scorecard with interpretable point allocations
per feature bin.

Score formula:
    Score = Offset + Factor × Σ(βi × WoEi)
    Factor = PDO / ln(2)
    Offset = Base_Score - Factor × ln(Base_Odds)

Usage:
    from src.scorecard import Scorecard
    sc = Scorecard(model, binner, base_score=600, pdo=20, base_odds=1.0)
    scores = sc.score(X_woe)
    table = sc.generate_scorecard_table()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


class Scorecard:
    """Points-based credit scorecard from logistic regression + WoE bins.

    Parameters:
        model: Fitted sklearn LogisticRegression or LogisticRegressionCV.
        binner: Fitted WOEBinner with bin_tables_ and iv_ attributes.
        base_score: Anchor score (e.g. 600).
        pdo: Points to Double Odds.
        base_odds: Good/Bad odds at the base score (default 1.0 = 50% PD).
        feature_names: Ordered list of feature names used in the model.
            If None, inferred from model.feature_names_in_ when available.
    """

    def __init__(
        self,
        model: LogisticRegression | LogisticRegressionCV,
        binner: object,
        base_score: float = 600,
        pdo: float = 20,
        base_odds: float = 1.0,
        feature_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.binner = binner
        self.base_score = base_score
        self.pdo = pdo
        self.base_odds = base_odds

        # Derive factor and offset
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

        # Feature names
        if feature_names is not None:
            self.feature_names = list(feature_names)
        elif hasattr(model, "feature_names_in_"):
            self.feature_names = list(model.feature_names_in_)
        else:
            raise ValueError(
                "feature_names must be provided or model must have "
                "feature_names_in_ attribute."
            )

        # Extract coefficients and intercept
        self.coefficients = model.coef_.ravel()
        self.intercept = model.intercept_[0]

        if len(self.coefficients) != len(self.feature_names):
            raise ValueError(
                f"Number of coefficients ({len(self.coefficients)}) does not "
                f"match number of features ({len(self.feature_names)})."
            )

    # ── Scoring ────────────────────────────────────────────────────────────

    def score(self, X_woe: pd.DataFrame) -> np.ndarray:
        """Compute credit scores from WoE-transformed features.

        Args:
            X_woe: DataFrame with WoE-transformed feature values.

        Returns:
            Array of credit scores (higher = lower risk).
        """
        X = X_woe[self.feature_names].values
        log_odds = X @ self.coefficients + self.intercept
        scores = self.offset - self.factor * log_odds
        return scores

    def score_to_pd(self, scores: np.ndarray | float) -> np.ndarray | float:
        """Convert credit score(s) back to probability of default.

        Args:
            scores: Credit score(s).

        Returns:
            Probability of default.
        """
        log_odds = (self.offset - np.asarray(scores)) / self.factor
        return 1.0 / (1.0 + np.exp(-log_odds))

    # ── Scorecard table ────────────────────────────────────────────────────

    def generate_scorecard_table(self) -> pd.DataFrame:
        """Generate a publication-ready scorecard table.

        Returns:
            DataFrame with columns: feature, bin, woe, coefficient, points.
        """
        rows = []
        # Bins to skip (optbinning artefacts with zero WoE contribution)
        _skip_bins = {"special", "missing", "totals", "nan", "none", ""}

        for feat, coef in zip(self.feature_names, self.coefficients):
            if hasattr(self.binner, "bin_tables_") and feat in self.binner.bin_tables_:
                table = self.binner.bin_tables_[feat]
                for _, row in table.iterrows():
                    bin_val = row["bin"]
                    bin_label = str(bin_val).strip()
                    # Skip optbinning artefact rows (Special, Missing, NaN, etc.)
                    try:
                        is_na = bool(pd.isna(bin_val))
                    except (ValueError, TypeError):
                        is_na = False
                    if bin_label.lower() in _skip_bins or is_na:
                        continue
                    woe_val = float(row["woe"]) if pd.notna(row.get("woe")) else 0.0
                    points = -self.factor * coef * woe_val
                    rows.append({
                        "feature": feat,
                        "bin": bin_label,
                        "woe": round(woe_val, 6),
                        "coefficient": round(coef, 6),
                        "points": round(points, 2),
                    })
            else:
                # Binary flag or feature without bin table
                rows.append({
                    "feature": feat,
                    "bin": "1 (flag present)",
                    "woe": 1.0,
                    "coefficient": round(coef, 6),
                    "points": round(-self.factor * coef * 1.0, 2),
                })
                rows.append({
                    "feature": feat,
                    "bin": "0 (flag absent)",
                    "woe": 0.0,
                    "coefficient": round(coef, 6),
                    "points": 0.0,
                })

        scorecard_df = pd.DataFrame(rows)

        # Add base score row
        base_row = pd.DataFrame([{
            "feature": "BASE_SCORE",
            "bin": "",
            "woe": np.nan,
            "coefficient": np.nan,
            "points": round(self.offset - self.factor * self.intercept, 2),
        }])
        scorecard_df = pd.concat([base_row, scorecard_df], ignore_index=True)

        return scorecard_df

    # ── Feature contributions ──────────────────────────────────────────────

    def feature_contributions(self, X_woe: pd.DataFrame) -> pd.DataFrame:
        """Compute per-feature point contributions for each observation.

        Args:
            X_woe: DataFrame with WoE-transformed feature values.

        Returns:
            DataFrame with same index as X_woe, columns = feature names,
            values = point contributions. A 'base_score' column and
            'total_score' column are appended.
        """
        X = X_woe[self.feature_names]
        contributions = pd.DataFrame(index=X.index)

        for feat, coef in zip(self.feature_names, self.coefficients):
            contributions[feat] = -self.factor * coef * X[feat].values

        contributions["base_score"] = self.offset - self.factor * self.intercept
        contributions["total_score"] = contributions.sum(axis=1)

        return contributions

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a text summary of scorecard parameters."""
        lines = [
            "Scorecard Summary",
            "=" * 50,
            f"  Base Score:        {self.base_score}",
            f"  PDO:               {self.pdo}",
            f"  Base Odds:         {self.base_odds}",
            f"  Factor:            {self.factor:.4f}",
            f"  Offset:            {self.offset:.4f}",
            f"  Intercept:         {self.intercept:.6f}",
            f"  Num Features:      {len(self.feature_names)}",
            "",
            "  Features & Coefficients:",
        ]
        for feat, coef in zip(self.feature_names, self.coefficients):
            lines.append(f"    {feat:<40s}  {coef:+.6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Scorecard(base={self.base_score}, pdo={self.pdo}, "
            f"n_features={len(self.feature_names)})"
        )
