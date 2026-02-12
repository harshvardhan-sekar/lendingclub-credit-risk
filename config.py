"""
Configuration constants for the LendingClub Credit Risk Analytics project.
"""
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
DATA_MODELS_PATH = PROJECT_ROOT / "data" / "models"
DATA_RESULTS_PATH = PROJECT_ROOT / "data" / "results"

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Target Variable ───────────────────────────────────────────────────────────
TARGET_COL = "default"
DEFAULT_STATUSES = ["Charged Off", "Default"]
NON_DEFAULT_STATUSES = ["Fully Paid"]
DROP_STATUSES = [
    "Current",
    "In Grace Period",
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
]

# ── Time-Based Split ──────────────────────────────────────────────────────────
TRAIN_END_YEAR = 2015
VAL_YEAR = 2016
TEST_START_YEAR = 2017

# ── Scorecard Parameters ──────────────────────────────────────────────────────
SCORECARD_BASE = 600
SCORECARD_PDO = 20  # Points to Double Odds

# ── PSI Thresholds (Population Stability Index) ──────────────────────────────
PSI_GREEN = 0.10   # PSI < 0.10 → Green (stable)
PSI_AMBER = 0.25   # 0.10 ≤ PSI < 0.25 → Amber (monitor)
PSI_RED = 0.25     # PSI ≥ 0.25 → Red (action required)

# ── Grade Ordering ────────────────────────────────────────────────────────────
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]

# ── FRED Macroeconomic Series ─────────────────────────────────────────────────
FRED_SERIES = [
    "UNRATE",            # Unemployment Rate
    "CSUSHPINSA",        # Case-Shiller Home Price Index
    "A191RL1Q225SBEA",   # Real GDP Growth Rate (quarterly)
    "CPIAUCSL",          # Consumer Price Index
    "DFF",               # Federal Funds Rate
    "UMCSENT",           # University of Michigan Consumer Sentiment
]
