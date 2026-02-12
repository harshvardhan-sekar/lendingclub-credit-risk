# LendingClub Credit Risk Analytics

Portfolio management and loss forecasting tool for LendingClub consumer loans. Implements a full PD → EAD → LGD → ECL pipeline with institutional-format receivables tracking, flow rate analysis, DCF-ECL methodology, and an interactive Streamlit dashboard.

## Prerequisites

- Python 3.10+
- Git
- (Optional) Kaggle API for automated data download
- (Optional) FRED API key for macroeconomic data

## Data Setup

> **Note:** The raw data files are not included in this repository due to size constraints. Follow the instructions below to download them.

### Option 1: Kaggle CLI (Recommended)

```bash
# Install and configure Kaggle API
pip install kaggle
# Place your kaggle.json in ~/.kaggle/

# Run the download script
python download_data.py
```

### Option 2: Manual Download

1. Go to [LendingClub Dataset on Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. Download and extract the dataset
3. Place the following files in `data/raw/`:

| File | Size | Rows |
|------|------|------|
| `accepted_2007_to_2018Q4.csv` | ~1.6 GB | 2,260,701 |
| `rejected_2007_to_2018Q4.csv` | ~1.7 GB | 27,648,741 |
| `benchmark_population_2014.csv` | ~7.7 MB | 200,000 |
| `LCDataDictionary.xlsx` | — | — |

### FRED API Setup

Register for a free API key at [https://fred.stlouisfed.org/](https://fred.stlouisfed.org/) and set it as an environment variable:

```bash
export FRED_API_KEY="your_api_key_here"
```

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

```
lendingclub-credit-risk/
├── CLAUDE.md                    # AI assistant context file
├── README.md
├── requirements.txt
├── config.py                    # Configuration constants and paths
├── download_data.py             # Data download helper script
├── .gitignore
│
├── data/
│   ├── raw/                     # Original dataset (gitignored)
│   ├── processed/               # Cleaned datasets, feature-engineered data
│   ├── models/                  # Serialized models and scorecard objects
│   └── results/                 # Metrics, ECL summaries, validation reports
│
├── notebooks/
│   ├── 01_EDA_and_Data_Cleaning.ipynb
│   ├── 02_WOE_IV_Feature_Engineering.ipynb
│   ├── 03_PD_Model_Scorecard.ipynb
│   ├── 04_PD_Model_ML_Ensemble.ipynb
│   ├── 05_EAD_Model.ipynb
│   ├── 06_LGD_Model.ipynb
│   ├── 07_ECL_Computation_and_Vintage_Analysis.ipynb
│   ├── 08_Model_Validation_and_Monitoring.ipynb
│   └── 09_Macro_Scenario_and_Strategy_Analysis.ipynb
│
├── src/                         # Reusable modules
│   ├── data_processing.py
│   ├── woe_binning.py
│   ├── scorecard.py
│   ├── models.py
│   ├── ecl_engine.py
│   ├── flow_rates.py
│   ├── ecl_projector.py
│   ├── macro_scenarios.py
│   ├── validation.py
│   └── visualization.py
│
├── app/                         # Streamlit dashboard
│   ├── streamlit_app.py
│   ├── pages/
│   ├── components/
│   └── utils/
│
├── docs/                        # Project documentation
├── reports/                     # Final presentation
└── tests/                       # Unit tests
```

## Usage

Notebooks should be executed in order (01 → 09). Each notebook reads outputs from the previous one.

To launch the Streamlit dashboard:

```bash
streamlit run app/streamlit_app.py
```
