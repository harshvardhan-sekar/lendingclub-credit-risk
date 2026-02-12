"""
Download LendingClub dataset from Kaggle.

Usage:
    python download_data.py

Requires the Kaggle API to be configured:
    pip install kaggle
    Place kaggle.json in ~/.kaggle/
"""
import subprocess
import sys
from pathlib import Path

DATA_RAW_PATH = Path(__file__).resolve().parent / "data" / "raw"

EXPECTED_FILES = {
    "accepted_2007_to_2018Q4.csv": {"size_approx": "~1.6 GB", "rows": "2,260,701"},
    "rejected_2007_to_2018Q4.csv": {"size_approx": "~1.7 GB", "rows": "27,648,741"},
    "benchmark_population_2014.csv": {"size_approx": "~7.7 MB", "rows": "200,000"},
}


def download_from_kaggle() -> None:
    """Attempt to download the LendingClub dataset using the Kaggle CLI."""
    DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "wordsforthewise/lending-club",
                "-p", str(DATA_RAW_PATH),
                "--unzip",
            ],
            check=True,
        )
        print("\nDownload complete. Checking files...")
        check_files()
    except FileNotFoundError:
        print_fallback_instructions()
    except subprocess.CalledProcessError as e:
        print(f"\nKaggle CLI failed with error: {e}")
        print_fallback_instructions()


def check_files() -> None:
    """Verify expected files exist and print their sizes."""
    print("\n" + "=" * 60)
    print("Expected files:")
    print("=" * 60)
    for filename, info in EXPECTED_FILES.items():
        filepath = DATA_RAW_PATH / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  [OK] {filename}: {size_mb:.1f} MB ({info['rows']} rows)")
        else:
            print(f"  [MISSING] {filename}: expected {info['size_approx']}, {info['rows']} rows")


def print_fallback_instructions() -> None:
    """Print manual download instructions when Kaggle API is unavailable."""
    print("\n" + "=" * 60)
    print("Kaggle API is not configured. Manual download instructions:")
    print("=" * 60)
    print()
    print("1. Go to: https://www.kaggle.com/datasets/wordsforthewise/lending-club")
    print("2. Click 'Download' (requires a free Kaggle account)")
    print("3. Extract the downloaded ZIP file")
    print(f"4. Place the CSV files in: {DATA_RAW_PATH}")
    print()
    print("Expected files after download:")
    for filename, info in EXPECTED_FILES.items():
        print(f"  - {filename}: {info['size_approx']}, {info['rows']} rows")
    print()
    print("Note: benchmark_population_2014.csv is a separate file.")
    print("If not included in the Kaggle download, ensure it is placed in data/raw/.")
    print()
    print("To configure the Kaggle API for future use:")
    print("  pip install kaggle")
    print("  # Create API token at https://www.kaggle.com/settings")
    print("  # Place kaggle.json in ~/.kaggle/")


if __name__ == "__main__":
    download_from_kaggle()
