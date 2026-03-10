"""
Basic cleaning script for the raw flights dataset.

This script:
- Loads data/raw/flights.csv
- Normalizes column names to lowercase
- Parses scheduled_departure as datetime (if present)
- Drops rows with missing critical fields
- Optionally filters out clearly invalid distances
- Writes cleaned data to data/clean/flights_clean.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_FLIGHTS_PATH = Path("data/raw/flights.csv")
CLEAN_FLIGHTS_PATH = Path("data/clean/flights_clean.csv")


def load_raw_flights(path: Path = RAW_FLIGHTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw flights file not found at: {path.resolve()}")
    return pd.read_csv(path, low_memory=False)


def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names to lowercase for consistency
    df.columns = [c.lower() for c in df.columns]

    # Parse scheduled departure if available
    if "scheduled_departure" in df.columns:
        df["scheduled_departure"] = pd.to_datetime(
            df["scheduled_departure"], errors="coerce"
        )

    # Drop rows missing critical identifiers
    critical_cols = ["airline", "origin_airport", "destination_airport"]
    present_critical = [c for c in critical_cols if c in df.columns]
    if present_critical:
        df = df.dropna(subset=present_critical)

    # Remove obviously invalid distances if distance column exists
    if "distance" in df.columns:
        df = df[df["distance"].notna()]
        df = df[df["distance"] >= 0]

    return df


def save_clean_flights(df: pd.DataFrame, path: Path = CLEAN_FLIGHTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    print(f"Loading raw flights from: {RAW_FLIGHTS_PATH}")
    df_raw = load_raw_flights()
    print(f"Raw shape: {df_raw.shape}")

    print("Cleaning flights data...")
    df_clean = clean_flights(df_raw)
    print(f"Cleaned shape: {df_clean.shape}")

    print(f"Saving cleaned flights to: {CLEAN_FLIGHTS_PATH}")
    save_clean_flights(df_clean)
    print("Done.")


if __name__ == "__main__":
    main()
