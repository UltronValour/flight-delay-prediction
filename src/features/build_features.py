"""
Feature engineering script for the Flight Delay Prediction project.

This script:
- Loads the raw flights dataset from data/raw/flights.csv
- Engineers time-based, route, weekend, and distance features
- Creates a binary target variable `delayed` (>=15 minutes)
- Selects a subset of useful modeling features
- Drops rows with missing values in these features
- Saves the processed dataset to data/processed/flight_features.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/flights.csv")
PROCESSED_DATA_PATH = Path("data/processed/flight_features.csv")


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw flights dataset from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")
    # Use low_memory=False to avoid mixed-type warnings on large files
    return pd.read_csv(path, low_memory=False)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time, route, weekend, distance, and target features.

    The raw dataset may contain uppercase column names. For robustness,
    all column names are converted to lowercase before feature logic.
    """
    # Work on a copy and normalize column names to lowercase
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    required_columns = [
        "scheduled_departure",
        "origin_airport",
        "destination_airport",
        "airline",
        "departure_delay",
    ]

    missing_required = [c for c in required_columns if c not in df.columns]
    if missing_required:
        raise KeyError(
            "Missing required columns in raw data: "
            + ", ".join(missing_required)
        )

    # Ensure scheduled_departure is datetime to derive time-based features
    df["scheduled_departure"] = pd.to_datetime(
        df["scheduled_departure"], errors="coerce"
    )

    # Time-based features
    df["departure_hour"] = df["scheduled_departure"].dt.hour
    df["departure_month"] = df["scheduled_departure"].dt.month
    df["departure_day_of_week"] = df["scheduled_departure"].dt.dayofweek

    # Route feature (origin_destination)
    df["route"] = df["origin_airport"].astype(str) + "_" + df[
        "destination_airport"
    ].astype(str)

    # Weekend indicator (Saturday=5, Sunday=6)
    df["is_weekend"] = (df["departure_day_of_week"] >= 5).astype(int)

    # Distance feature: use existing distance column if present, else raise
    if "distance" not in df.columns:
        raise KeyError(
            "Expected 'distance' column not found in raw data. "
            "Please ensure the flights dataset includes a 'distance' field."
        )

    # Target variable: delayed if departure_delay >= 15 minutes
    df["delayed"] = (df["departure_delay"] >= 15).astype(int)

    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the subset of features used for modeling."""
    feature_columns = [
        "airline",
        "origin_airport",
        "destination_airport",
        "departure_hour",
        "departure_month",
        "departure_day_of_week",
        "distance",
        "delayed",
    ]

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise KeyError(
            "Missing expected feature columns after engineering: "
            + ", ".join(missing_features)
        )

    return df[feature_columns]


def save_processed_data(df: pd.DataFrame, path: Path) -> None:
    """Save the processed feature dataset to CSV, creating directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Run the full feature engineering pipeline."""
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df_raw = load_raw_data(RAW_DATA_PATH)
    print(f"Raw data shape: {df_raw.shape}")

    print("Engineering features...")
    df_features = engineer_features(df_raw)

    print("Selecting model features...")
    df_model = select_model_features(df_features)

    # Drop rows with missing values in selected features
    before_drop_shape = df_model.shape
    df_model = df_model.dropna()
    after_drop_shape = df_model.shape

    print(
        f"Dropped rows with missing values: "
        f"{before_drop_shape[0] - after_drop_shape[0]} rows removed."
    )
    print(f"Final feature dataset shape: {after_drop_shape}")

    print(f"Saving processed data to: {PROCESSED_DATA_PATH}")
    save_processed_data(df_model, PROCESSED_DATA_PATH)
    print("Feature generation completed successfully.")


if __name__ == "__main__":
    main()
