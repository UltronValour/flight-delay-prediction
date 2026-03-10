"""
Data merging script for the Flight Delay Prediction project.

This script:
- Loads cleaned flights and airports tables
- Merges airport metadata for both origin and destination airports
- Produces an enriched flights dataset that can be used for EDA or
  feature engineering.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CLEAN_FLIGHTS_PATH = Path("data/clean/flights_clean.csv")
CLEAN_AIRPORTS_PATH = Path("data/clean/airports_clean.csv")
MERGED_OUTPUT_PATH = Path("data/clean/flights_merged.csv")


def load_clean_flights(path: Path = CLEAN_FLIGHTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Clean flights file not found at: {path.resolve()}")
    return pd.read_csv(path, low_memory=False)


def load_clean_airports(path: Path = CLEAN_AIRPORTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Clean airports file not found at: {path.resolve()}")
    return pd.read_csv(path)


def merge_flights_and_airports(
    flights: pd.DataFrame, airports: pd.DataFrame
) -> pd.DataFrame:
    """
    Enrich flights with origin/destination airport metadata.

    Expects:
    - flights: columns including origin_airport, destination_airport
    - airports: at least an iata_code column, plus optional name/city/state
    """
    flights = flights.copy()
    airports = airports.copy()

    flights.columns = [c.lower() for c in flights.columns]
    airports.columns = [c.lower() for c in airports.columns]

    if "iata_code" not in airports.columns:
        raise KeyError("Clean airports table must contain an 'iata_code' column.")

    # Prepare origin airports metadata
    origin_cols = [c for c in airports.columns if c != "iata_code"]
    airports_origin = airports.rename(
        columns={
            "iata_code": "origin_airport",
            **{c: f"origin_{c}" for c in origin_cols},
        }
    )

    # Prepare destination airports metadata
    dest_cols = [c for c in airports.columns if c != "iata_code"]
    airports_dest = airports.rename(
        columns={
            "iata_code": "destination_airport",
            **{c: f"destination_{c}" for c in dest_cols},
        }
    )

    # Merge origin metadata
    if "origin_airport" in flights.columns:
        flights = flights.merge(
            airports_origin,
            on="origin_airport",
            how="left",
        )

    # Merge destination metadata
    if "destination_airport" in flights.columns:
        flights = flights.merge(
            airports_dest,
            on="destination_airport",
            how="left",
        )

    return flights


def save_merged_flights(df: pd.DataFrame, path: Path = MERGED_OUTPUT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    print(f"Loading clean flights from: {CLEAN_FLIGHTS_PATH}")
    flights = load_clean_flights()
    print(f"Flights shape: {flights.shape}")

    print(f"Loading clean airports from: {CLEAN_AIRPORTS_PATH}")
    airports = load_clean_airports()
    print(f"Airports shape: {airports.shape}")

    print("Merging flights with airport metadata...")
    merged = merge_flights_and_airports(flights, airports)
    print(f"Merged shape: {merged.shape}")

    print(f"Saving merged flights to: {MERGED_OUTPUT_PATH}")
    save_merged_flights(merged)
    print("Done.")


if __name__ == "__main__":
    main()
