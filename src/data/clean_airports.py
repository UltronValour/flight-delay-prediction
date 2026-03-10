"""
Basic cleaning script for an airports lookup table.

If present, this script:
- Loads data/raw/airports.csv
- Normalizes IATA codes
- Keeps a useful subset of columns
- Drops duplicates and rows without valid IATA codes
- Writes cleaned data to data/clean/airports_clean.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_AIRPORTS_PATH = Path("data/raw/airports.csv")
CLEAN_AIRPORTS_PATH = Path("data/clean/airports_clean.csv")


def load_raw_airports(path: Path = RAW_AIRPORTS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw airports file not found at: {path.resolve()}")
    return pd.read_csv(path)


def clean_airports(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Common columns in many public airports datasets
    possible_cols = {
        "iata_code": ["iata_code", "iata", "iata_code_unique"],
        "name": ["name", "airport_name"],
        "city": ["city"],
        "state": ["state", "state_name", "region_name"],
    }

    col_map = {}
    for target, candidates in possible_cols.items():
        for c in candidates:
            if c in df.columns:
                col_map[target] = c
                break

    # Build a trimmed DataFrame with available columns
    trimmed = pd.DataFrame()
    for target, source in col_map.items():
        trimmed[target] = df[source]

    if "iata_code" in trimmed.columns:
        trimmed["iata_code"] = trimmed["iata_code"].astype(str).str.upper().str.strip()
        trimmed = trimmed[trimmed["iata_code"].str.len() == 3]
        trimmed = trimmed.drop_duplicates(subset=["iata_code"])

    return trimmed


def save_clean_airports(df: pd.DataFrame, path: Path = CLEAN_AIRPORTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    print(f"Loading raw airports from: {RAW_AIRPORTS_PATH}")
    df_raw = load_raw_airports()
    print(f"Raw shape: {df_raw.shape}")

    print("Cleaning airports data...")
    df_clean = clean_airports(df_raw)
    print(f"Cleaned shape: {df_clean.shape}")

    print(f"Saving cleaned airports to: {CLEAN_AIRPORTS_PATH}")
    save_clean_airports(df_clean)
    print("Done.")


if __name__ == "__main__":
    main()
