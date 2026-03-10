"""
Prediction script for the Flight Delay Prediction project.

Loads the trained model artefact (pipeline + decision threshold) and
provides a simple function to obtain delay predictions for a single
flight-like input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


# Make path relative to this script so it works from anywhere
# src/models/predict.py -> ../../models/flight_delay_model.pkl
ARTEFACT_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "flight_delay_model.pkl"


def load_artefact(path: Path = ARTEFACT_PATH) -> Dict[str, Any]:
    """Load the saved model artefact containing pipeline and threshold."""
    if not path.exists():
        raise FileNotFoundError(f"Model artefact not found at: {path.resolve()}")

    artefact = joblib.load(path)

    # Basic sanity checks
    if not isinstance(artefact, dict):
        raise ValueError("Loaded artefact is not a dictionary as expected.")
    if "model" not in artefact or "threshold" not in artefact:
        raise KeyError(
            "Artefact dictionary must contain 'model' and 'threshold' keys."
        )

    return artefact


def predict_delay(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict flight delay given a single input record.

    Parameters
    ----------
    input_data : dict
        Dictionary with the following expected keys:
        - airline
        - origin_airport
        - destination_airport
        - departure_hour
        - departure_month
        - departure_day_of_week
        - distance

    Returns
    -------
    dict
        {
            "prediction": "Delayed" or "Not Delayed",
            "probability": float (probability of being delayed),
        }
    """
    artefact = load_artefact()
    model = artefact["model"]
    threshold = float(artefact["threshold"])

    # Ensure input is a DataFrame with a single row
    input_df = pd.DataFrame([input_data])

    # Get predicted probability for the positive class (delayed = 1)
    proba_delayed = model.predict_proba(input_df)[:, 1][0]

    # Apply stored decision threshold
    is_delayed = proba_delayed >= threshold

    prediction_label = "Delayed" if is_delayed else "Not Delayed"

    return {
        "prediction": prediction_label,
        "probability": float(proba_delayed),
    }


if __name__ == "__main__":
    # Example usage with a sample flight
    sample_input = {
        "airline": "AA",
        "origin_airport": "JFK",
        "destination_airport": "LAX",
        "departure_hour": 15,
        "departure_month": 7,
        "departure_day_of_week": 4,  # 0=Monday, 6=Sunday
        "distance": 2475,
    }

    result = predict_delay(sample_input)
    print("Sample input:", sample_input)
    print("Prediction result:", result)
