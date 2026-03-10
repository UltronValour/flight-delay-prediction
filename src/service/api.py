"""
FastAPI backend for the Flight Delay Prediction project.

Exposes a simple REST API that:
- Loads the trained model artefact (pipeline + decision threshold)
- Accepts flight details as input
- Returns a delay prediction with probability
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI

from .schemas import FlightInput, PredictionResponse


ARTEFACT_PATH = Path("models/flight_delay_model.pkl")


def load_artefact(path: Path = ARTEFACT_PATH) -> Dict[str, Any]:
    """Load the saved model artefact containing pipeline and threshold."""
    if not path.exists():
        raise FileNotFoundError(f"Model artefact not found at: {path.resolve()}")

    artefact = joblib.load(path)

    if not isinstance(artefact, dict):
        raise ValueError("Loaded artefact is not a dictionary as expected.")
    if "model" not in artefact or "threshold" not in artefact:
        raise KeyError(
            "Artefact dictionary must contain 'model' and 'threshold' keys."
        )

    return artefact


# Load artefact once at startup
artefact = load_artefact()
model = artefact["model"]
threshold: float = float(artefact["threshold"])


app = FastAPI(title="Flight Delay Prediction API")


@app.get("/")
async def root() -> Dict[str, str]:
    """Health-check/root endpoint."""
    return {"message": "Flight Delay Prediction API is running"}


@app.post("/predict_delay", response_model=PredictionResponse)
async def predict_delay(flight: FlightInput) -> Dict[str, Any]:
    """
    Predict whether a flight will be delayed.

    Uses the trained sklearn Pipeline and stored decision threshold.
    """
    # Convert Pydantic model to DataFrame with a single row
    input_df = pd.DataFrame([flight.model_dump()])

    # Probability of delayed class (1)
    proba_delayed = float(model.predict_proba(input_df)[:, 1][0])

    # Apply decision threshold
    is_delayed = proba_delayed >= threshold
    prediction_label = "Delayed" if is_delayed else "Not Delayed"

    return {
        "prediction": prediction_label,
        "probability": proba_delayed,
    }
