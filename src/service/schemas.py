"""
Pydantic schemas for the Flight Delay Prediction API.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FlightInput(BaseModel):
    """Request body for flight delay prediction."""

    airline: str = Field(..., description="Airline code, e.g., AA, UA, DL")
    origin_airport: str = Field(..., description="Origin airport IATA code, e.g., JFK")
    destination_airport: str = Field(
        ..., description="Destination airport IATA code, e.g., LAX"
    )
    departure_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Scheduled local departure hour (0–23).",
    )
    departure_month: int = Field(
        ...,
        ge=1,
        le=12,
        description="Scheduled departure month (1–12).",
    )
    departure_day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day of week (0 = Monday, …, 6 = Sunday).",
    )
    distance: float = Field(
        ...,
        ge=0,
        description="Great-circle distance between origin and destination in miles.",
    )


class PredictionResponse(BaseModel):
    """Response schema for prediction results."""

    prediction: Literal["Delayed", "Not Delayed"]
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that the flight will be delayed.",
    )
