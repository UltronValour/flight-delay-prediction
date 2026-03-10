"""
Streamlit dashboard for the Flight Delay Prediction project.

This UI collects flight details from the user, sends them to the FastAPI
backend, and displays the predicted delay status and probability.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Add the project root to sys.path so Streamlit Cloud can find 'src'
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import pandas as pd
import streamlit as st

from src.models.predict import predict_delay

def configure_page() -> None:
    st.set_page_config(
        page_title="Flight Delay Prediction • Dashboard",
        page_icon="✈️",
        layout="wide",
    )

    # Subtle, modern theming tweaks
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #020617 !important;
            color: #f9fafb !important;
        }
        .main {
            padding: 1.5rem 3rem 3rem 3rem;
            background: radial-gradient(circle at top, #1e293b 0, #020617 55%);
            color: #f9fafb;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #020617 0%, #020617 45%, #0b1120 100%);
            border-right: 1px solid #1e293b;
        }
        .stButton>button {
            border-radius: 999px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            border: 1px solid #38bdf8;
            background: linear-gradient(90deg, #0ea5e9, #22c55e);
            color: white;
        }
        .stButton>button:hover {
            filter: brightness(1.05);
        }
        h1, h2, h3, h4, label {
            color: #f9fafb !important;
        }
        .metric-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9ca3af;
        }
        .metric-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #f9fafb;
        }

        /* Make input widgets darker with high-contrast text */
        .stTextInput>div>div>input,
        .stNumberInput>div>div>input {
            background-color: #020617 !important;
            color: #f9fafb !important;
            border-radius: 0.5rem !important;
            border: 1px solid #1e293b !important;
        }
        .stSelectbox>div>div>div,
        .stSelectbox>div>div>div>span {
            background-color: #020617 !important;
            color: #f9fafb !important;
        }
        [data-baseweb="input"] input {
            color: #f9fafb !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_payload(
    airline: str,
    origin_airport: str,
    destination_airport: str,
    departure_hour: int,
    departure_month: int,
    departure_day_of_week: int,
    distance: float,
) -> Dict[str, Any]:
    return {
        "airline": airline.strip().upper(),
        "origin_airport": origin_airport.strip().upper(),
        "destination_airport": destination_airport.strip().upper(),
        "departure_hour": int(departure_hour),
        "departure_month": int(departure_month),
        "departure_day_of_week": int(departure_day_of_week),
        "distance": float(distance),
    }



def get_risk_band(probability: float) -> str:
    """Convert raw probability into a qualitative risk band."""
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Medium"
    return "Low"


def main() -> None:
    configure_page()

    header_left, header_right = st.columns([1.6, 1.0])
    with header_left:
        st.title("Flight Delay Prediction Dashboard")
        st.caption(
            "Production-style interface for scoring individual flights using a "
            "machine learning model trained on historical US flight data."
        )
    with header_right:
        st.markdown(
            """
            <div style="padding:0.75rem 1rem;border-radius:0.75rem;
                        background:rgba(15,23,42,0.75);border:1px solid #1e293b;">
              <div class="metric-label">Model objective</div>
              <div class="metric-value">Catch delays (high recall)</div>
              <div style="margin-top:0.5rem;font-size:0.8rem;color:#9ca3af;">
                Tuned decision threshold balances recall and precision
                for the delayed class.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.markdown(
            """
            <div style="
                padding:1.25rem 1rem;
                margin:0.5rem 0.25rem 1rem 0.25rem;
                border-radius:0.9rem;
                background:rgba(15,23,42,0.95);
                border:1px solid #1e293b;
                box-shadow:0 12px 30px rgba(15,23,42,0.6);
            ">
              <div style="font-size:0.8rem;text-transform:uppercase;
                          letter-spacing:0.12em;color:#38bdf8;
                          margin-bottom:0.25rem;">
                Project Overview
              </div>
              <h2 style="margin:0 0 0.5rem 0;">Flight Delay<br/>Prediction</h2>
              <p style="font-size:0.85rem;color:#9ca3af;margin-bottom:0.5rem;">
                Interactive dashboard powered by a machine learning model
                trained on historical US flight operations to estimate the
                risk of delay for an individual flight.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### How to use")
        st.markdown(
            "- Enter basic flight details on the right\n"
            "- Click **Predict Delay** to score the flight\n"
            "- Use the probability and risk band for decisions"
        )

        st.markdown("##### Model notes")
        st.markdown(
            "- Target: delay ≥ 15 minutes\n"
            "- Handles class imbalance with weights\n"
            "- Tuned threshold for higher delayed-flight recall"
        )

        st.markdown("---")
        st.caption("Flight Delay Prediction System")

    input_col, result_col = st.columns([1.4, 1.0])

    with input_col:
        st.subheader("Flight Details")

        airline = st.text_input("Airline (e.g., AA, UA, DL)", value="AA")
        origin_airport = st.text_input("Origin Airport (IATA code)", value="JFK")
        destination_airport = st.text_input(
            "Destination Airport (IATA code)", value="LAX"
        )

        col_time_1, col_time_2, col_time_3 = st.columns(3)
        with col_time_1:
            departure_hour = st.number_input(
                "Departure Hour",
                min_value=0,
                max_value=23,
                value=15,
                step=1,
                help="Local scheduled departure hour (0–23).",
            )
        with col_time_2:
            departure_month = st.number_input(
                "Departure Month",
                min_value=1,
                max_value=12,
                value=7,
                step=1,
                help="Month number (1–12).",
            )
        with col_time_3:
            departure_day_of_week = st.number_input(
                "Day of Week",
                min_value=0,
                max_value=6,
                value=3,
                step=1,
                help="0 = Monday, …, 6 = Sunday.",
            )

        distance = st.number_input(
            "Distance (miles)",
            min_value=0.0,
            value=2475.0,
            step=10.0,
            help="Great-circle distance between origin and destination.",
        )

        st.markdown("---")
        predict_clicked = st.button("Predict Delay", type="primary")

    with result_col:
        st.subheader("Prediction & Risk")
        placeholder = st.empty()

        if predict_clicked:
            if not airline or not origin_airport or not destination_airport:
                st.error("Please provide airline, origin, and destination codes.")
                return

            payload = build_payload(
                airline,
                origin_airport,
                destination_airport,
                departure_hour,
                departure_month,
                departure_day_of_week,
                distance,
            )

            with st.spinner("Scoring flight..."):
                try:
                    result = predict_delay(payload)
                except Exception as exc:
                    st.error(f"Error scoring flight: {exc}")
                    result = None

            if result is None:
                return

            prediction = result.get("prediction")
            probability = float(result.get("probability", 0.0))
            probability_pct = probability * 100
            risk_band = get_risk_band(probability)

            if risk_band == "High":
                header_text = "High Risk of Delay"
                container = placeholder.warning
            elif risk_band == "Medium":
                header_text = "Moderate Risk of Delay"
                container = placeholder.info
            else:
                header_text = "Low Risk of Delay"
                container = placeholder.success

            message = (
                f"### {header_text}\n"
                f"**Prediction:** {prediction}\n\n"
                f"**Estimated Probability:** {probability_pct:.2f}% "
                f"({risk_band} risk)"
            )
            container(message)

            st.markdown("#### Probability of Delay")
            st.progress(min(max(probability, 0.0), 1.0))

            st.markdown("#### Input Summary")
            st.dataframe(pd.DataFrame([payload]))

    st.markdown("---")
    st.caption("v1.0 • Flight Delay Prediction • Built with FastAPI & Streamlit • Made by Valour Moraes")


if __name__ == "__main__":
    main()

