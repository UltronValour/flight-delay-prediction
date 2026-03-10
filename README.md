# Flight Delay Prediction System ✈️

A production-style machine learning pipeline to **predict flight delays (≥ 15 minutes)** using historical flight data.  
The project includes:

- End‑to‑end **data pipeline** (raw → clean → merged → features)
- **Model training** with class‑imbalance handling and threshold tuning
- **FastAPI** backend for predictions
- **Streamlit dashboard** for an interactive, professional UI
- **Evaluation utilities** and a basic test suite

---

## 1. Project Overview

Modern flight operations generate huge amounts of data, but passengers and airlines still struggle with delays.  
This project builds an ML system that:

- Learns from historical flight data
- Predicts whether a given flight is **delayed by at least 15 minutes**
- Surfaces the prediction and risk level through a web UI

Use cases:

- Operational teams exploring delay risk for specific flights
- Educational / portfolio example of a full ML MLOps‑style workflow

---

## 2. Tech Stack

- **Language**: Python
- **Data & ML**: `pandas`, `numpy`, `scikit-learn`
- **Modeling**: RandomForest, Logistic Regression, class_weight + threshold tuning
- **Serving**: FastAPI, Uvicorn
- **UI**: Streamlit
- **Persistence**: `joblib`
- **Testing**: `pytest` + FastAPI `TestClient`

---

## 3. Repository Structure

```text
flight-delay-prediction/
├─ data/
│  ├─ raw/                # Original source data (flights.csv, airports.csv)
│  ├─ clean/              # Outputs from cleaning/merging scripts
│  └─ processed/          # Final modeling-ready features (flight_features.csv)
│
├─ models/
│  └─ flight_delay_model.pkl   # Saved artefact: {"model": Pipeline, "threshold": float}
│
├─ notebooks/
│  └─ 01_flight_eda.ipynb      # Exploratory data analysis
│
├─ src/
│  ├─ data/
│  │  ├─ clean_flights.py      # Clean raw flights data
│  │  ├─ clean_airports.py     # Clean airports lookup table
│  │  └─ merge_data.py         # Enrich flights with airport metadata
│  │
│  ├─ features/
│  │  └─ build_features.py     # Feature engineering + target creation
│  │
│  ├─ models/
│  │  ├─ train_model.py        # Train models, tune threshold, save artefact
│  │  ├─ evaluate_model.py     # Extra evaluation: CM, ROC-AUC, PR-AUC
│  │  └─ predict.py            # `predict_delay` helper for single-flight scoring
│  │
│  ├─ service/
│  │  ├─ api.py                # FastAPI app exposing /predict_delay
│  │  └─ schemas.py            # Pydantic schemas (FlightInput, PredictionResponse)
│  │
│  ├─ ui/
│  │  └─ app.py                # Streamlit dashboard (calls FastAPI)
│  │
│  └─ utils/
│     └─ helpers.py            # Shared helpers (e.g., dict_to_dataframe)
│
├─ tests/
│  └─ test_api.py              # Smoke test for FastAPI root endpoint
│
├─ requirements.txt
└─ README.md
```

---

## 4. Data Pipeline

The logical flow:

1. **Raw ingestion**
   - `data/raw/flights.csv` – historical flight records  
   - `data/raw/airports.csv` – airport metadata (if available)

2. **Cleaning (`src/data/`)**
   - `clean_flights.py`  
     - Lowercases columns  
     - Parses `scheduled_departure` (if present)  
     - Drops rows missing `AIRLINE`, `ORIGIN_AIRPORT`, `DESTINATION_AIRPORT`  
     - Removes invalid / negative `DISTANCE` rows  
     - Saves → `data/clean/flights_clean.csv`
   - `clean_airports.py`  
     - Normalizes airport columns, standardizes `iata_code`  
     - Keeps useful columns (e.g. name, city, state), drops duplicates  
     - Saves → `data/clean/airports_clean.csv`

3. **Merging (`src/data/merge_data.py`)**
   - Enriches flights with **origin and destination airport metadata**:
     - Joins `flights_clean` with `airports_clean` on IATA codes
     - Adds columns like `origin_name`, `origin_city`, `destination_name`, etc.
   - Saves → `data/clean/flights_merged.csv`

4. **Feature engineering (`src/features/build_features.py`)**
   - Reads the flights dataset (currently using the raw file; can be pointed to `flights_merged.csv` if you want enriched features).
   - Normalizes column names to lowercase.
   - Creates:
     - `departure_hour`, `departure_month`, `departure_day_of_week`
     - `route = origin_airport + "_" + destination_airport`
     - `is_weekend` flag
     - Uses `distance` as-is
   - Defines target:
     - `delayed = 1` if `departure_delay >= 15` minutes, else `0`
   - Selects modeling columns:
     - `airline`, `origin_airport`, `destination_airport`,  
       `departure_hour`, `departure_month`, `departure_day_of_week`,  
       `distance`, `delayed`
   - Drops rows with missing values in those fields.
   - Saves → `data/processed/flight_features.csv`

---

## 5. Modeling

### Training (`src/models/train_model.py`)

- Loads `data/processed/flight_features.csv` and samples 500,000 rows for speed.
- Splits into:
  - `X` = features (all columns except `delayed`)
  - `y` = target (`delayed`)
- Categorical encoding:
  - `ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore")` for:
    - `airline`, `origin_airport`, `destination_airport`
- Train/validation split:
  - 80% train / 20% test
  - `stratify=y`, `random_state=42`

- Models trained (wrapped in `Pipeline`):
  - `RandomForestClassifier(n_estimators=200, class_weight="balanced")`
  - `LogisticRegression(max_iter=2000, class_weight="balanced")`

- **Threshold tuning**:
  - For each model, computes `predict_proba` on the test set.
  - Sweeps thresholds in `[0.10, 0.55]` (step 0.05).
  - For each threshold:
    - Prints accuracy, precision, recall, F1 + classification report.
  - Selects the **best threshold by F1** for that model.

- **Model selection & artefact**:
  - Chooses the model with the highest F1 across thresholds.
  - Saves to `models/flight_delay_model.pkl` as:
    ```python
    {"model": fitted_sklearn_pipeline, "threshold": best_threshold}
    ```

### Extra evaluation (`src/models/evaluate_model.py`)

- Reloads `data/processed/flight_features.csv` and the saved artefact.
- Creates a new 20% test split.
- Applies the stored threshold to `predict_proba`.
- Reports:
  - Classification report
  - Confusion matrix
  - ROC‑AUC
  - PR‑AUC

---

## 6. Serving Layer (FastAPI)

File: `src/service/api.py`

- Loads the saved artefact at startup.
- Schemas in `src/service/schemas.py`:
  - `FlightInput`:
    - `airline: str`
    - `origin_airport: str`
    - `destination_airport: str`
    - `departure_hour: int` (`0–23`)
    - `departure_month: int` (`1–12`)
    - `departure_day_of_week: int` (`0–6`)
    - `distance: float` (miles)
  - `PredictionResponse`:
    - `prediction: "Delayed" | "Not Delayed"`
    - `probability: float` (0–1)

Endpoints:

- `GET /`
  - Health check: `{"message": "Flight Delay Prediction API is running"}`

- `POST /predict_delay`
  - Request body: `FlightInput`
  - Flow:
    - Convert request to single‑row `DataFrame`
    - Call `model.predict_proba` → `P(delayed)`
    - Apply stored threshold
  - Response: `PredictionResponse`

Run locally:

```bash
uvicorn src.service.api:app --reload
```

---

## 7. Streamlit Dashboard

File: `src/ui/app.py`

- Dark, responsive dashboard titled **“Flight Delay Prediction Dashboard”**.
- **Sidebar**:
  - Project Overview card explaining what the app does.
  - “How to use” and “Model notes” sections.
- **Main layout**:
  - Left: flight input form.
  - Right: prediction + risk panel.

Inputs:

- `airline` (IATA airline code, e.g. `AA`, `6E`, `AI`)
- `origin_airport`, `destination_airport` (IATA airport codes, e.g. `DEL`, `BOM`, `JFK`)
- `departure_hour` (0–23)
- `departure_month` (1–12)
- `departure_day_of_week` (0=Monday … 6=Sunday)
- `distance` (miles)

When you click **“Predict Delay”**:

1. Streamlit builds a JSON payload.
2. Sends `POST` to `http://127.0.0.1:8000/predict_delay`.
3. Displays:
   - Prediction (“Delayed” / “Not Delayed”)
   - Probability as a percentage
   - Qualitative risk band (**Low / Medium / High**)
   - Progress bar for probability
   - Input summary table

Run locally (in a separate terminal from FastAPI):

```bash
streamlit run src/ui/app.py
```

---

## 8. Prediction Helper (Script)

File: `src/models/predict.py`

- Function:

```python
def predict_delay(input_data: dict) -> dict:
    """
    {
      "airline": ...,
      "origin_airport": ...,
      "destination_airport": ...,
      "departure_hour": ...,
      "departure_month": ...,
      "departure_day_of_week": ...,
      "distance": ...
    }
    -> {
      "prediction": "Delayed"/"Not Delayed",
      "probability": float
    }
    """
```

- Uses the same artefact (`model` + `threshold`) under the hood.
- Includes a `__main__` example for quick CLI testing.

---

## 9. Tests

File: `tests/test_api.py`

- Basic smoke test:

```python
from fastapi.testclient import TestClient
from src.service.api import app

client = TestClient(app)

def test_root_ok():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()
```

Run tests:

```bash
pytest
```

---

## 10. Setup & Local Usage

### 10.1. Environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# or source venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

### 10.2. Data pipeline

```bash
# Optional, if you have airports.csv
python -m src.data.clean_airports

python -m src.data.clean_flights
python -m src.data.merge_data

python -m src.features.build_features
```

### 10.3. Train model

```bash
python -m src.models.train_model
```

Outputs:

- Console metrics (per threshold, per model).
- Best model + threshold.
- `models/flight_delay_model.pkl`.

### 10.4. Evaluate saved model

```bash
python -m src.models.evaluate_model
```

### 10.5. Run API + UI

Two terminals:

**Terminal 1 – FastAPI**

```bash
venv\Scripts\activate
uvicorn src.service.api:app --reload
```

**Terminal 2 – Streamlit**

```bash
venv\Scripts\activate
streamlit run src/ui/app.py
```

Open the Streamlit URL (usually `http://localhost:8501`) in your browser.

---

## 11. Deploying to Streamlit Cloud

Streamlit Cloud runs only the Streamlit app process. You have two options:

1. **Simplest for deployment (recommended)**  
   - Refactor `src/ui/app.py` to **load the artefact directly** instead of calling `http://127.0.0.1:8000`.
   - You can reuse the logic from `src/models/predict.py` (`load_artefact` + `predict_delay`).
   - This avoids needing to deploy FastAPI separately.

2. **Separate FastAPI deployment**  
   - Deploy `src/service/api.py` to a cloud service (e.g. Render, Railway, Azure Web Apps).
   - Update `API_URL` in `src/ui/app.py` to the public FastAPI URL.
   - Then deploy Streamlit app to Streamlit Cloud pointing to that public API.

For a student portfolio / capstone, **option 1 is usually enough** and easiest.

On Streamlit Cloud you will:

- Connect your GitHub repo.
- Set the app entry point to `src/ui/app.py`.
- Ensure `requirements.txt` is in the repo root.

---

## 12. Possible Future Improvements

- Use enriched `flights_merged.csv` in feature engineering for airport‑level features.
- Add route‑level historical stats (e.g. average delay per route/month).
- Add more evaluation plots (ROC, PR) into a notebook for the report.
- Extend tests:
  - `POST /predict_delay` with a synthetic payload.
  - `predict_delay` function behaviour for edge cases.
- Add simple monitoring/logging of prediction requests.

---

## 13. Credits

- **Author**: Valour Moraes  
- **Footer in app**:  
  `v1.0 • Flight Delay Prediction • Built with FastAPI & Streamlit • Made by Valour Moraes`

