"""
Additional evaluation utilities for the Flight Delay Prediction model.

Loads the processed features and saved model artefact, reconstructs a
train/test split, and reports richer metrics such as confusion matrix,
ROC-AUC, and PR-AUC at the chosen decision threshold.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/flight_features.csv")
ARTEFACT_PATH = Path("models/flight_delay_model.pkl")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found at: {path.resolve()}")
    return pd.read_csv(path, low_memory=False)


def load_artefact(path: Path = ARTEFACT_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Model artefact not found at: {path.resolve()}")
    artefact = joblib.load(path)
    if not isinstance(artefact, dict) or "model" not in artefact or "threshold" not in artefact:
        raise ValueError("Model artefact must be a dict with 'model' and 'threshold'.")
    return artefact


def main() -> None:
    print(f"Loading data from: {DATA_PATH}")
    df = load_data()

    if "delayed" not in df.columns:
        raise KeyError("Expected target column 'delayed' not found in dataset.")

    X = df.drop(columns=["delayed"])
    y = df["delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print(f"Test set shape: {X_test.shape}")

    artefact = load_artefact()
    model = artefact["model"]
    threshold: float = float(artefact["threshold"])

    # Get probabilities and apply the stored threshold
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("\n=== Evaluation at stored threshold ===")
    print(f"Threshold: {threshold:.2f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    # Threshold-independent metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"\nROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC : {pr_auc:.4f}")

    # Optional: text-based confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    print("\nConfusion matrix labels (row=true, col=pred):", disp.display_labels)


if __name__ == "__main__":
    main()
