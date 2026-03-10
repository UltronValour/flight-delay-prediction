"""
Training script for the Flight Delay Prediction project.

Loads engineered features from data/processed/flight_features.csv,
trains classification models, evaluates them, and saves the best model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("data/processed/flight_features.csv")
MODEL_PATH = Path("models/flight_delay_model.pkl")


def load_data(path: Path) -> pd.DataFrame:
    """Load the processed feature dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found at: {path.resolve()}")

    df = pd.read_csv(path, low_memory=False)

    # Fix mixed types in categorical columns
    for col in ["airline", "origin_airport", "destination_airport"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


def prepare_features_and_target(df: pd.DataFrame):
    """Separate features (X) and target (y)."""
    if "delayed" not in df.columns:
        raise KeyError("Expected target column 'delayed' not found in dataset.")

    X = df.drop(columns=["delayed"])
    y = df["delayed"]
    return X, y


def build_preprocessor(categorical_features: list[str]) -> ColumnTransformer:
    """Create a column transformer with one-hot encoding for categorical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )
    return preprocessor


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """Build model pipelines for RandomForest and LogisticRegression."""
    models: dict[str, Pipeline] = {}

    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )

    log_reg_clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )

    models["RandomForestClassifier"] = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf_clf),
        ]
    )

    models["LogisticRegression"] = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", log_reg_clf),
        ]
    )

    return models


def evaluate_at_threshold(
    name: str, y_test, y_proba, threshold: float
) -> dict[str, float]:
    """Evaluate model predictions at a specific probability threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n=== Evaluation for {name} at threshold {threshold:.2f} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def save_model(model: Pipeline, path: Path) -> None:
    """Persist the best model pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"\nBest model saved to: {path.resolve()}")


def main() -> None:
    """Run the training workflow."""
    print(f"Loading processed data from: {DATA_PATH}")
    df = load_data(DATA_PATH)
    # Sample dataset for faster training
    df = df.sample(n=500000, random_state=42)
    print(f"Dataset shape after sampling: {df.shape}")

    # Categorical columns as specified
    categorical_features = [
        "airline",
        "origin_airport",
        "destination_airport",
    ]

    X, y = prepare_features_and_target(df)

    # Ensure categorical columns are consistently treated as strings
    for col in categorical_features:
        X[col] = X[col].astype(str)

    # Train-test split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    preprocessor = build_preprocessor(categorical_features)
    models = build_models(preprocessor)

    print(
        "Class imbalance handling enabled: using class_weight='balanced' "
        "for RandomForestClassifier and LogisticRegression.\n"
        "Will also tune the decision threshold based on F1-score."
    )

    best_model_name = None
    best_model = None
    best_f1 = -1.0
    best_threshold = 0.5

    # Thresholds to try for converting probabilities to class labels
    threshold_grid = np.arange(0.1, 0.6, 0.05)

    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        # Use predicted probabilities for the positive class
        y_proba = model.predict_proba(X_test)[:, 1]

        model_best_f1 = -1.0
        model_best_threshold = 0.5

        print("\nThreshold tuning (based on F1-score):")
        for thr in threshold_grid:
            metrics = evaluate_at_threshold(name, y_test, y_proba, thr)
            if metrics["f1"] > model_best_f1:
                model_best_f1 = metrics["f1"]
                model_best_threshold = thr

        print(
            f"\nBest threshold for {name}: {model_best_threshold:.2f} "
            f"(F1 = {model_best_f1:.4f})"
        )

        if model_best_f1 > best_f1:
            best_f1 = model_best_f1
            best_threshold = model_best_threshold
            best_model_name = name
            best_model = model

    print(
        f"\nBest model based on F1 score: {best_model_name} "
        f"(F1 = {best_f1:.4f} at threshold {best_threshold:.2f})"
    )

    if best_model is not None:
        # Save both the trained pipeline and the chosen decision threshold
        artefact = {
            "model": best_model,
            "threshold": best_threshold,
        }
        save_model(artefact, MODEL_PATH)
    else:
        print("No model was trained successfully; nothing to save.")


if __name__ == "__main__":
    main()
