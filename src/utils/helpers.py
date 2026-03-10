"""
Utility helpers for the Flight Delay Prediction project.

These are small, reusable functions that are shared across modules.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def dict_to_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a flat dictionary into a single-row pandas DataFrame.

    Useful for turning JSON/payloads into model-ready tabular format.
    """
    return pd.DataFrame([data])

