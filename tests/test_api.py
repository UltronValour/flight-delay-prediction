"""
Smoke tests for the Flight Delay Prediction API.

These tests verify that the FastAPI application starts correctly and that
the root endpoint responds with a 200 status code.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.service.api import app


client = TestClient(app)


def test_root_ok() -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "message" in body

