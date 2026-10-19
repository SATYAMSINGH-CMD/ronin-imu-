"""
Unit and Integration Tests for FastAPI RoNIN Serving Engine
Tests:
1. /health endpoint and metadata
2. /predict/window velocity estimation and latency budget
3. Error handling on invalid tensor inputs
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.serving.api import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verifies service health check and latency SLA parameters."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "RoNIN Inertial Navigation"
    assert data["num_parameters"] > 300000
    assert data["latency_budget_ms"] == 5.0


def test_predict_window_success(client):
    """Verifies that a valid (6, 200) tensor produces a fast, valid velocity prediction."""
    # Synthetic normal walking window
    dummy_features = np.random.normal(0, 0.5, size=(6, 200)).tolist()

    response = client.post("/predict/window", json={"features": dummy_features})
    assert response.status_code == 200
    data = response.json()

    assert "velocity_hif" in data
    assert len(data["velocity_hif"]) == 2
    assert "speed_mps" in data
    assert "latency_ms" in data
    assert data["latency_ms"] <= 15.0  # Safe threshold for unit test runners


def test_predict_window_invalid_shape(client):
    """Verifies that malformed tensor shapes trigger HTTP 400."""
    bad_features = np.zeros((4, 150)).tolist()  # Wrong channel and sample count
    response = client.post("/predict/window", json={"features": bad_features})
    assert response.status_code == 400
    assert "Expected tensor shape (6, 200)" in response.json()["detail"]
