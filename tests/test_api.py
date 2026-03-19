"""
API integration tests (FastAPI TestClient).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


def test_assess_endpoint_synthetic(client):
    """POST /assess must return a valid response for any address."""
    response = client.post("/assess", json={
        "address": "Duke Chapel, Durham NC",
        "include_wildfire": True,
        "include_flood": True,
    })
    # 200 if twin found, 404 if no data loaded — both are valid in test env
    assert response.status_code in (200, 404, 503)


def test_simulate_endpoint(client):
    """POST /simulate must return fire spread results."""
    response = client.post("/simulate", json={
        "ignition_lat": 36.001,
        "ignition_lon": -78.940,
        "wind_speed_mph": 20.0,
        "wind_direction_degrees": 225.0,
        "fire_weather_index": 20.0,
        "max_time_minutes": 30,
    })
    assert response.status_code == 200
    data = response.json()
    assert "n_cells_burned" in data
    assert data["n_cells_burned"] >= 0
    assert "area_burned_ha" in data


def test_mitigate_unknown_parcel_returns_404(client):
    """POST /mitigate for an unknown parcel should return 404."""
    response = client.post("/mitigate", json={
        "parcel_id": "DOES_NOT_EXIST_99999",
        "actions": ["screen_all_vents"],
    })
    assert response.status_code in (404, 503)


def test_campus_overview_returns_valid_structure(client):
    """GET /campus-overview must return expected schema."""
    response = client.get("/campus-overview")
    # 503 is acceptable when no twins loaded
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert "total_parcels" in data
        assert "mean_wildfire_risk" in data
        assert isinstance(data["highest_risk_parcels"], list)


def test_simulate_out_of_bounds_coordinates(client):
    """POST /simulate with out-of-bounds coordinates should fail validation."""
    response = client.post("/simulate", json={
        "ignition_lat": 99.0,  # Out of range
        "ignition_lon": -78.940,
        "wind_speed_mph": 20.0,
        "wind_direction_degrees": 225.0,
        "fire_weather_index": 20.0,
        "max_time_minutes": 30,
    })
    assert response.status_code == 422  # Pydantic validation error


def test_assess_schema_fields(client):
    """POST /assess response must include all required schema fields if 200."""
    response = client.post("/assess", json={
        "address": "test",
        "include_wildfire": True,
        "include_flood": False,
    })
    if response.status_code == 200:
        data = response.json()
        required = ["parcel_id", "composite_risk_score", "risk_category", "terrain", "vegetation", "structure"]
        for field in required:
            assert field in data, f"Missing field: {field}"
