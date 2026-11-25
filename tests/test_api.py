"""API tests with FastAPI's TestClient. They use the committed model file,
not the CSVs."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

GOOD_RIDE = {
    "distance_km": 8.5,
    "wait_min": 12.0,
    "passengers": 2,
    "hour": 18,
    "luggage_bags": 1,
    "fuel_price_rs": 99.5,
    "booking": "app",
    "zone": "Station Area",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_a_fare():
    response = client.post("/predict", json=GOOD_RIDE)
    assert response.status_code == 200
    assert response.json()["fare_rs"] > 0


def test_wait_above_55_is_rejected():
    response = client.post("/predict", json={**GOOD_RIDE, "wait_min": 60})
    assert response.status_code == 422


def test_negative_distance_is_rejected():
    response = client.post("/predict", json={**GOOD_RIDE, "distance_km": -1})
    assert response.status_code == 422


def test_missing_field_is_rejected():
    ride = dict(GOOD_RIDE)
    del ride["zone"]
    response = client.post("/predict", json=ride)
    assert response.status_code == 422