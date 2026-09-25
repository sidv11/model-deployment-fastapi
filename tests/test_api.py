"""
Automated tests for the API.

Run:
    pytest -v

These use FastAPI's TestClient, which calls the app directly in
memory. No server needs to be running for these tests. This is the
difference between "I clicked around and it seemed fine" and "there
is a repeatable check that proves it works."
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    # Using TestClient as a context manager triggers the app's lifespan
    # startup event (loading the model), the same way a real server would
    # on boot. Without this, the model never loads and every prediction fails.
    with TestClient(app) as c:
        yield c


SAFE_APPLICANT = {
    "age": 45,
    "annual_income": 120000,
    "employment_years": 15,
    "credit_score": 790,
    "loan_amount": 15000,
    "debt_to_income": 0.12,
    "num_late_payments": 0,
    "has_cosigner": 1,
}

RISKY_APPLICANT = {
    "age": 24,
    "annual_income": 28000,
    "employment_years": 0.5,
    "credit_score": 520,
    "loan_amount": 24000,
    "debt_to_income": 0.65,
    "num_late_payments": 6,
    "has_cosigner": 0,
}


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_expected_shape(client):
    response = client.post("/predict", json=SAFE_APPLICANT)
    assert response.status_code == 200
    body = response.json()
    for key in ("default_probability", "will_default", "risk_band", "decision_threshold", "model_version"):
        assert key in body
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_band"] in ("low", "medium", "high")


def test_safe_applicant_scores_lower_than_risky_applicant(client):
    """The model should agree with common sense on two very different applicants."""
    safe = client.post("/predict", json=SAFE_APPLICANT).json()
    risky = client.post("/predict", json=RISKY_APPLICANT).json()
    assert safe["default_probability"] < risky["default_probability"]


def test_predict_rejects_missing_field(client):
    incomplete = dict(SAFE_APPLICANT)
    del incomplete["credit_score"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422  # FastAPI's validation error code


def test_predict_rejects_out_of_range_credit_score(client):
    bad = dict(SAFE_APPLICANT)
    bad["credit_score"] = 999  # above the valid 300-850 range
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_rejects_negative_income(client):
    bad = dict(SAFE_APPLICANT)
    bad["annual_income"] = -500
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_batch_predict_matches_individual_calls(client):
    batch_response = client.post("/predict/batch", json=[SAFE_APPLICANT, RISKY_APPLICANT])
    assert batch_response.status_code == 200
    predictions = batch_response.json()["predictions"]
    assert len(predictions) == 2

    single_safe = client.post("/predict", json=SAFE_APPLICANT).json()
    single_risky = client.post("/predict", json=RISKY_APPLICANT).json()
    assert predictions[0]["default_probability"] == single_safe["default_probability"]
    assert predictions[1]["default_probability"] == single_risky["default_probability"]


def test_batch_predict_rejects_empty_list(client):
    response = client.post("/predict/batch", json=[])
    assert response.status_code == 400


def test_root_endpoint_is_reachable(client):
    response = client.get("/")
    assert response.status_code == 200
