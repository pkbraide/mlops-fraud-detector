"""API contract tests for the fraud detection service.

These tests run against whichever model path is active: the trained joblib
artifact when ``models/fraud_model.joblib`` is present, or the deterministic
rule-based mock when it is not. Assertions that depend on a specific decision
boundary are guarded on the mock flag so the suite stays green in both modes.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_utils import fraud_model

client = TestClient(app)

VALID_MODEL_VERSIONS = {"joblib_model", "mock_rule"}
RESPONSE_KEYS = {"prediction", "is_fraud", "model_version"}


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == "Enterprise Real-Time Fraud Detection Engine"
    assert body["model_version"] in VALID_MODEL_VERSIONS
    assert isinstance(body["mock_mode"], bool)
    assert body["mock_mode"] is fraud_model.is_mock


def test_predict_valid_legit():
    payload = {"amount": 42.50, "distance_from_home": 3.2, "use_chip": 1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == RESPONSE_KEYS
    assert body["prediction"] in (0, 1)
    assert isinstance(body["is_fraud"], bool)
    assert body["is_fraud"] is bool(body["prediction"])
    assert body["model_version"] in VALID_MODEL_VERSIONS
    assert body["model_version"] == fraud_model.model_version

    if fraud_model.is_mock:
        # Small, local, chip-present transaction never trips the mock rule.
        assert body["prediction"] == 0
        assert body["is_fraud"] is False


def test_predict_mock_fraud_trigger():
    payload = {"amount": 5000, "distance_from_home": 120.0, "use_chip": 0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == RESPONSE_KEYS
    assert body["model_version"] == fraud_model.model_version

    if fraud_model.is_mock:
        assert body["model_version"] == "mock_rule"
        assert body["prediction"] == 1
        assert body["is_fraud"] is True
    else:
        assert body["model_version"] == "joblib_model"
        assert body["prediction"] in (0, 1)


def test_predict_response_is_consistent():
    """The same payload must always yield the same decision."""
    payload = {"amount": 1500.0, "distance_from_home": 75.0, "use_chip": 0}
    first = client.post("/predict", json=payload).json()
    second = client.post("/predict", json=payload).json()
    assert first == second


def test_validation_string_in_float_field():
    payload = {"amount": "not_a_number", "distance_from_home": 10.0, "use_chip": 1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_validation_negative_amount():
    payload = {"amount": -25.0, "distance_from_home": 10.0, "use_chip": 1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_validation_zero_amount():
    payload = {"amount": 0, "distance_from_home": 10.0, "use_chip": 1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_validation_bad_use_chip():
    payload = {"amount": 100.0, "distance_from_home": 10.0, "use_chip": 5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("missing_field", ["amount", "distance_from_home", "use_chip"])
def test_validation_missing_field(missing_field):
    payload = {"amount": 100.0, "distance_from_home": 10.0, "use_chip": 1}
    del payload[missing_field]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    errors = response.json()["detail"]
    assert any(missing_field in err.get("loc", []) for err in errors)


def test_validation_empty_body():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_openapi_schema_exposes_example():
    """The Pydantic v2 json_schema_extra example must surface in OpenAPI."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert schema["info"]["title"] == "Enterprise Real-Time Fraud Detection Engine"
    assert schema["info"]["version"] == "1.0.0"

    request_schema = schema["components"]["schemas"]["FraudRequest"]
    assert request_schema["example"] == {
        "amount": 249.99,
        "distance_from_home": 12.4,
        "use_chip": 1,
    }
