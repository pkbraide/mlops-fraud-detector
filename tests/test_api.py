"""API contract tests for the fraud detection service.

These tests run against whichever model paths are active: the trained joblib
artifacts when ``models/fraud_model_v2.joblib`` / ``models/fraud_model.joblib``
are present, or the deterministic rule-based fallbacks when they are not.
Assertions that depend on a specific decision boundary are guarded on the
relevant mock flag so the suite stays green in every combination.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_utils import FEATURE_COLUMNS_V2, fraud_model

client = TestClient(app)

SERVICE_NAME = "Enterprise Real-Time Fraud Detection Engine"
VALID_MODEL_VERSIONS = {"joblib_model_v2", "joblib_model", "mock_rule"}
RESPONSE_KEYS = {"prediction", "is_fraud", "model_version"}

V2_LEGIT = {
    "distance_from_home": 5.2,
    "distance_from_last_transaction": 0.4,
    "ratio_to_median_purchase_price": 0.9,
    "repeat_retailer": 1,
    "used_chip": 1,
    "used_pin_number": 0,
    "online_order": 0,
}
V2_FRAUD = {
    "distance_from_home": 210.5,
    "distance_from_last_transaction": 12.3,
    "ratio_to_median_purchase_price": 6.2,
    "repeat_retailer": 0,
    "used_chip": 0,
    "used_pin_number": 0,
    "online_order": 1,
}


# ----------------------------------------------------------------- health / UI
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == SERVICE_NAME
    assert body["version"] == "2.0.0"
    assert body["model_version"] in VALID_MODEL_VERSIONS
    assert body["model_version"] == fraud_model.model_version
    assert isinstance(body["mock_mode"], bool)
    assert body["mock_mode"] is fraud_model.is_mock

    paths = {e["path"]: e for e in body["endpoints"]}
    assert paths["/v2/predict"]["status"] == "current"
    assert paths["/predict"]["status"] == "deprecated"
    assert paths["/v2/predict"]["model_version"] == fraud_model.v2_model_version
    assert paths["/predict"]["model_version"] == fraud_model.v1_model_version


def test_root_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert f"<title>{SERVICE_NAME}</title>" in response.text
    assert 'id="predict-form"' in response.text
    assert "/static/app.js" in response.text
    # Every v2 feature has a form control.
    for name in FEATURE_COLUMNS_V2:
        assert f'name="{name}"' in response.text


def test_static_assets_served():
    css = client.get("/static/style.css")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]

    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert "/v2/predict" in js.text


# ------------------------------------------------------------------ v1 (legacy)
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
    assert body["model_version"] == fraud_model.v1_model_version

    if fraud_model.v1_is_mock:
        # Small, local, chip-present transaction never trips the mock rule.
        assert body["prediction"] == 0
        assert body["is_fraud"] is False


def test_predict_mock_fraud_trigger():
    payload = {"amount": 5000, "distance_from_home": 120.0, "use_chip": 0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == RESPONSE_KEYS
    assert body["model_version"] == fraud_model.v1_model_version

    if fraud_model.v1_is_mock:
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


def test_v1_regression_schema_unchanged():
    """POST /predict must keep the exact v1 request/response contract."""
    payload = {"amount": 2500, "distance_from_home": 180.0, "use_chip": 0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert list(body.keys()) == ["prediction", "is_fraud", "model_version"]
    assert isinstance(body["prediction"], int)
    assert isinstance(body["is_fraud"], bool)
    assert body["model_version"] in ("joblib_model", "mock_rule")
    # v1 is never served by the v2 model.
    assert body["model_version"] != "joblib_model_v2"


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


# ------------------------------------------------------------------ v2 (current)
def test_v2_predict_valid_legit():
    response = client.post("/v2/predict", json=V2_LEGIT)
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == RESPONSE_KEYS
    assert body["model_version"] == fraud_model.v2_model_version
    assert body["model_version"] in ("joblib_model_v2", "mock_rule")
    assert body["is_fraud"] is bool(body["prediction"])
    # This example is classified legitimate by both the v2 model and the rule.
    assert body["prediction"] == 0
    assert body["is_fraud"] is False


def test_v2_predict_valid_fraud():
    response = client.post("/v2/predict", json=V2_FRAUD)
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == RESPONSE_KEYS
    assert body["model_version"] == fraud_model.v2_model_version
    assert body["is_fraud"] is bool(body["prediction"])

    if fraud_model.v2_is_mock:
        assert body["model_version"] == "mock_rule"
        assert body["prediction"] == 1
    else:
        assert body["model_version"] == "joblib_model_v2"
        # Verified against the shipped artifact; re-check if the model is retrained.
        assert body["prediction"] == 1
    assert body["is_fraud"] is True


def test_v2_predict_response_is_consistent():
    first = client.post("/v2/predict", json=V2_FRAUD).json()
    second = client.post("/v2/predict", json=V2_FRAUD).json()
    assert first == second


@pytest.mark.parametrize(
    "binary_field", ["repeat_retailer", "used_chip", "used_pin_number", "online_order"]
)
def test_v2_validation_out_of_range_binary(binary_field):
    payload = dict(V2_LEGIT)
    payload[binary_field] = 2
    response = client.post("/v2/predict", json=payload)
    assert response.status_code == 422

    errors = response.json()["detail"]
    assert any(binary_field in err.get("loc", []) for err in errors)


def test_v2_validation_string_in_float_field():
    payload = dict(V2_LEGIT)
    payload["ratio_to_median_purchase_price"] = "not_a_number"
    response = client.post("/v2/predict", json=payload)
    assert response.status_code == 422

    errors = response.json()["detail"]
    assert any("ratio_to_median_purchase_price" in err.get("loc", []) for err in errors)


def test_v2_validation_negative_distance():
    payload = dict(V2_LEGIT)
    payload["distance_from_home"] = -1.0
    response = client.post("/v2/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("missing_field", FEATURE_COLUMNS_V2)
def test_v2_validation_missing_field(missing_field):
    payload = dict(V2_LEGIT)
    del payload[missing_field]
    response = client.post("/v2/predict", json=payload)
    assert response.status_code == 422

    errors = response.json()["detail"]
    assert any(missing_field in err.get("loc", []) for err in errors)


def test_v2_zero_values_are_valid():
    """ge=0 must accept exact zeros (gt=0 would wrongly reject them)."""
    payload = {name: 0 for name in FEATURE_COLUMNS_V2}
    response = client.post("/v2/predict", json=payload)
    assert response.status_code == 200


# ------------------------------------------------------------------- OpenAPI
def test_openapi_schema_exposes_example():
    """Pydantic v2 json_schema_extra examples must surface in OpenAPI."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert schema["info"]["title"] == SERVICE_NAME
    assert schema["info"]["version"] == "2.0.0"

    v1_schema = schema["components"]["schemas"]["FraudRequest"]
    assert v1_schema["example"] == {
        "amount": 249.99,
        "distance_from_home": 12.4,
        "use_chip": 1,
    }

    v2_schema = schema["components"]["schemas"]["FraudRequestV2"]
    assert v2_schema["example"] == V2_LEGIT
    assert set(v2_schema["required"]) == set(FEATURE_COLUMNS_V2)


def test_openapi_marks_v1_deprecated():
    schema = client.get("/openapi.json").json()
    assert schema["paths"]["/predict"]["post"]["deprecated"] is True
    assert "/v2/predict" in schema["paths"]["/predict"]["post"]["description"]
    assert schema["paths"]["/v2/predict"]["post"].get("deprecated") is not True
    # The HTML root is intentionally excluded from the API schema.
    assert "/" not in schema["paths"]
