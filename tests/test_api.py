"""API contract tests for the fraud detection service.

The suite runs against whichever artifacts are present and must stay green in
every combination (all three, v3 only, v2 only, v1 only, none). Assertions that
depend on a trained decision boundary are conditioned on the relevant
``*_is_mock`` flag; rule-based paths are asserted to return ``confidence: null``.
"""

import io
import json
import os

import pytest
from fastapi.testclient import TestClient

from app.main import MAX_BATCH_ROWS, app
from app.model_utils import FEATURE_COLUMNS_V2, FEATURE_COLUMNS_V3, fraud_model

client = TestClient(app)

SERVICE_NAME = "Enterprise Real-Time Fraud Detection Engine"
VALID_MODEL_VERSIONS = {
    "joblib_model_v3",
    "joblib_model_v2",
    "joblib_model",
    "mock_rule",
}
V1_RESPONSE_KEYS = ["prediction", "is_fraud", "model_version"]
SCORED_KEYS = {"prediction", "is_fraud", "confidence", "threshold", "model_version"}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_CSV = os.path.join(REPO_ROOT, "app", "static", "data", "sample_batch.csv")
PRESETS_JSON = os.path.join(REPO_ROOT, "app", "static", "data", "v3_presets.json")

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
# A real card_transdata hold-out row the v2 forest scores at p = 0.74.
V2_BORDERLINE = {
    "distance_from_home": 100.009,
    "distance_from_last_transaction": 3.4752,
    "ratio_to_median_purchase_price": 1.66,
    "repeat_retailer": 1,
    "used_chip": 0,
    "used_pin_number": 0,
    "online_order": 1,
}

with open(PRESETS_JSON, encoding="utf-8") as _fh:
    V3_PRESETS = {p["id"]: p for p in json.load(_fh)["presets"]}
V3_LEGIT = V3_PRESETS["legit_routine"]["values"]
V3_UNUSUAL = V3_PRESETS["legit_unusual"]["values"]  # p ~ 0.21 with the shipped forest
V3_FRAUD = V3_PRESETS["fraud_caught"]["values"]  # p ~ 0.82 with the shipped forest

with open(SAMPLE_CSV, "rb") as _fh:
    SAMPLE_CSV_BYTES = _fh.read()
SAMPLE_CSV_ROWS = (
    len([ln for ln in SAMPLE_CSV_BYTES.decode().splitlines() if ln.strip()]) - 1
)


def _csv_upload(name, content, mime="text/csv"):
    return {"file": (name, io.BytesIO(content), mime)}


def _assert_scored_shape(body, expected_version):
    assert set(body.keys()) == SCORED_KEYS
    assert body["prediction"] in (0, 1)
    assert body["is_fraud"] is bool(body["prediction"])
    assert body["model_version"] == expected_version
    assert 0 < body["threshold"] < 1
    if expected_version == "mock_rule":
        assert body["confidence"] is None
    else:
        assert isinstance(body["confidence"], float)
        assert 0.0 <= body["confidence"] <= 1.0
        # prediction must be consistent with confidence and threshold
        assert body["prediction"] == int(body["confidence"] >= body["threshold"])


# ------------------------------------------------------------------ pages
@pytest.mark.parametrize(
    "path,title_fragment",
    [
        ("/", f"<title>{SERVICE_NAME}</title>"),
        ("/batch", "<title>Batch scoring"),
        ("/model", "<title>Model card"),
    ],
)
def test_pages_serve_html(path, title_fragment):
    response = client.get(path)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert title_fragment in response.text
    # shared chrome on every page
    assert 'id="model-badge"' in response.text
    assert "github.com/pkbraide/mlops-fraud-detector" in response.text
    assert "/docs" in response.text
    assert "Cold start" in response.text


def test_root_serves_html():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="predict-form"' in response.text
    assert 'name="model" value="v3"' in response.text
    assert 'id="threshold"' in response.text
    assert 'id="v3-presets"' in response.text
    for name in FEATURE_COLUMNS_V2:
        assert f'name="{name}"' in response.text


def test_pages_answer_head_requests():
    for path in ("/", "/batch", "/model", "/health"):
        response = client.head(path)
        assert response.status_code == 200, path


def test_static_assets_served():
    css = client.get("/static/css/style.css")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    for script in ("common.js", "demo.js", "batch.js", "model.js"):
        js = client.get(f"/static/js/{script}")
        assert js.status_code == 200, script
        assert "javascript" in js.headers["content-type"]
    assert client.get("/static/data/sample_batch.csv").status_code == 200
    assert client.get("/static/data/v3_presets.json").status_code == 200


def test_unknown_html_path_returns_404_page():
    response = client.get("/definitely-not-a-page")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Not found" in response.text
    assert "/definitely-not-a-page" in response.text


@pytest.mark.parametrize(
    "path", ["/v2/nope", "/v3/nope", "/v1/predict", "/static/missing.css"]
)
def test_unknown_api_path_returns_404_json(path):
    response = client.get(path)
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Not Found"}


# ------------------------------------------------------------------ health / metrics
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    body = response.json()
    assert body["status"] == "healthy"
    assert body["service"] == SERVICE_NAME
    assert body["version"] == "3.0.0"
    assert body["model_version"] in VALID_MODEL_VERSIONS
    assert body["model_version"] == fraud_model.model_version
    assert body["mock_mode"] is fraud_model.is_mock

    models = body["models"]
    assert models["v3"] == {
        "model_version": fraud_model.v3_model_version,
        "loaded": not fraud_model.v3_is_mock,
    }
    assert models["v2"]["loaded"] is (not fraud_model.v2_is_mock)
    assert models["v1"]["loaded"] is (not fraud_model.v1_is_mock)


def test_health_lists_every_route_with_model_version():
    endpoints = {e["path"]: e for e in client.get("/health").json()["endpoints"]}
    for path in ("/", "/batch", "/model", "/health", "/metrics", "/docs"):
        assert path in endpoints, path
    for path, version, loaded in (
        ("/v3/predict", fraud_model.v3_model_version, not fraud_model.v3_is_mock),
        ("/v2/predict", fraud_model.v2_model_version, not fraud_model.v2_is_mock),
        ("/v2/batch", fraud_model.v2_model_version, not fraud_model.v2_is_mock),
        ("/predict", fraud_model.v1_model_version, not fraud_model.v1_is_mock),
    ):
        assert endpoints[path]["model_version"] == version, path
        assert endpoints[path]["loaded"] is loaded, path
        assert endpoints[path]["artifact"].startswith("models/")
        assert endpoints[path]["fallback_rule"]
    assert endpoints["/predict"]["status"] == "deprecated"
    assert endpoints["/v3/predict"]["status"] == "current"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == "3.0.0"
    models = body["models"]
    assert set(models) == {"v1", "v2", "v3"}

    for key, is_mock, features in (
        ("v3", fraud_model.v3_is_mock, FEATURE_COLUMNS_V3),
        ("v2", fraud_model.v2_is_mock, FEATURE_COLUMNS_V2),
    ):
        block = models[key]
        assert block["loaded"] is (not is_mock)
        assert block["features"] == features
        assert block["n_features"] == len(features)
        assert block["fallback_rule"]
        if is_mock:
            assert block["metrics"] is None
            assert block["dataset"] is None
        else:
            metrics = block["metrics"]
            for name in ("fraud_precision", "fraud_recall", "fraud_f1", "pr_auc"):
                assert 0.0 <= metrics[name] <= 1.0, (key, name)
            cm = metrics["confusion_matrix"]
            assert set(cm) == {"tn", "fp", "fn", "tp"}
            assert sum(cm.values()) == metrics["n_test_rows"]
            assert set(metrics["feature_importances"]) == set(features)
            assert abs(sum(metrics["feature_importances"].values()) - 1.0) < 0.01
            assert block["dataset"]["openml_data_id"]
            assert 0 < block["dataset"]["fraud_prevalence"] < 1

    assert models["v1"]["metrics"] is None
    assert "note" in models["v1"]


@pytest.mark.skipif(
    fraud_model.v3_is_mock or fraud_model.v2_is_mock, reason="needs both artifacts"
)
def test_metrics_v3_is_the_harder_benchmark():
    """Sanity check on the stored numbers: v3 must not look like a toy."""
    models = client.get("/metrics").json()["models"]
    v3, v2 = models["v3"]["metrics"], models["v2"]["metrics"]
    assert v3["fraud_recall"] < 0.95
    assert v3["pr_auc"] < 0.95
    assert v3["pr_auc"] < v2["pr_auc"]
    assert models["v3"]["dataset"]["fraud_prevalence"] < 0.01


# ------------------------------------------------------------------ v1 (legacy)
def test_predict_valid_legit():
    response = client.post(
        "/predict", json={"amount": 42.50, "distance_from_home": 3.2, "use_chip": 1}
    )
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == V1_RESPONSE_KEYS
    assert body["prediction"] in (0, 1)
    assert body["is_fraud"] is bool(body["prediction"])
    assert body["model_version"] == fraud_model.v1_model_version
    if fraud_model.v1_is_mock:
        assert body["prediction"] == 0


def test_predict_mock_fraud_trigger():
    response = client.post(
        "/predict", json={"amount": 5000, "distance_from_home": 120.0, "use_chip": 0}
    )
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == V1_RESPONSE_KEYS
    if fraud_model.v1_is_mock:
        assert body["model_version"] == "mock_rule"
        assert body["prediction"] == 1
    else:
        assert body["model_version"] == "joblib_model"


def test_predict_response_is_consistent():
    payload = {"amount": 1500.0, "distance_from_home": 75.0, "use_chip": 0}
    assert (
        client.post("/predict", json=payload).json()
        == client.post("/predict", json=payload).json()
    )


def test_v1_regression_schema_unchanged():
    """POST /predict must keep the exact v1 contract: no confidence, no threshold."""
    payload = {"amount": 2500, "distance_from_home": 180.0, "use_chip": 0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == V1_RESPONSE_KEYS
    assert body["model_version"] in ("joblib_model", "mock_rule")
    # threshold is not part of the v1 contract and is ignored rather than honoured
    with_threshold = client.post("/predict", json=dict(payload, threshold=0.99))
    assert with_threshold.status_code == 200
    assert with_threshold.json() == body


@pytest.mark.parametrize(
    "payload",
    [
        {"amount": "not_a_number", "distance_from_home": 10.0, "use_chip": 1},
        {"amount": -25.0, "distance_from_home": 10.0, "use_chip": 1},
        {"amount": 0, "distance_from_home": 10.0, "use_chip": 1},
        {"amount": 100.0, "distance_from_home": 10.0, "use_chip": 5},
        {},
    ],
)
def test_v1_validation_rejects_bad_payloads(payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize("missing_field", ["amount", "distance_from_home", "use_chip"])
def test_validation_missing_field(missing_field):
    payload = {"amount": 100.0, "distance_from_home": 10.0, "use_chip": 1}
    del payload[missing_field]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert any(missing_field in err.get("loc", []) for err in response.json()["detail"])


# ------------------------------------------------------------------ v2
def test_v2_predict_valid_legit():
    response = client.post("/v2/predict", json=V2_LEGIT)
    assert response.status_code == 200
    body = response.json()
    _assert_scored_shape(body, fraud_model.v2_model_version)
    assert body["threshold"] == 0.5
    assert body["prediction"] == 0  # legit for both the forest and the rule


def test_v2_predict_valid_fraud():
    response = client.post("/v2/predict", json=V2_FRAUD)
    assert response.status_code == 200
    body = response.json()
    _assert_scored_shape(body, fraud_model.v2_model_version)
    assert body["prediction"] == 1  # fraud for both the forest and the rule
    if not fraud_model.v2_is_mock:
        assert body["confidence"] >= 0.5


def test_v2_confidence_present_or_null_by_mode():
    body = client.post("/v2/predict", json=V2_FRAUD).json()
    if fraud_model.v2_is_mock:
        assert body["confidence"] is None
        assert body["model_version"] == "mock_rule"
    else:
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["model_version"] == "joblib_model_v2"


def test_v2_threshold_changes_borderline_prediction():
    strict = client.post("/v2/predict", json=dict(V2_BORDERLINE, threshold=0.9)).json()
    lenient = client.post("/v2/predict", json=dict(V2_BORDERLINE, threshold=0.1)).json()
    assert strict["threshold"] == 0.9 and lenient["threshold"] == 0.1
    if fraud_model.v2_is_mock:
        # the rule ignores the threshold and never reports a probability
        assert strict["confidence"] is None and lenient["confidence"] is None
        assert strict["prediction"] == lenient["prediction"]
    else:
        assert strict["confidence"] == lenient["confidence"]
        assert 0.1 < strict["confidence"] < 0.9
        assert strict["prediction"] == 0
        assert lenient["prediction"] == 1


def test_v2_default_threshold_is_half():
    body = client.post("/v2/predict", json=V2_LEGIT).json()
    assert body["threshold"] == 0.5


@pytest.mark.parametrize("bad", [0, 1, 1.5, -0.2, "abc"])
def test_v2_threshold_validation(bad):
    response = client.post("/v2/predict", json=dict(V2_LEGIT, threshold=bad))
    assert response.status_code == 422
    assert any("threshold" in err.get("loc", []) for err in response.json()["detail"])


@pytest.mark.parametrize(
    "binary_field", ["repeat_retailer", "used_chip", "used_pin_number", "online_order"]
)
def test_v2_validation_out_of_range_binary(binary_field):
    response = client.post("/v2/predict", json=dict(V2_LEGIT, **{binary_field: 2}))
    assert response.status_code == 422
    assert any(binary_field in err.get("loc", []) for err in response.json()["detail"])


def test_v2_validation_string_in_float_field():
    payload = dict(V2_LEGIT, ratio_to_median_purchase_price="not_a_number")
    response = client.post("/v2/predict", json=payload)
    assert response.status_code == 422


def test_v2_validation_negative_distance():
    response = client.post("/v2/predict", json=dict(V2_LEGIT, distance_from_home=-1.0))
    assert response.status_code == 422


@pytest.mark.parametrize("missing_field", FEATURE_COLUMNS_V2)
def test_v2_validation_missing_field(missing_field):
    payload = dict(V2_LEGIT)
    del payload[missing_field]
    response = client.post("/v2/predict", json=payload)
    assert response.status_code == 422
    assert any(missing_field in err.get("loc", []) for err in response.json()["detail"])


def test_v2_zero_values_are_valid():
    response = client.post("/v2/predict", json={name: 0 for name in FEATURE_COLUMNS_V2})
    assert response.status_code == 200


# ------------------------------------------------------------------ v3
def test_v3_predict_valid_legit():
    response = client.post("/v3/predict", json=V3_LEGIT)
    assert response.status_code == 200
    body = response.json()
    _assert_scored_shape(body, fraud_model.v3_model_version)
    assert body["prediction"] == 0


def test_v3_predict_valid_fraud():
    response = client.post("/v3/predict", json=V3_FRAUD)
    assert response.status_code == 200
    body = response.json()
    _assert_scored_shape(body, fraud_model.v3_model_version)
    assert body["prediction"] == 1  # caught by the forest at 0.5 and by the rule
    if not fraud_model.v3_is_mock:
        assert 0.5 <= body["confidence"] <= 1.0


def test_v3_confidence_present_or_null_by_mode():
    body = client.post("/v3/predict", json=V3_UNUSUAL).json()
    if fraud_model.v3_is_mock:
        assert body["confidence"] is None
        assert body["model_version"] == "mock_rule"
    else:
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["model_version"] == "joblib_model_v3"


def test_v3_threshold_changes_borderline_prediction():
    strict = client.post("/v3/predict", json=dict(V3_FRAUD, threshold=0.9)).json()
    lenient = client.post("/v3/predict", json=dict(V3_FRAUD, threshold=0.1)).json()
    if fraud_model.v3_is_mock:
        assert strict["confidence"] is None and lenient["confidence"] is None
        assert strict["prediction"] == lenient["prediction"]
    else:
        assert 0.1 < strict["confidence"] < 0.9
        assert strict["prediction"] == 0
        assert lenient["prediction"] == 1


def test_v3_presets_are_real_rows_with_ground_truth():
    assert len(V3_PRESETS) == 3
    truths = sorted(p["ground_truth"] for p in V3_PRESETS.values())
    assert truths == [0, 0, 1]
    for preset in V3_PRESETS.values():
        assert list(preset["values"].keys()) == FEATURE_COLUMNS_V3
        response = client.post("/v3/predict", json=preset["values"])
        assert response.status_code == 200
        if not fraud_model.v3_is_mock:
            # the live model must reproduce the probability stored at training time
            assert (
                abs(response.json()["confidence"] - preset["holdout_probability"])
                < 1e-6
            )


@pytest.mark.parametrize("bad", [0, 1, 1.5, "abc"])
def test_v3_threshold_validation(bad):
    response = client.post("/v3/predict", json=dict(V3_LEGIT, threshold=bad))
    assert response.status_code == 422


def test_v3_bad_feature_count_missing():
    payload = dict(V3_FRAUD)
    del payload["V28"]
    response = client.post("/v3/predict", json=payload)
    assert response.status_code == 422
    assert any("V28" in err.get("loc", []) for err in response.json()["detail"])


def test_v3_bad_feature_count_extra():
    response = client.post("/v3/predict", json=dict(V3_FRAUD, V29=0.1))
    assert response.status_code == 422
    assert any("V29" in err.get("loc", []) for err in response.json()["detail"])


def test_v3_string_in_float_field():
    response = client.post("/v3/predict", json=dict(V3_FRAUD, V14="abc"))
    assert response.status_code == 422


def test_v3_empty_body():
    assert client.post("/v3/predict", json={}).status_code == 422


# ------------------------------------------------------------------ v2 batch
def test_batch_valid_csv():
    response = client.post(
        "/v2/batch", files=_csv_upload("sample.csv", SAMPLE_CSV_BYTES)
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == fraud_model.v2_model_version
    assert body["threshold"] == 0.5
    summary = body["summary"]
    assert summary["total"] == SAMPLE_CSV_ROWS == len(body["rows"])
    assert summary["flagged"] == sum(r["prediction"] for r in body["rows"])
    assert summary["flag_rate"] == round(summary["flagged"] / summary["total"], 4)
    assert summary["processing_ms"] >= 0
    assert [r["row"] for r in body["rows"]] == list(range(1, SAMPLE_CSV_ROWS + 1))
    for row in body["rows"]:
        assert set(row["inputs"]) == set(FEATURE_COLUMNS_V2)
        assert row["is_fraud"] is bool(row["prediction"])
        if fraud_model.v2_is_mock:
            assert row["confidence"] is None
        else:
            assert 0.0 <= row["confidence"] <= 1.0
            assert row["prediction"] == int(row["confidence"] >= 0.5)
    # the sample file carries five confirmed frauds; the forest finds them all
    if not fraud_model.v2_is_mock:
        assert summary["flagged"] == 5


def test_batch_threshold_query_param():
    response = client.post(
        "/v2/batch?threshold=0.9", files=_csv_upload("sample.csv", SAMPLE_CSV_BYTES)
    )
    assert response.status_code == 200
    assert response.json()["threshold"] == 0.9
    bad = client.post(
        "/v2/batch?threshold=1.5", files=_csv_upload("s.csv", SAMPLE_CSV_BYTES)
    )
    assert bad.status_code == 422


def test_batch_missing_column_names_it():
    content = b"distance_from_home,used_chip\n1.0,1\n"
    response = client.post("/v2/batch", files=_csv_upload("bad.csv", content))
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Missing required column(s)" in detail
    for name in ("distance_from_last_transaction", "online_order"):
        assert name in detail


def test_batch_non_csv_rejected():
    response = client.post(
        "/v2/batch", files=_csv_upload("notes.txt", b"hello", "text/plain")
    )
    assert response.status_code == 400
    assert ".csv" in response.json()["detail"]


def test_batch_bad_value_names_column_and_line():
    lines = SAMPLE_CSV_BYTES.decode().splitlines()
    parts = lines[1].split(",")
    parts[2] = "abc"
    lines[1] = ",".join(parts)
    response = client.post(
        "/v2/batch", files=_csv_upload("bad.csv", "\n".join(lines).encode())
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "ratio_to_median_purchase_price" in detail
    assert "line(s) [2]" in detail


def test_batch_binary_out_of_range_rejected():
    lines = SAMPLE_CSV_BYTES.decode().splitlines()
    parts = lines[1].split(",")
    parts[FEATURE_COLUMNS_V2.index("used_chip")] = "2"
    lines[1] = ",".join(parts)
    response = client.post(
        "/v2/batch", files=_csv_upload("bad.csv", "\n".join(lines).encode())
    )
    assert response.status_code == 400
    assert "used_chip" in response.json()["detail"]


def test_batch_row_limit_and_size_limit():
    header = ",".join(FEATURE_COLUMNS_V2)
    too_many = (
        header + "\n" + "\n".join(["1,1,1,1,1,0,0"] * (MAX_BATCH_ROWS + 1)) + "\n"
    )
    response = client.post(
        "/v2/batch", files=_csv_upload("rows.csv", too_many.encode())
    )
    assert response.status_code == 400
    assert str(MAX_BATCH_ROWS) in response.json()["detail"]

    too_big = client.post(
        "/v2/batch", files=_csv_upload("big.csv", b"a" * (2 * 1024 * 1024 + 1))
    )
    assert too_big.status_code == 413


def test_batch_empty_and_header_only():
    assert client.post("/v2/batch", files=_csv_upload("e.csv", b"")).status_code == 400
    header_only = (",".join(FEATURE_COLUMNS_V2) + "\n").encode()
    assert (
        client.post("/v2/batch", files=_csv_upload("h.csv", header_only)).status_code
        == 400
    )


def test_batch_never_leaks_stack_trace():
    response = client.post(
        "/v2/batch", files=_csv_upload("junk.csv", b"\xff\xfe\x00garbage")
    )
    assert response.status_code == 400
    assert "Traceback" not in response.text
    assert "pandas" not in response.text.lower()


# ------------------------------------------------------------------ OpenAPI
def test_openapi_schema():
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == SERVICE_NAME
    assert schema["info"]["version"] == "3.0.0"
    paths = schema["paths"]
    for path in (
        "/predict",
        "/v2/predict",
        "/v3/predict",
        "/v2/batch",
        "/health",
        "/metrics",
    ):
        assert path in paths, path
    for page in ("/", "/batch", "/model"):
        assert page not in paths, page
    assert paths["/predict"]["post"]["deprecated"] is True
    assert "/v3/predict" in paths["/predict"]["post"]["description"]

    schemas = schema["components"]["schemas"]
    assert schemas["FraudRequest"]["example"] == {
        "amount": 249.99,
        "distance_from_home": 12.4,
        "use_chip": 1,
    }
    assert set(schemas["FraudRequestV2"]["required"]) == set(FEATURE_COLUMNS_V2)
    assert schemas["FraudRequestV2"]["properties"]["threshold"]["default"] == 0.5
    assert set(schemas["FraudRequestV3"]["required"]) == set(FEATURE_COLUMNS_V3)
    assert schemas["FraudRequestV3"]["additionalProperties"] is False
    assert "confidence" in schemas["ScoredResponse"]["properties"]
    assert "confidence" not in schemas["FraudResponse"]["properties"]
