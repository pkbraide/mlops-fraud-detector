"""FastAPI entrypoint for the Enterprise Real-Time Fraud Detection Engine."""

import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field, create_model
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.model_utils import (
    BINARY_COLUMNS_V2,
    DEFAULT_THRESHOLD,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V3,
    NON_NEGATIVE_COLUMNS_V2,
    fraud_model,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

SERVICE_NAME = "Enterprise Real-Time Fraud Detection Engine"
SERVICE_VERSION = "3.0.0"
SERVICE_DESCRIPTION = (
    "Real-time card-fraud scoring with a browser UI at /, batch scoring at /batch "
    "and a model card at /model. POST /v3/predict scores a transaction with a "
    "class-weighted RandomForest trained on the ULB creditcard benchmark (real "
    "anonymised transactions, 0.172% fraud). POST /v2/predict and /v2/batch use "
    "the model trained on the synthetic card_transdata set (7 behavioural features). "
    "POST /predict is the deprecated v1 contract kept byte-identical. Every scored "
    "response carries `confidence` (positive-class probability, null on the "
    "rule-based fallback), the `threshold` applied, and the `model_version` that "
    "produced it. GET /metrics exposes every number the model card renders, read "
    "from the artifacts' stored metadata."
)
REPO_URL = "https://github.com/pkbraide/mlops-fraud-detector"
LIVE_URL = "https://mlops-fraud-detector.onrender.com"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")
PRESETS_PATH = os.path.join(STATIC_DIR, "data", "v3_presets.json")
SAMPLE_CSV_URL = "/static/data/sample_batch.csv"

MAX_BATCH_BYTES = 2 * 1024 * 1024
MAX_BATCH_ROWS = 1000

API_PREFIXES = (
    "/v1",
    "/v2",
    "/v3",
    "/predict",
    "/health",
    "/metrics",
    "/static",
    "/openapi",
    "/docs",
    "/redoc",
)

app = FastAPI(
    title=SERVICE_NAME,
    description=SERVICE_DESCRIPTION,
    version=SERVICE_VERSION,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _load_presets() -> Dict[str, Any]:
    """Real hold-out rows written by notebooks/train_v3.ipynb for the v3 demo."""
    try:
        with open(PRESETS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        logger.warning(
            "v3 presets file not found at %s; UI presets disabled.", PRESETS_PATH
        )
        return {"features": FEATURE_COLUMNS_V3, "presets": []}
    except (OSError, ValueError) as exc:
        logger.warning(
            "Could not read v3 presets (%s: %s); UI presets disabled.",
            type(exc).__name__,
            exc,
        )
        return {"features": FEATURE_COLUMNS_V3, "presets": []}
    data.setdefault("features", FEATURE_COLUMNS_V3)
    data.setdefault("presets", [])
    return data


V3_PRESETS = _load_presets()


# --------------------------------------------------------------------- schemas
THRESHOLD_DESCRIPTION = (
    "Decision threshold applied to the fraud probability. Lower it to trade "
    "precision for recall. Must be strictly between 0 and 1."
)


class FraudRequest(BaseModel):
    """v1 inbound transaction features (deprecated; kept byte-identical)."""

    amount: float = Field(..., gt=0, description="Transaction amount in USD.")
    distance_from_home: float = Field(
        ..., gt=0, description="Distance in km between merchant and cardholder home."
    )
    use_chip: int = Field(
        ..., ge=0, le=1, description="1 if the EMV chip was used, 0 for swipe/manual."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 249.99,
                "distance_from_home": 12.4,
                "use_chip": 1,
            }
        }
    }


class FraudResponse(BaseModel):
    """v1 scoring result (deprecated; kept byte-identical)."""

    prediction: int = Field(..., description="1 = fraud, 0 = legitimate.")
    is_fraud: bool = Field(..., description="Convenience boolean of `prediction`.")
    model_version: str = Field(..., description="Either 'joblib_model' or 'mock_rule'.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 0,
                "is_fraud": False,
                "model_version": "joblib_model",
            }
        }
    }


class FraudRequestV2(BaseModel):
    """v2 inbound transaction features (card_transdata schema)."""

    distance_from_home: float = Field(
        ..., ge=0, description="Km between transaction location and cardholder home."
    )
    distance_from_last_transaction: float = Field(
        ..., ge=0, description="Km between this and the previous transaction."
    )
    ratio_to_median_purchase_price: float = Field(
        ...,
        ge=0,
        description="Transaction price divided by the cardholder's median purchase.",
    )
    repeat_retailer: int = Field(
        ..., ge=0, le=1, description="1 if the retailer was used before by this card."
    )
    used_chip: int = Field(..., ge=0, le=1, description="1 if the EMV chip was used.")
    used_pin_number: int = Field(..., ge=0, le=1, description="1 if a PIN was entered.")
    online_order: int = Field(
        ..., ge=0, le=1, description="1 for a card-not-present online purchase."
    )
    threshold: float = Field(
        DEFAULT_THRESHOLD, gt=0, lt=1, description=THRESHOLD_DESCRIPTION
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "distance_from_home": 5.2,
                "distance_from_last_transaction": 0.4,
                "ratio_to_median_purchase_price": 0.9,
                "repeat_retailer": 1,
                "used_chip": 1,
                "used_pin_number": 0,
                "online_order": 0,
                "threshold": 0.5,
            }
        }
    }


def _v3_example() -> Dict[str, Any]:
    presets = V3_PRESETS.get("presets") or []
    if presets:
        example = dict(presets[0]["values"])
    else:
        example = {name: 0.0 for name in FEATURE_COLUMNS_V3}
    example["threshold"] = DEFAULT_THRESHOLD
    return example


_v3_fields: Dict[str, Any] = {
    name: (
        float,
        Field(
            ...,
            description=(
                "Transaction amount (EUR)."
                if name == "Amount"
                else f"PCA component {name} of the anonymised ULB feature set."
            ),
        ),
    )
    for name in FEATURE_COLUMNS_V3
}
_v3_fields["threshold"] = (
    float,
    Field(DEFAULT_THRESHOLD, gt=0, lt=1, description=THRESHOLD_DESCRIPTION),
)

FraudRequestV3 = create_model(  # type: ignore[call-overload]
    "FraudRequestV3",
    __config__=ConfigDict(
        extra="forbid",
        json_schema_extra={"example": _v3_example()},
    ),
    __doc__=(
        "v3 inbound transaction features: the 28 PCA components V1..V28 and Amount "
        "from the ULB creditcard benchmark. Exactly these fields are required; "
        "unknown fields are rejected."
    ),
    **_v3_fields,
)


class ScoredResponse(BaseModel):
    """Scoring result for the v2 and v3 routes."""

    prediction: int = Field(..., description="1 = fraud, 0 = legitimate.")
    is_fraud: bool = Field(..., description="Convenience boolean of `prediction`.")
    confidence: Optional[float] = Field(
        ...,
        description=(
            "Model probability of the positive (fraud) class, rounded to 4 dp. "
            "null when the rule-based fallback served the request."
        ),
    )
    threshold: float = Field(..., description="Decision threshold that was applied.")
    model_version: str = Field(
        ...,
        description="One of 'joblib_model_v3', 'joblib_model_v2' or 'mock_rule'.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 1,
                "is_fraud": True,
                "confidence": 0.82,
                "threshold": 0.5,
                "model_version": "joblib_model_v3",
            }
        }
    }


class BatchRow(BaseModel):
    """One scored CSV row."""

    row: int = Field(..., description="1-based data row number (header excluded).")
    inputs: Dict[str, float] = Field(
        ..., description="The seven v2 features as parsed."
    )
    prediction: int
    is_fraud: bool
    confidence: Optional[float] = Field(
        ..., description="Fraud probability, null on the rule-based fallback."
    )


class BatchSummary(BaseModel):
    """Aggregate view of a batch."""

    total: int
    flagged: int
    flag_rate: float = Field(..., description="flagged / total, rounded to 4 dp.")
    processing_ms: float
    threshold: float
    model_version: str


class BatchResponse(BaseModel):
    """Response of POST /v2/batch."""

    model_version: str
    threshold: float
    summary: BatchSummary
    rows: List[BatchRow]

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_version": "joblib_model_v2",
                "threshold": 0.5,
                "summary": {
                    "total": 2,
                    "flagged": 1,
                    "flag_rate": 0.5,
                    "processing_ms": 3.2,
                    "threshold": 0.5,
                    "model_version": "joblib_model_v2",
                },
                "rows": [
                    {
                        "row": 1,
                        "inputs": {
                            "distance_from_home": 5.2,
                            "distance_from_last_transaction": 0.4,
                            "ratio_to_median_purchase_price": 0.9,
                            "repeat_retailer": 1,
                            "used_chip": 1,
                            "used_pin_number": 0,
                            "online_order": 0,
                        },
                        "prediction": 0,
                        "is_fraud": False,
                        "confidence": 0.0,
                    },
                    {
                        "row": 2,
                        "inputs": {
                            "distance_from_home": 210.5,
                            "distance_from_last_transaction": 12.3,
                            "ratio_to_median_purchase_price": 6.2,
                            "repeat_retailer": 0,
                            "used_chip": 0,
                            "used_pin_number": 0,
                            "online_order": 1,
                        },
                        "prediction": 1,
                        "is_fraud": True,
                        "confidence": 1.0,
                    },
                ],
            }
        }
    }


# ---------------------------------------------------------------- page helpers
NAV_ITEMS = [
    {"href": "/", "label": "Demo", "page": "demo"},
    {"href": "/batch", "label": "Batch", "page": "batch"},
    {"href": "/model", "label": "Model card", "page": "model"},
    {"href": "/docs", "label": "API docs", "page": "docs"},
]


def _safe_json(data: Any) -> str:
    """JSON for embedding inside a <script> block."""
    return json.dumps(data).replace("</", "<\\/")


def _render(
    request: Request, template: str, page: str, status_code: int = 200, **context: Any
) -> HTMLResponse:
    base = {
        "service_name": SERVICE_NAME,
        "service_version": SERVICE_VERSION,
        "repo_url": REPO_URL,
        "nav_items": NAV_ITEMS,
        "page": page,
    }
    base.update(context)
    return templates.TemplateResponse(request, template, base, status_code=status_code)


def _wants_json(request: Request) -> bool:
    path = request.url.path
    if path.startswith(API_PREFIXES):
        return True
    accept = request.headers.get("accept", "")
    return "application/json" in accept and "text/html" not in accept


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTML 404 for pages, JSON for API paths; everything else is FastAPI's default."""
    if exc.status_code == 404 and not _wants_json(request):
        if request.method == "HEAD":
            return Response(status_code=404, media_type="text/html")
        return _render(
            request, "404.html", "404", status_code=404, path=request.url.path
        )
    return await http_exception_handler(request, exc)


# ----------------------------------------------------------------------- pages
@app.api_route(
    "/", methods=["GET", "HEAD"], response_class=HTMLResponse, include_in_schema=False
)
def index(request: Request) -> HTMLResponse:
    """Landing page and interactive demo for /v2/predict and /v3/predict."""
    return _render(
        request,
        "index.html",
        "demo",
        v2_features=FEATURE_COLUMNS_V2,
        v3_presets_json=_safe_json(V3_PRESETS),
        v3_preset_count=len(V3_PRESETS.get("presets") or []),
    )


@app.api_route(
    "/batch",
    methods=["GET", "HEAD"],
    response_class=HTMLResponse,
    include_in_schema=False,
)
def batch_page(request: Request) -> HTMLResponse:
    """CSV batch scoring page for /v2/batch."""
    return _render(
        request,
        "batch.html",
        "batch",
        v2_features=FEATURE_COLUMNS_V2,
        sample_csv_url=SAMPLE_CSV_URL,
        max_rows=MAX_BATCH_ROWS,
        max_mb=MAX_BATCH_BYTES // (1024 * 1024),
    )


@app.api_route(
    "/model",
    methods=["GET", "HEAD"],
    response_class=HTMLResponse,
    include_in_schema=False,
)
def model_page(request: Request) -> HTMLResponse:
    """Model card. Every number is fetched client-side from GET /metrics."""
    return _render(request, "model.html", "model")


# ------------------------------------------------------------------ health/meta
@app.head("/health", include_in_schema=False)
def health_head() -> Response:
    """HEAD support for uptime monitors; the JSON body lives on GET."""
    return Response(status_code=200, media_type="application/json")


@app.get("/health", tags=["health"])
def health_check() -> dict:
    """Liveness/readiness probe. Lists every route with its serving artifact."""
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "model_version": fraud_model.model_version,
        "mock_mode": fraud_model.is_mock,
        "models": {
            "v3": {
                "model_version": fraud_model.v3_model_version,
                "loaded": not fraud_model.v3_is_mock,
            },
            "v2": {
                "model_version": fraud_model.v2_model_version,
                "loaded": not fraud_model.v2_is_mock,
            },
            "v1": {
                "model_version": fraud_model.v1_model_version,
                "loaded": not fraud_model.v1_is_mock,
            },
        },
        "endpoints": fraud_model.route_report(),
    }


@app.get("/metrics", tags=["model-card"])
def metrics() -> dict:
    """Evaluation metrics and dataset metadata read from the stored artifacts."""
    report = fraud_model.metrics_report()
    report["service"] = SERVICE_NAME
    report["version"] = SERVICE_VERSION
    return report


# ------------------------------------------------------------------- inference
@app.post(
    "/predict",
    response_model=FraudResponse,
    tags=["inference"],
    deprecated=True,
    summary="Score a transaction (v1, deprecated)",
    description=(
        "Deprecated v1 contract kept byte-identical for existing clients. "
        "It is served by the synthetic-data v1 model and returns no confidence. "
        "New integrations should call POST /v3/predict (real anonymised benchmark) "
        "or POST /v2/predict (behavioural features)."
    ),
)
def predict(request: FraudRequest) -> FraudResponse:
    """Score a single transaction with the v1 (synthetic) model."""
    try:
        prediction = fraud_model.predict(
            amount=request.amount,
            distance_from_home=request.distance_from_home,
            use_chip=request.use_chip,
        )
    except Exception as exc:  # surface any inference failure as a clean 500
        logger.exception("v1 inference failed for request %s", request.model_dump())
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {type(exc).__name__}"
        ) from exc

    return FraudResponse(
        prediction=prediction,
        is_fraud=bool(prediction),
        model_version=fraud_model.v1_model_version,
    )


@app.post(
    "/v2/predict",
    response_model=ScoredResponse,
    tags=["inference"],
    summary="Score a transaction (v2, behavioural features)",
    description=(
        "Scores a card transaction with the RandomForest trained on the OpenML "
        "card_transdata set (7 behavioural features). Returns the fraud probability "
        "as `confidence` and applies the optional `threshold` (default 0.5). Falls "
        "back to a deterministic rule with `confidence: null` if the artifact is "
        "unavailable."
    ),
)
def predict_v2(request: FraudRequestV2) -> ScoredResponse:
    """Score a single transaction with the v2 (card_transdata) model."""
    payload = request.model_dump()
    threshold = payload.pop("threshold")
    try:
        result = fraud_model.predict_v2(payload, threshold=threshold)
    except Exception as exc:  # surface any inference failure as a clean 500
        logger.exception("v2 inference failed for request %s", payload)
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {type(exc).__name__}"
        ) from exc

    return ScoredResponse(
        prediction=result.prediction,
        is_fraud=result.is_fraud,
        confidence=result.confidence,
        threshold=result.threshold,
        model_version=result.model_version,
    )


@app.post(
    "/v3/predict",
    response_model=ScoredResponse,
    tags=["inference"],
    summary="Score a transaction (v3, ULB anonymised benchmark)",
    description=(
        "Scores a transaction described by the ULB creditcard features (PCA "
        "components V1..V28 plus Amount) with a class-weighted RandomForest trained "
        "on 227,845 real transactions at 0.172% fraud prevalence. Returns the fraud "
        "probability as `confidence` and applies the optional `threshold`. Falls back "
        "to a deterministic rule with `confidence: null` if the artifact is "
        "unavailable."
    ),
)
def predict_v3(request: FraudRequestV3) -> ScoredResponse:  # type: ignore[valid-type]
    """Score a single transaction with the v3 (ULB creditcard) model."""
    payload = request.model_dump()
    threshold = payload.pop("threshold")
    try:
        result = fraud_model.predict_v3(payload, threshold=threshold)
    except Exception as exc:  # surface any inference failure as a clean 500
        logger.exception("v3 inference failed")
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {type(exc).__name__}"
        ) from exc

    return ScoredResponse(
        prediction=result.prediction,
        is_fraud=result.is_fraud,
        confidence=result.confidence,
        threshold=result.threshold,
        model_version=result.model_version,
    )


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=400, detail=detail)


def _parse_batch_csv(raw: bytes) -> pd.DataFrame:
    """Parse and validate an uploaded CSV into a clean v2 feature frame."""
    if not raw.strip():
        raise _bad_request("The uploaded file is empty.")
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError:
        raise _bad_request("The file is not UTF-8 text; expected a plain-text CSV.")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise _bad_request(f"Could not parse CSV: {str(exc).splitlines()[0][:200]}")

    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [c for c in FEATURE_COLUMNS_V2 if c not in frame.columns]
    if missing:
        raise _bad_request(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected header: {','.join(FEATURE_COLUMNS_V2)}"
        )
    if len(frame) == 0:
        raise _bad_request("The CSV has a header but no data rows.")
    if len(frame) > MAX_BATCH_ROWS:
        raise _bad_request(
            f"Too many rows: {len(frame)} (maximum {MAX_BATCH_ROWS} per upload)."
        )

    data = frame[FEATURE_COLUMNS_V2].copy()
    for col in FEATURE_COLUMNS_V2:
        numeric = pd.to_numeric(data[col], errors="coerce")
        bad = numeric.isna()
        if bad.any():
            lines = [int(i) + 2 for i in bad[bad].index[:5]]  # header = line 1
            raise _bad_request(
                f"Column '{col}' has non-numeric or empty values "
                f"(e.g. line(s) {lines})."
            )
        data[col] = numeric.astype(float)

    for col in BINARY_COLUMNS_V2:
        bad = ~data[col].isin([0.0, 1.0])
        if bad.any():
            lines = [int(i) + 2 for i in bad[bad].index[:5]]
            raise _bad_request(
                f"Column '{col}' must contain only 0 or 1 (e.g. line(s) {lines})."
            )
    for col in NON_NEGATIVE_COLUMNS_V2:
        bad = data[col] < 0
        if bad.any():
            lines = [int(i) + 2 for i in bad[bad].index[:5]]
            raise _bad_request(
                f"Column '{col}' must be greater than or equal to 0 "
                f"(e.g. line(s) {lines})."
            )
    return data


@app.post(
    "/v2/batch",
    response_model=BatchResponse,
    tags=["inference"],
    summary="Score a CSV of transactions (v2)",
    description=(
        "Upload a UTF-8 CSV (multipart field `file`, .csv extension, at most 2 MB and "
        f"{MAX_BATCH_ROWS} data rows) whose header contains the seven v2 feature "
        "columns in any order. Extra columns are ignored. Every row is scored with "
        "the v2 model and returned with its confidence; malformed input yields a 400 "
        "naming the offending column. A real sample file is served at "
        "/static/data/sample_batch.csv."
    ),
    responses={
        400: {
            "description": (
                "Malformed CSV: missing/invalid column, bad values, too many rows."
            )
        },
        413: {"description": "File larger than 2 MB."},
    },
)
async def batch_v2(
    file: UploadFile = File(..., description="CSV file with the v2 feature columns."),
    threshold: float = Query(
        DEFAULT_THRESHOLD, gt=0, lt=1, description=THRESHOLD_DESCRIPTION
    ),
) -> BatchResponse:
    """Score every row of an uploaded CSV with the v2 model."""
    started = time.perf_counter()
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise _bad_request(
            f"Only .csv files are accepted (received '{filename or 'unnamed file'}')."
        )

    raw = await file.read()
    if len(raw) > MAX_BATCH_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(raw) / (1024 * 1024):.1f} MB; the limit is 2 MB.",
        )

    data = _parse_batch_csv(raw)

    try:
        predictions, confidences = fraud_model.predict_v2_batch(
            data, threshold=threshold
        )
    except Exception as exc:  # surface any inference failure as a clean 500
        logger.exception("v2 batch inference failed (%d rows)", len(data))
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {type(exc).__name__}"
        ) from exc

    records = data.to_dict(orient="records")
    rows = [
        BatchRow(
            row=i + 1,
            inputs={k: float(v) for k, v in record.items()},
            prediction=int(pred),
            is_fraud=bool(pred),
            confidence=None if confidences is None else float(confidences[i]),
        )
        for i, (record, pred) in enumerate(zip(records, predictions))
    ]
    flagged = int(sum(r.prediction for r in rows))
    summary = BatchSummary(
        total=len(rows),
        flagged=flagged,
        flag_rate=round(flagged / len(rows), 4),
        processing_ms=round((time.perf_counter() - started) * 1000, 1),
        threshold=threshold,
        model_version=fraud_model.v2_model_version,
    )
    return BatchResponse(
        model_version=fraud_model.v2_model_version,
        threshold=threshold,
        summary=summary,
        rows=rows,
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
