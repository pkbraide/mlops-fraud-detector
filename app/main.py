"""FastAPI entrypoint for the Enterprise Real-Time Fraud Detection Engine."""

import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.model_utils import FEATURE_COLUMNS_V2, fraud_model

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

SERVICE_NAME = "Enterprise Real-Time Fraud Detection Engine"
SERVICE_VERSION = "2.0.0"
SERVICE_DESCRIPTION = (
    "Real-time card-fraud scoring API with a browser UI at /. "
    "POST /v2/predict scores a transaction with a RandomForest trained on the "
    "OpenML card_transdata set (1M real labelled transactions, 7 features). "
    "POST /predict is the deprecated v1 contract (3 synthetic features) kept for "
    "backwards compatibility. When an artifact cannot be loaded the service degrades "
    "to a deterministic rule-based fallback; every response reports the "
    "`model_version` that produced it so consumers can tell which path served them."
)
REPO_URL = "https://github.com/pkbraide/mlops-fraud-detector"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")

app = FastAPI(
    title=SERVICE_NAME,
    description=SERVICE_DESCRIPTION,
    version=SERVICE_VERSION,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# --------------------------------------------------------------------- schemas
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
            }
        }
    }


class FraudResponse(BaseModel):
    """Scoring result (shared by v1 and v2)."""

    prediction: int = Field(..., description="1 = fraud, 0 = legitimate.")
    is_fraud: bool = Field(..., description="Convenience boolean of `prediction`.")
    model_version: str = Field(
        ...,
        description="One of 'joblib_model_v2', 'joblib_model' or 'mock_rule'.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": 0,
                "is_fraud": False,
                "model_version": "joblib_model_v2",
            }
        }
    }


# ---------------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request) -> HTMLResponse:
    """Browser UI for scoring a transaction against /v2/predict."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "service_name": SERVICE_NAME,
            "service_version": SERVICE_VERSION,
            "repo_url": REPO_URL,
            "features": FEATURE_COLUMNS_V2,
        },
    )


@app.get("/health", tags=["health"])
def health_check() -> dict:
    """Liveness/readiness probe. Reports which model path is active."""
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "model_version": fraud_model.model_version,
        "mock_mode": fraud_model.is_mock,
        "endpoints": [
            {
                "method": "POST",
                "path": "/v2/predict",
                "status": "current",
                "model_version": fraud_model.v2_model_version,
            },
            {
                "method": "POST",
                "path": "/predict",
                "status": "deprecated",
                "model_version": fraud_model.v1_model_version,
            },
            {"method": "GET", "path": "/health", "status": "current"},
            {"method": "GET", "path": "/", "status": "current"},
            {"method": "GET", "path": "/docs", "status": "current"},
        ],
    }


@app.post(
    "/predict",
    response_model=FraudResponse,
    tags=["inference"],
    deprecated=True,
    summary="Score a transaction (v1, deprecated)",
    description=(
        "Deprecated v1 contract kept byte-identical for existing clients. "
        "It is served by the synthetic-data v1 model. New integrations should call "
        "POST /v2/predict, which uses the model trained on real transaction data."
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
    response_model=FraudResponse,
    tags=["inference"],
    summary="Score a transaction (v2, current)",
    description=(
        "Scores a card transaction with the RandomForest trained on the OpenML "
        "card_transdata set. Feature order is read from the artifact itself, so it "
        "cannot drift from training. Falls back to a deterministic rule "
        "(`model_version = 'mock_rule'`) if the artifact is unavailable."
    ),
)
def predict_v2(request: FraudRequestV2) -> FraudResponse:
    """Score a single transaction with the v2 (real-data) model."""
    try:
        prediction = fraud_model.predict_v2(request.model_dump())
    except Exception as exc:  # surface any inference failure as a clean 500
        logger.exception("v2 inference failed for request %s", request.model_dump())
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {type(exc).__name__}"
        ) from exc

    return FraudResponse(
        prediction=prediction,
        is_fraud=bool(prediction),
        model_version=fraud_model.v2_model_version,
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
