"""FastAPI entrypoint for the Enterprise Real-Time Fraud Detection Engine."""

import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model_utils import fraud_model

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

SERVICE_NAME = "Enterprise Real-Time Fraud Detection Engine"

app = FastAPI(
    title=SERVICE_NAME,
    description=(
        "Low-latency REST API that scores card transactions for fraud risk in "
        "real time. A scikit-learn RandomForest classifier trained on "
        "transaction amount, distance from the cardholder's home and chip usage "
        "is served from a joblib artifact. When no artifact is available the "
        "service degrades to a deterministic rule-based fallback so the API "
        "contract stays stable across CI, local development and production. "
        "Every response reports the `model_version` that produced it."
    ),
    version="1.0.0",
)


class FraudRequest(BaseModel):
    """Inbound transaction features."""

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
    """Scoring result."""

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


@app.get("/", tags=["health"])
def health_check() -> dict:
    """Liveness/readiness probe. Also reports which model path is active."""
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "model_version": fraud_model.model_version,
        "mock_mode": fraud_model.is_mock,
    }


@app.post("/predict", response_model=FraudResponse, tags=["inference"])
def predict(request: FraudRequest) -> FraudResponse:
    """Score a single transaction."""
    try:
        prediction = fraud_model.predict(
            amount=request.amount,
            distance_from_home=request.distance_from_home,
            use_chip=request.use_chip,
        )
    except Exception as exc:  # surface any inference failure as a clean 500
        logger.exception("Inference failed for request %s", request.model_dump())
        raise HTTPException(
            status_code=500, detail=f"Inference failed: {type(exc).__name__}"
        ) from exc

    return FraudResponse(
        prediction=prediction,
        is_fraud=bool(prediction),
        model_version=fraud_model.model_version,
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,
    )
