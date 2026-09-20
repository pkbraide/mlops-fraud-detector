"""Model loading and inference utilities for the fraud detection engine.

Three generations of artifact are supported side by side, each serving its own
route and each reporting the artifact that actually produced a prediction:

* **v3** -- ``models/fraud_model_v3.joblib``: dict ``{"model", "features", "metrics",
  "dataset", ...}`` trained on the ULB *creditcard* benchmark (real anonymised
  transactions, 0.172 % fraud, PCA features V1-V28 + Amount). Serves ``/v3/predict``.
* **v2** -- ``models/fraud_model_v2.joblib``: same dict contract, trained on the
  synthetic OpenML ``card_transdata`` set (7 behavioural features). Serves
  ``/v2/predict`` and ``/v2/batch``.
* **v1** -- ``models/fraud_model.joblib``: bare estimator on ``make_classification``
  data (3 features). Serves the deprecated ``/predict`` byte-for-byte.

Dict artifacts carry their own feature list; the loader refuses an artifact whose
order differs from the API's, so columns can never be silently shuffled.

When an artifact is missing or unloadable its route degrades to a deterministic
rule. Rule paths never invent a probability: ``confidence`` is ``None``. The
``model_version`` strings are the only permitted values:

``"joblib_model_v3"`` / ``"joblib_model_v2"`` / ``"joblib_model"`` / ``"mock_rule"``.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_VERSION_V3 = "joblib_model_v3"
MODEL_VERSION_V2 = "joblib_model_v2"
MODEL_VERSION_V1 = "joblib_model"
MODEL_VERSION_MOCK = "mock_rule"

DEFAULT_THRESHOLD = 0.5

# v1 column order (synthetic model). MUST match the v1 notebook.
FEATURE_COLUMNS = ["amount", "distance_from_home", "use_chip"]

# v2 column order (card_transdata). The artifact carries its own copy.
FEATURE_COLUMNS_V2 = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]
BINARY_COLUMNS_V2 = ["repeat_retailer", "used_chip", "used_pin_number", "online_order"]
NON_NEGATIVE_COLUMNS_V2 = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
]

# v3 column order (ULB creditcard on OpenML: no Time column). The artifact carries
# its own copy and we assert the two agree at load time.
FEATURE_COLUMNS_V3 = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Deterministic v1 fallback thresholds.
MOCK_AMOUNT_THRESHOLD = 1000.0
MOCK_DISTANCE_THRESHOLD = 50.0

# Deterministic v2 fallback: the strongest univariate fraud signals in card_transdata.
MOCK_V2_RATIO_THRESHOLD = 4.0
MOCK_V2_DISTANCE_THRESHOLD = 100.0

# Deterministic v3 fallback. V14 and V10 are the two most important components in
# the trained forest; on the training split their fraud medians are about -6.7 and
# -4.6 while fewer than 0.1 % of legitimate rows fall below -4.5 / -3.4.
MOCK_V3_V14_THRESHOLD = -5.0
MOCK_V3_V10_THRESHOLD = -3.0

MOCK_RULE_V1 = "amount > 1000 and distance_from_home > 50 and use_chip == 0"
MOCK_RULE_V2 = (
    "(ratio_to_median_purchase_price > 4 or distance_from_home > 100) "
    "and online_order == 1 and used_pin_number == 0"
)
MOCK_RULE_V3 = "V14 < -5.0 and V10 < -3.0"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "fraud_model.joblib")
DEFAULT_MODEL_V2_PATH = os.path.join(_REPO_ROOT, "models", "fraud_model_v2.joblib")
DEFAULT_MODEL_V3_PATH = os.path.join(_REPO_ROOT, "models", "fraud_model_v3.joblib")


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of scoring one transaction."""

    prediction: int
    confidence: Optional[float]
    threshold: float
    model_version: str

    @property
    def is_fraud(self) -> bool:
        return bool(self.prediction)


def _positive_class_index(model: Any) -> int:
    classes = list(getattr(model, "classes_", [0, 1]))
    return classes.index(1) if 1 in classes else len(classes) - 1


def _score(
    model: Any, X: np.ndarray, threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(predictions, positive-class probabilities)`` for a feature matrix."""
    proba = model.predict_proba(X)[:, _positive_class_index(model)]
    predictions = (proba >= threshold).astype(int)
    return predictions, proba


def _relpath(path: str) -> str:
    try:
        return os.path.relpath(path, _REPO_ROOT).replace(os.sep, "/")
    except ValueError:  # different drive on Windows
        return path


class FraudDetectionModel:
    """Load the v3, v2 and v1 fraud classifiers with rule-based safety nets."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        model_v2_path: str = DEFAULT_MODEL_V2_PATH,
        model_v3_path: str = DEFAULT_MODEL_V3_PATH,
    ) -> None:
        self.model_path = model_path
        self.model_v2_path = model_v2_path
        self.model_v3_path = model_v3_path

        self.v1_model: Optional[Any] = None

        self.v2_model: Optional[Any] = None
        self.v2_features: List[str] = list(FEATURE_COLUMNS_V2)
        self.v2_metadata: Dict[str, Any] = {}

        self.v3_model: Optional[Any] = None
        self.v3_features: List[str] = list(FEATURE_COLUMNS_V3)
        self.v3_metadata: Dict[str, Any] = {}

        loaded = self._load_dict_artifact(model_v3_path, FEATURE_COLUMNS_V3, "v3")
        if loaded is not None:
            self.v3_model, self.v3_features, self.v3_metadata = loaded

        loaded = self._load_dict_artifact(model_v2_path, FEATURE_COLUMNS_V2, "v2")
        if loaded is not None:
            self.v2_model, self.v2_features, self.v2_metadata = loaded

        self._load_v1()

        logger.info(
            "FraudDetectionModel ready: model_version=%s (v3=%s, v2=%s, v1=%s)",
            self.model_version,
            self.v3_model is not None,
            self.v2_model is not None,
            self.v1_model is not None,
        )

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _load_dict_artifact(
        path: str, expected_features: Sequence[str], label: str
    ) -> Optional[Tuple[Any, List[str], Dict[str, Any]]]:
        """Load a ``{"model", "features", ...}`` artifact or return ``None``."""
        try:
            payload = joblib.load(path)
        except FileNotFoundError:
            logger.warning(
                "%s artifact not found at %s; its route will use the fallback rule.",
                label,
                path,
            )
            return None
        except Exception as exc:  # any load failure must degrade safely
            logger.warning(
                "Failed to load %s artifact from %s (%s: %s); using fallback rule.",
                label,
                path,
                type(exc).__name__,
                exc,
            )
            return None

        if not isinstance(payload, dict) or "model" not in payload:
            logger.warning(
                "%s artifact at %s is not a {'model', 'features'} dict; ignoring it.",
                label,
                path,
            )
            return None

        features = list(payload.get("features") or [])
        if features != list(expected_features):
            logger.warning(
                "%s artifact feature order %s does not match expected %s; ignoring it.",
                label,
                features,
                list(expected_features),
            )
            return None

        model = payload["model"]
        if not hasattr(model, "predict_proba"):
            logger.warning(
                "%s artifact model %s has no predict_proba; ignoring it.",
                label,
                type(model).__name__,
            )
            return None

        metadata = {k: v for k, v in payload.items() if k not in ("model", "features")}
        logger.info(
            "Loaded %s fraud model from %s (%s, %d features)",
            label,
            path,
            type(model).__name__,
            len(features),
        )
        return model, features, metadata

    def _load_v1(self) -> None:
        try:
            self.v1_model = joblib.load(self.model_path)
            logger.info(
                "Loaded v1 fraud model from %s (%s)",
                self.model_path,
                type(self.v1_model).__name__,
            )
        except FileNotFoundError:
            logger.warning(
                "v1 artifact not found at %s; /predict will use the fallback rule.",
                self.model_path,
            )
            self.v1_model = None
        except Exception as exc:  # any load failure must degrade safely
            logger.warning(
                "Failed to load v1 artifact from %s (%s: %s); using fallback rule.",
                self.model_path,
                type(exc).__name__,
                exc,
            )
            self.v1_model = None

    # --------------------------------------------------------------- properties
    @property
    def model(self) -> Optional[Any]:
        """The most capable estimator loaded: v3, else v2, else v1, else ``None``."""
        for candidate in (self.v3_model, self.v2_model, self.v1_model):
            if candidate is not None:
                return candidate
        return None

    @property
    def is_mock(self) -> bool:
        """True only when *no* trained artifact could be loaded."""
        return self.model is None

    @property
    def model_version(self) -> str:
        """Most capable path available: v3 > v2 > v1 > mock."""
        if self.v3_model is not None:
            return MODEL_VERSION_V3
        if self.v2_model is not None:
            return MODEL_VERSION_V2
        if self.v1_model is not None:
            return MODEL_VERSION_V1
        return MODEL_VERSION_MOCK

    @property
    def v1_is_mock(self) -> bool:
        return self.v1_model is None

    @property
    def v1_model_version(self) -> str:
        return MODEL_VERSION_MOCK if self.v1_model is None else MODEL_VERSION_V1

    @property
    def v2_is_mock(self) -> bool:
        return self.v2_model is None

    @property
    def v2_model_version(self) -> str:
        return MODEL_VERSION_MOCK if self.v2_model is None else MODEL_VERSION_V2

    @property
    def v3_is_mock(self) -> bool:
        return self.v3_model is None

    @property
    def v3_model_version(self) -> str:
        return MODEL_VERSION_MOCK if self.v3_model is None else MODEL_VERSION_V3

    # ------------------------------------------------------------------ v1 API
    @staticmethod
    def _mock_predict(amount: float, distance_from_home: float, use_chip: int) -> int:
        """Deterministic v1 heuristic used when no v1 artifact is available."""
        is_fraud = (
            amount > MOCK_AMOUNT_THRESHOLD
            and distance_from_home > MOCK_DISTANCE_THRESHOLD
            and use_chip == 0
        )
        return 1 if is_fraud else 0

    def predict(self, amount: float, distance_from_home: float, use_chip: int) -> int:
        """v1 scoring: ``1`` if predicted fraudulent, else ``0``. Unchanged since 1.0"""
        if self.v1_model is None:
            return self._mock_predict(amount, distance_from_home, use_chip)

        features = np.array(
            [[float(amount), float(distance_from_home), int(use_chip)]],
            dtype=float,
        )
        prediction = self.v1_model.predict(features)
        return int(prediction[0])

    # ------------------------------------------------------------------ v2 API
    @staticmethod
    def _mock_predict_v2_array(
        ratio: np.ndarray,
        distance_from_home: np.ndarray,
        online_order: np.ndarray,
        used_pin_number: np.ndarray,
    ) -> np.ndarray:
        unusual = (ratio > MOCK_V2_RATIO_THRESHOLD) | (
            distance_from_home > MOCK_V2_DISTANCE_THRESHOLD
        )
        return (unusual & (online_order == 1) & (used_pin_number == 0)).astype(int)

    @classmethod
    def _mock_predict_v2(cls, payload: Dict[str, Any]) -> int:
        """Deterministic v2 heuristic used when no v2 artifact is available."""
        result = cls._mock_predict_v2_array(
            np.array([float(payload["ratio_to_median_purchase_price"])]),
            np.array([float(payload["distance_from_home"])]),
            np.array([int(payload["online_order"])]),
            np.array([int(payload["used_pin_number"])]),
        )
        return int(result[0])

    def _vectorise(
        self, payload: Dict[str, Any], features: Sequence[str], label: str
    ) -> np.ndarray:
        """Build a single feature row in the exact order stored in the artifact."""
        missing = [name for name in features if name not in payload]
        if missing:
            raise ValueError(f"Missing {label} features: {missing}")
        row = [float(payload[name]) for name in features]
        return np.array([row], dtype=float)

    def predict_v2(
        self, payload: Dict[str, Any], threshold: float = DEFAULT_THRESHOLD
    ) -> ScoreResult:
        """v2 scoring from a ``{feature_name: value}`` mapping."""
        if self.v2_model is None:
            return ScoreResult(
                prediction=self._mock_predict_v2(payload),
                confidence=None,
                threshold=threshold,
                model_version=MODEL_VERSION_MOCK,
            )

        X = self._vectorise(payload, self.v2_features, "v2")
        predictions, proba = _score(self.v2_model, X, threshold)
        return ScoreResult(
            prediction=int(predictions[0]),
            confidence=round(float(proba[0]), 4),
            threshold=threshold,
            model_version=MODEL_VERSION_V2,
        )

    def predict_v2_batch(
        self, frame: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Score many v2 rows at once. Returns ``(predictions, confidences|None)``."""
        if self.v2_model is None:
            predictions = self._mock_predict_v2_array(
                frame["ratio_to_median_purchase_price"].to_numpy(dtype=float),
                frame["distance_from_home"].to_numpy(dtype=float),
                frame["online_order"].to_numpy(dtype=int),
                frame["used_pin_number"].to_numpy(dtype=int),
            )
            return predictions, None

        X = frame[self.v2_features].to_numpy(dtype=float)
        predictions, proba = _score(self.v2_model, X, threshold)
        return predictions, np.round(proba, 4)

    # ------------------------------------------------------------------ v3 API
    @staticmethod
    def _mock_predict_v3(payload: Dict[str, Any]) -> int:
        """Deterministic v3 heuristic used when no v3 artifact is available."""
        is_fraud = (
            float(payload["V14"]) < MOCK_V3_V14_THRESHOLD
            and float(payload["V10"]) < MOCK_V3_V10_THRESHOLD
        )
        return 1 if is_fraud else 0

    def predict_v3(
        self, payload: Dict[str, Any], threshold: float = DEFAULT_THRESHOLD
    ) -> ScoreResult:
        """v3 scoring from a ``{feature_name: value}`` mapping (V1..V28, Amount)."""
        if self.v3_model is None:
            return ScoreResult(
                prediction=self._mock_predict_v3(payload),
                confidence=None,
                threshold=threshold,
                model_version=MODEL_VERSION_MOCK,
            )

        X = self._vectorise(payload, self.v3_features, "v3")
        predictions, proba = _score(self.v3_model, X, threshold)
        return ScoreResult(
            prediction=int(predictions[0]),
            confidence=round(float(proba[0]), 4),
            threshold=threshold,
            model_version=MODEL_VERSION_V3,
        )

    # --------------------------------------------------------------- reporting
    def route_report(self) -> List[Dict[str, Any]]:
        """Every route with the artifact that serves it and whether it loaded."""
        return [
            {
                "method": "POST",
                "path": "/v3/predict",
                "status": "current",
                "model_version": self.v3_model_version,
                "loaded": self.v3_model is not None,
                "artifact": _relpath(self.model_v3_path),
                "fallback_rule": MOCK_RULE_V3,
            },
            {
                "method": "POST",
                "path": "/v2/predict",
                "status": "current",
                "model_version": self.v2_model_version,
                "loaded": self.v2_model is not None,
                "artifact": _relpath(self.model_v2_path),
                "fallback_rule": MOCK_RULE_V2,
            },
            {
                "method": "POST",
                "path": "/v2/batch",
                "status": "current",
                "model_version": self.v2_model_version,
                "loaded": self.v2_model is not None,
                "artifact": _relpath(self.model_v2_path),
                "fallback_rule": MOCK_RULE_V2,
            },
            {
                "method": "POST",
                "path": "/predict",
                "status": "deprecated",
                "model_version": self.v1_model_version,
                "loaded": self.v1_model is not None,
                "artifact": _relpath(self.model_path),
                "fallback_rule": MOCK_RULE_V1,
            },
            {"method": "GET", "path": "/", "status": "current"},
            {"method": "GET", "path": "/batch", "status": "current"},
            {"method": "GET", "path": "/model", "status": "current"},
            {"method": "GET", "path": "/health", "status": "current"},
            {"method": "GET", "path": "/metrics", "status": "current"},
            {"method": "GET", "path": "/docs", "status": "current"},
        ]

    def _artifact_block(
        self,
        route: str,
        model: Optional[Any],
        model_version: str,
        artifact_path: str,
        features: Sequence[str],
        metadata: Dict[str, Any],
        fallback_rule: str,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        block: Dict[str, Any] = {
            "route": route,
            "model_version": model_version,
            "loaded": model is not None,
            "artifact": _relpath(artifact_path),
            "estimator": type(model).__name__ if model is not None else None,
            "features": list(features),
            "n_features": len(features),
            "fallback_rule": fallback_rule,
            "dataset": metadata.get("dataset"),
            "metrics": metadata.get("metrics"),
            "trained_at": metadata.get("trained_at"),
            "sklearn_version": metadata.get("sklearn_version"),
            "n_rows_used": metadata.get("n_rows_used"),
            "subsampled": metadata.get("subsampled"),
            "fit_seconds": metadata.get("fit_seconds"),
            "model_params": metadata.get("model_params"),
        }
        if note:
            block["note"] = note
        return block

    def metrics_report(self) -> Dict[str, Any]:
        """Everything the model card needs, read from the artifacts' stored metadata."""
        return {
            "source": "artifact metadata (joblib dict payloads); nothing is hardcoded",
            "models": {
                "v3": self._artifact_block(
                    "/v3/predict",
                    self.v3_model,
                    self.v3_model_version,
                    self.model_v3_path,
                    self.v3_features,
                    self.v3_metadata,
                    MOCK_RULE_V3,
                ),
                "v2": self._artifact_block(
                    "/v2/predict",
                    self.v2_model,
                    self.v2_model_version,
                    self.model_v2_path,
                    self.v2_features,
                    self.v2_metadata,
                    MOCK_RULE_V2,
                ),
                "v1": self._artifact_block(
                    "/predict",
                    self.v1_model,
                    self.v1_model_version,
                    self.model_path,
                    FEATURE_COLUMNS,
                    {},
                    MOCK_RULE_V1,
                    note=(
                        "v1 is a bare estimator trained on make_classification "
                        "synthetic data; the artifact stores no metrics or dataset "
                        "metadata and the route is deprecated."
                    ),
                ),
            },
        }


# Module-level singleton: artifacts are loaded once at import time, not per
# request. FastAPI handlers import this instance directly.
fraud_model = FraudDetectionModel()
