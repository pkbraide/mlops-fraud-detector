"""Model loading and inference utilities for the fraud detection engine.

Two generations of artifact are supported side by side:

* **v2** -- ``models/fraud_model_v2.joblib``: a dict ``{"model": ..., "features":
  [...]}`` trained on the real OpenML ``card_transdata`` set (7 features). The
  feature list is stored *inside* the artifact so the inference column order can
  never silently drift from the training column order.
* **v1** -- ``models/fraud_model.joblib``: a bare estimator trained on synthetic
  data (3 features). Kept so ``POST /predict`` stays byte-identical.

When neither artifact can be loaded the class degrades to deterministic
rule-based fallbacks so the API surface keeps working in CI and on a fresh
clone. ``model_version`` reports the most capable path that loaded and can
only ever hold one of three literal values:

* ``"joblib_model_v2"`` -- the v2 artifact loaded (primary).
* ``"joblib_model"``    -- only the v1 artifact loaded.
* ``"mock_rule"``       -- no artifact loaded; rule-based fallback.

Each endpoint additionally reports the version that *served that request*
(``v1_model_version`` / ``v2_model_version``) because a v2 model cannot score
a v1 payload and vice versa.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_VERSION_V2 = "joblib_model_v2"
MODEL_VERSION_V1 = "joblib_model"
MODEL_VERSION_MOCK = "mock_rule"

# v1 column order (synthetic model). MUST match the v1 notebook.
FEATURE_COLUMNS = ["amount", "distance_from_home", "use_chip"]

# v2 column order (card_transdata). The artifact carries its own copy and we
# assert the two agree at load time.
FEATURE_COLUMNS_V2 = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]

# Thresholds for the deterministic v1 fallback rule.
MOCK_AMOUNT_THRESHOLD = 1000.0
MOCK_DISTANCE_THRESHOLD = 50.0

# Thresholds for the deterministic v2 fallback rule. These mirror the strongest
# univariate fraud signals in card_transdata: very large purchases relative to
# the cardholder's median, or far-from-home purchases, made online without a PIN.
MOCK_V2_RATIO_THRESHOLD = 4.0
MOCK_V2_DISTANCE_THRESHOLD = 100.0

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "fraud_model.joblib")
DEFAULT_MODEL_V2_PATH = os.path.join(_REPO_ROOT, "models", "fraud_model_v2.joblib")


class FraudDetectionModel:
    """Load the v2 and v1 fraud classifiers with rule-based safety nets."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        model_v2_path: str = DEFAULT_MODEL_V2_PATH,
    ) -> None:
        self.model_path = model_path
        self.model_v2_path = model_v2_path

        self.v1_model: Optional[Any] = None
        self.v2_model: Optional[Any] = None
        self.v2_features: List[str] = list(FEATURE_COLUMNS_V2)
        self.v2_metadata: Dict[str, Any] = {}

        self._load_v2()
        self._load_v1()

        logger.info(
            "FraudDetectionModel ready: model_version=%s (v2 loaded=%s, v1 loaded=%s)",
            self.model_version,
            self.v2_model is not None,
            self.v1_model is not None,
        )

    # ------------------------------------------------------------------ loading
    def _load_v2(self) -> None:
        try:
            payload = joblib.load(self.model_v2_path)
        except FileNotFoundError:
            logger.warning(
                "v2 artifact not found at %s; v2 endpoint will use the fallback rule.",
                self.model_v2_path,
            )
            return
        except Exception as exc:  # any load failure must degrade safely
            logger.warning(
                "Failed to load v2 artifact from %s (%s: %s); using fallback rule.",
                self.model_v2_path,
                type(exc).__name__,
                exc,
            )
            return

        if not isinstance(payload, dict) or "model" not in payload:
            logger.warning(
                "v2 artifact at %s is not a {'model', 'features'} dict; ignoring it.",
                self.model_v2_path,
            )
            return

        features = list(payload.get("features") or [])
        if features != FEATURE_COLUMNS_V2:
            logger.warning(
                "v2 artifact feature order %s does not match expected %s; ignoring it.",
                features,
                FEATURE_COLUMNS_V2,
            )
            return

        self.v2_model = payload["model"]
        self.v2_features = features
        self.v2_metadata = {
            k: v for k, v in payload.items() if k not in ("model", "features")
        }
        logger.info(
            "Loaded v2 fraud model from %s (%s, %d features)",
            self.model_v2_path,
            type(self.v2_model).__name__,
            len(self.v2_features),
        )

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
                "v1 artifact not found at %s; v1 endpoint will use the fallback rule.",
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
        """The primary estimator: v2 if loaded, else v1, else ``None``."""
        if self.v2_model is not None:
            return self.v2_model
        return self.v1_model

    @property
    def is_mock(self) -> bool:
        """True only when *no* trained artifact could be loaded."""
        return self.v2_model is None and self.v1_model is None

    @property
    def model_version(self) -> str:
        """Most capable path available: v2 > v1 > mock."""
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
        """Version string that serves ``POST /predict``."""
        return MODEL_VERSION_MOCK if self.v1_model is None else MODEL_VERSION_V1

    @property
    def v2_is_mock(self) -> bool:
        return self.v2_model is None

    @property
    def v2_model_version(self) -> str:
        """Version string that serves ``POST /v2/predict``."""
        return MODEL_VERSION_MOCK if self.v2_model is None else MODEL_VERSION_V2

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
        """v1 scoring: ``1`` if predicted fraudulent, else ``0``."""
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
    def _mock_predict_v2(
        distance_from_home: float,
        distance_from_last_transaction: float,
        ratio_to_median_purchase_price: float,
        repeat_retailer: int,
        used_chip: int,
        used_pin_number: int,
        online_order: int,
    ) -> int:
        """Deterministic v2 heuristic used when no v2 artifact is available."""
        unusual_size_or_place = (
            ratio_to_median_purchase_price > MOCK_V2_RATIO_THRESHOLD
            or distance_from_home > MOCK_V2_DISTANCE_THRESHOLD
        )
        is_fraud = unusual_size_or_place and online_order == 1 and used_pin_number == 0
        return 1 if is_fraud else 0

    def _vectorise_v2(self, payload: Dict[str, Any]) -> np.ndarray:
        """Build the feature row in the exact order stored in the artifact."""
        missing = [name for name in self.v2_features if name not in payload]
        if missing:
            raise ValueError(f"Missing v2 features: {missing}")
        row: Sequence[float] = [float(payload[name]) for name in self.v2_features]
        return np.array([row], dtype=float)

    def predict_v2(self, payload: Dict[str, Any]) -> int:
        """v2 scoring from a ``{feature_name: value}`` mapping."""
        if self.v2_model is None:
            return self._mock_predict_v2(
                distance_from_home=float(payload["distance_from_home"]),
                distance_from_last_transaction=float(
                    payload["distance_from_last_transaction"]
                ),
                ratio_to_median_purchase_price=float(
                    payload["ratio_to_median_purchase_price"]
                ),
                repeat_retailer=int(payload["repeat_retailer"]),
                used_chip=int(payload["used_chip"]),
                used_pin_number=int(payload["used_pin_number"]),
                online_order=int(payload["online_order"]),
            )

        features = self._vectorise_v2(payload)
        prediction = self.v2_model.predict(features)
        return int(prediction[0])


# Module-level singleton: artifacts are loaded once at import time, not per
# request. FastAPI handlers import this instance directly.
fraud_model = FraudDetectionModel()
