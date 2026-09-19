"""Model loading and inference utilities for the fraud detection engine.

The :class:`FraudDetectionModel` wraps a scikit-learn artifact serialised with
``joblib``. When the artifact cannot be loaded (fresh clone, CI runner, corrupt
file, incompatible library version) the class transparently degrades to a
deterministic rule-based mock so the API surface keeps working and every
downstream consumer can still be exercised end to end.

The ``model_version`` attribute exposes which path is serving predictions and
is surfaced on every API response. It can only ever hold one of two literal
values:

* ``"joblib_model"`` -- the trained artifact was loaded successfully.
* ``"mock_rule"``    -- the deterministic fallback rule is in use.
"""

import logging
import os
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Column order MUST match the DataFrame used in notebooks/train.ipynb.
FEATURE_COLUMNS = ["amount", "distance_from_home", "use_chip"]

MODEL_VERSION_JOBLIB = "joblib_model"
MODEL_VERSION_MOCK = "mock_rule"

# Thresholds for the deterministic fallback rule.
MOCK_AMOUNT_THRESHOLD = 1000.0
MOCK_DISTANCE_THRESHOLD = 50.0

# Resolve the default artifact path relative to the repository root so the
# model loads regardless of the process' current working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "fraud_model.joblib")


class FraudDetectionModel:
    """Load a serialised fraud classifier with a rule-based safety net."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self.model: Optional[object] = None
        self.is_mock: bool = True
        self.model_version: str = MODEL_VERSION_MOCK
        self._load()

    def _load(self) -> None:
        """Attempt to load the joblib artifact, falling back to mock mode."""
        try:
            self.model = joblib.load(self.model_path)
            self.is_mock = False
            self.model_version = MODEL_VERSION_JOBLIB
            logger.info(
                "Loaded fraud model from %s (%s)",
                self.model_path,
                type(self.model).__name__,
            )
        except FileNotFoundError:
            logger.warning(
                "Model artifact not found at %s; using rule-based mock model.",
                self.model_path,
            )
            self._activate_mock()
        except Exception as exc:  # any load failure must degrade safely
            logger.warning(
                "Failed to load model from %s (%s: %s); using rule-based mock model.",
                self.model_path,
                type(exc).__name__,
                exc,
            )
            self._activate_mock()

    def _activate_mock(self) -> None:
        self.model = None
        self.is_mock = True
        self.model_version = MODEL_VERSION_MOCK

    @staticmethod
    def _mock_predict(amount: float, distance_from_home: float, use_chip: int) -> int:
        """Deterministic heuristic used when no trained artifact is available."""
        is_fraud = (
            amount > MOCK_AMOUNT_THRESHOLD
            and distance_from_home > MOCK_DISTANCE_THRESHOLD
            and use_chip == 0
        )
        return 1 if is_fraud else 0

    def predict(self, amount: float, distance_from_home: float, use_chip: int) -> int:
        """Return ``1`` if the transaction is predicted fraudulent, else ``0``."""
        if self.is_mock or self.model is None:
            return self._mock_predict(amount, distance_from_home, use_chip)

        # Feature vector in the exact column order used at training time.
        features = np.array(
            [[float(amount), float(distance_from_home), int(use_chip)]],
            dtype=float,
        )
        prediction = self.model.predict(features)
        return int(prediction[0])


# Module-level singleton: the artifact is loaded once at import time, not per
# request. FastAPI handlers import this instance directly.
fraud_model = FraudDetectionModel()
