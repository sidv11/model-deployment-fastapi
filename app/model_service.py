"""
Everything about loading the trained model and turning one applicant
into one prediction lives here, separate from the web plumbing in
main.py. This way the model logic can be tested on its own, without
spinning up a server.
"""

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "loan_default_model.joblib"
DECISION_THRESHOLD = 0.5


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested before the model has loaded."""


class LoanDefaultModel:
    """A thin, testable wrapper around the trained LightGBM pipeline."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_order: list[str] = []
        self.model_version = "unknown"
        self.metrics: dict = {}

    def load(self) -> None:
        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.feature_order = bundle["feature_order"]
        self.model_version = bundle.get("model_version", "unknown")
        self.metrics = bundle.get("metrics", {})

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _risk_band(self, probability: float) -> str:
        if probability < 0.2:
            return "low"
        if probability < 0.5:
            return "medium"
        return "high"

    def predict_one(self, applicant: dict) -> dict:
        if not self.is_loaded:
            raise ModelNotLoadedError("Model is not loaded yet.")

        row = pd.DataFrame([applicant])[self.feature_order]
        probability = float(self.model.predict_proba(row)[0, 1])

        return {
            "default_probability": round(probability, 4),
            "will_default": probability >= DECISION_THRESHOLD,
            "risk_band": self._risk_band(probability),
            "decision_threshold": DECISION_THRESHOLD,
            "model_version": self.model_version,
        }

    def predict_many(self, applicants: list[dict]) -> list[dict]:
        return [self.predict_one(a) for a in applicants]


# A single shared instance the app loads once at startup and reuses
# for every request, instead of reloading the model from disk each time.
model_service = LoanDefaultModel()
