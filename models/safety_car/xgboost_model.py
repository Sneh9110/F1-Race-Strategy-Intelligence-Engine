from typing import Any, Dict, Optional
from .base import BaseSafetyCarModel, PredictionInput, PredictionOutput, ModelConfig
import joblib

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


class XGBoostSafetyCarModel(BaseSafetyCarModel):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self.model = None
        self.feature_names = []

    def train(self, X, y, **kwargs):
        params = self.config.hyperparameters if self.config else {}
        if XGBClassifier is None:
            raise RuntimeError("xgboost not available")
        self.model = XGBClassifier(**params)
        self.model.fit(X, y, **kwargs)
        self.feature_names = list(X.columns) if hasattr(X, "columns") else []

    def predict(self, inp: PredictionInput) -> PredictionOutput:
        prob = self.predict_probability(inp)
        window = self.predict_deployment_window(prob, inp)
        confidence = float(min(max(prob, 0.0), 1.0))
        risk = {"model_prob": prob}
        return PredictionOutput(sc_probability=prob, deployment_window=window, confidence=confidence, risk_factors=risk, metadata={})

    def predict_probability(self, inp: PredictionInput) -> float:
        feat = self._extract_features(inp)
        if self.model is None:
            raise RuntimeError("model not trained or loaded")
        # simplistic: require DataFrame in production
        # here we try to build single-row array
        import pandas as pd
        X = pd.DataFrame([feat])
        p = self.model.predict_proba(X)[0, 1]
        return float(p)

    def predict_deployment_window(self, prob: float, inp: PredictionInput):
        if prob > 0.7:
            return (inp.current_lap + 1, min(inp.total_laps, inp.current_lap + 8))
        if prob > 0.4:
            return (inp.current_lap + 2, min(inp.total_laps, inp.current_lap + 15))
        return None

    def save(self, path: str):
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data.get("model")
        self.feature_names = data.get("feature_names", [])
