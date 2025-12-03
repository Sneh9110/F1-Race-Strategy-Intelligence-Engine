from typing import Optional, Dict, Any
from .base import BaseSafetyCarModel, PredictionInput, PredictionOutput, ModelConfig
from .xgboost_model import XGBoostSafetyCarModel
from .lightgbm_model import LightGBMSafetyCarModel
import joblib


class EnsembleSafetyCarModel(BaseSafetyCarModel):
    def __init__(self, config: Optional[ModelConfig] = None, weights: Optional[Dict[str, float]] = None):
        super().__init__(config)
        self.xgb = XGBoostSafetyCarModel(config)
        self.lgb = LightGBMSafetyCarModel(config)
        self.weights = weights or {"xgboost": 0.4, "lightgbm": 0.6}

    def train(self, X, y, **kwargs):
        # train both
        self.xgb.train(X, y, **kwargs)
        self.lgb.train(X, y, **kwargs)

    def predict(self, inp: PredictionInput) -> PredictionOutput:
        p1 = self.xgb.predict_probability(inp)
        p2 = self.lgb.predict_probability(inp)
        w1 = self.weights.get("xgboost", 0.5)
        w2 = self.weights.get("lightgbm", 0.5)
        prob = w1 * p1 + w2 * p2
        # simple window: intersection if both non-null
        wA = self.xgb.predict_deployment_window(p1, inp)
        wB = self.lgb.predict_deployment_window(p2, inp)
        window = None
        if wA and wB:
            start = max(wA[0], wB[0])
            end = min(wA[1], wB[1])
            if start <= end:
                window = (start, end)
            else:
                window = (min(wA[0], wB[0]), max(wA[1], wB[1]))
        else:
            window = wA or wB
        confidence = float(prob * (1 - abs(p1 - p2)))
        risk = {"xgboost": p1, "lightgbm": p2}
        return PredictionOutput(sc_probability=prob, deployment_window=window, confidence=confidence, risk_factors=risk, metadata={})

    def predict_probability(self, inp: PredictionInput) -> float:
        return self.predict(inp).sc_probability

    def predict_deployment_window(self, prob: float, inp: PredictionInput):
        return self.predict(inp).deployment_window

    def save(self, path: str):
        joblib.dump({"weights": self.weights}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.weights = data.get("weights", self.weights)
