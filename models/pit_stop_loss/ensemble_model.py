from typing import Optional, Dict
from .base import BasePitStopLossModel, PredictionInput, PredictionOutput, ModelConfig
from .xgboost_model import XGBoostPitStopLossModel
from .lightgbm_model import LightGBMPitStopLossModel
import joblib


class EnsemblePitStopLossModel(BasePitStopLossModel):
    def __init__(self, config: Optional[ModelConfig] = None, weights: Optional[Dict[str, float]] = None):
        super().__init__(config)
        self.xgb = XGBoostPitStopLossModel(config)
        self.lgb = LightGBMPitStopLossModel(config)
        self.weights = weights or {"xgboost": 0.5, "lightgbm": 0.5}

    def train(self, X, y, **kwargs):
        self.xgb.train(X, y, **kwargs)
        self.lgb.train(X, y, **kwargs)

    def predict(self, inp: PredictionInput) -> PredictionOutput:
        p1 = self.xgb.predict_total_loss(inp)
        p2 = self.lgb.predict_total_loss(inp)
        w1 = self.weights.get("xgboost", 0.5)
        w2 = self.weights.get("lightgbm", 0.5)
        tot = w1 * p1 + w2 * p2
        pit_delta = tot - 0.0
        congestion = inp.cars_in_pit_window * 1.0
        window_sens = max(self.xgb.predict_window_sensitivity(inp), self.lgb.predict_window_sensitivity(inp))
        confidence = 1.0 - abs(p1 - p2) / max(1.0, abs(p1) + abs(p2))
        return PredictionOutput(total_pit_loss=tot, pit_delta=pit_delta, window_sensitivity=window_sens, congestion_penalty=congestion, base_pit_loss=0.0, confidence=confidence, metadata={})

    def save(self, path: str):
        joblib.dump({"weights": self.weights}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.weights = data.get("weights", self.weights)
