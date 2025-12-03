from typing import Optional
from .base import BasePitStopLossModel, PredictionInput, PredictionOutput, ModelConfig
import joblib

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


class XGBoostPitStopLossModel(BasePitStopLossModel):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self.model = None

    def train(self, X, y, **kwargs):
        if XGBRegressor is None:
            raise RuntimeError("xgboost not available")
        params = self.config.hyperparameters if self.config else {}
        self.model = XGBRegressor(**params)
        self.model.fit(X, y, **kwargs)

    def predict(self, inp: PredictionInput) -> PredictionOutput:
        tot = self.predict_total_loss(inp)
        base = 0.0
        pit_delta = tot - base
        congestion = inp.cars_in_pit_window * 1.0
        window_sens = 0.5
        return PredictionOutput(total_pit_loss=float(tot), pit_delta=float(pit_delta), window_sensitivity=float(window_sens), congestion_penalty=float(congestion), base_pit_loss=float(base), confidence=0.7, metadata={})

    def predict_total_loss(self, inp: PredictionInput) -> float:
        feat = self._extract_features(inp)
        if self.model is None:
            raise RuntimeError("model not trained or loaded")
        import pandas as pd
        X = pd.DataFrame([feat])
        pred = self.model.predict(X)[0]
        return float(pred)

    def predict_window_sensitivity(self, inp: PredictionInput) -> float:
        return 0.5

    def save(self, path: str):
        joblib.dump({"model": self.model}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data.get("model")
