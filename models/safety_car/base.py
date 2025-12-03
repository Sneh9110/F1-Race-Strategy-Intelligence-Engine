from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, validator
from abc import ABC, abstractmethod
import pandas as pd


class ModelConfig(BaseModel):
    model_type: str = Field(...)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    training_config: Dict[str, Any] = Field(default_factory=dict)
    inference_config: Dict[str, Any] = Field(default_factory=dict)
    fallback_config: Dict[str, Any] = Field(default_factory=dict)
    version: str = "0.0.0"


class IncidentLog(BaseModel):
    lap: int
    sector: Optional[str]
    severity: str  # 'minor','moderate','major'


class PredictionInput(BaseModel):
    track_name: str
    current_lap: int
    total_laps: int
    race_progress: float
    incident_logs: List[IncidentLog] = []
    driver_proximity_data: Optional[pd.DataFrame] = None
    sector_risks: Dict[str, float] = {}
    weather_conditions: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

    @validator("current_lap")
    def lap_range(cls, v, values):
        if v < 1:
            raise ValueError("current_lap must be >= 1")
        return v

    @validator("race_progress")
    def progress_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("race_progress must be in [0,1]")
        return v


class PredictionOutput(BaseModel):
    sc_probability: float = Field(..., ge=0.0, le=1.0)
    deployment_window: Optional[Tuple[int, int]] = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    risk_factors: Dict[str, float] = {}
    metadata: Dict[str, Any] = {}


class BaseSafetyCarModel(ABC):
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(model_type="base")

    @abstractmethod
    def train(self, X, y, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, inp: PredictionInput) -> PredictionOutput:
        raise NotImplementedError()

    @abstractmethod
    def predict_probability(self, inp: PredictionInput) -> float:
        raise NotImplementedError()

    @abstractmethod
    def predict_deployment_window(self, prob: float, inp: PredictionInput) -> Optional[Tuple[int, int]]:
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError()

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError()

    def _validate_input(self, inp: PredictionInput):
        # basic validation hook
        if inp.total_laps < inp.current_lap:
            raise ValueError("total_laps must be >= current_lap")

    def _extract_features(self, inp: PredictionInput) -> Dict[str, Any]:
        # placeholder: integrate with features.safety_car_features
        features = {
            "race_progress": inp.race_progress,
            "current_lap": inp.current_lap,
            "incident_count": len(inp.incident_logs),
        }
        # include sector risks
        for s, v in inp.sector_risks.items():
            features[f"sector_{s}"] = v
        return features
