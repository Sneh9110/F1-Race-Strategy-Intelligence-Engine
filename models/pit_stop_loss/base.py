from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
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


class PredictionInput(BaseModel):
    track_name: str
    current_lap: int
    cars_in_pit_window: int
    pit_stop_duration: float
    traffic_density: float
    tire_compound_change: bool
    current_position: int
    gap_to_ahead: Optional[float] = None
    gap_to_behind: Optional[float] = None
    fuel_load: Optional[float] = None
    weather_conditions: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

    @validator("current_lap")
    def lap_range(cls, v):
        if v < 1:
            raise ValueError("current_lap must be >=1")
        return v

    @validator("traffic_density")
    def traffic_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("traffic_density must be in [0,1]")
        return v


class PredictionOutput(BaseModel):
    total_pit_loss: float
    pit_delta: float
    window_sensitivity: float
    congestion_penalty: float
    base_pit_loss: float
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = {}


class BasePitStopLossModel(ABC):
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(model_type="base_pit")

    @abstractmethod
    def train(self, X, y, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, inp: PredictionInput) -> PredictionOutput:
        raise NotImplementedError()

    @abstractmethod
    def predict_total_loss(self, inp: PredictionInput) -> float:
        raise NotImplementedError()

    @abstractmethod
    def predict_window_sensitivity(self, inp: PredictionInput) -> float:
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError()

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError()

    def _extract_features(self, inp: PredictionInput) -> Dict[str, Any]:
        features = {
            "cars_in_pit_window": inp.cars_in_pit_window,
            "pit_stop_duration": inp.pit_stop_duration,
            "traffic_density": inp.traffic_density,
            "tire_change": float(inp.tire_compound_change),
        }
        if inp.gap_to_ahead is not None:
            features["gap_to_ahead"] = inp.gap_to_ahead
        return features
