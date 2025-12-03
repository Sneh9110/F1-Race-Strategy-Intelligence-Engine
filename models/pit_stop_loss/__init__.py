"""Pit Stop Loss model package.

This package provides pit stop loss regression models and helpers.
Exports: BasePitStopLossModel, XGBoostPitStopLossModel, LightGBMPitStopLossModel,
EnsemblePitStopLossModel, PitStopLossPredictor, PitStopLossModelTrainer,
PitStopLossModelRegistry, PitStopLossFallbackHeuristics, PredictionInput,
PredictionOutput, ModelConfig

Example:
    from models.pit_stop_loss import PitStopLossPredictor
    predictor = PitStopLossPredictor()
    out = predictor.predict(sample_input)
"""
__version__ = "1.0.0"

from .base import BasePitStopLossModel, PredictionInput, PredictionOutput, ModelConfig
from .xgboost_model import XGBoostPitStopLossModel
from .lightgbm_model import LightGBMPitStopLossModel
from .ensemble_model import EnsemblePitStopLossModel
from .training import ModelTrainer as PitStopLossModelTrainer
from .registry import ModelRegistry as PitStopLossModelRegistry
from .inference import PitStopLossPredictor
from .fallback import FallbackHeuristics as PitStopLossFallbackHeuristics

__all__ = [
    "BasePitStopLossModel",
    "PredictionInput",
    "PredictionOutput",
    "ModelConfig",
    "XGBoostPitStopLossModel",
    "LightGBMPitStopLossModel",
    "EnsemblePitStopLossModel",
    "PitStopLossModelTrainer",
    "PitStopLossModelRegistry",
    "PitStopLossPredictor",
    "PitStopLossFallbackHeuristics",
]
