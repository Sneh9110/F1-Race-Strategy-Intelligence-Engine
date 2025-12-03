"""Safety Car Probability model package.

This package provides safety car probability prediction models and helpers.
Exports: BaseSafetyCarModel, XGBoostSafetyCarModel, LightGBMSafetyCarModel,
EnsembleSafetyCarModel, SafetyCarPredictor, SafetyCarModelTrainer,
SafetyCarModelRegistry, SafetyCarFallbackHeuristics, PredictionInput,
PredictionOutput, ModelConfig

Example:
    from models.safety_car import SafetyCarPredictor
    predictor = SafetyCarPredictor()
    out = predictor.predict(sample_input)
"""
__version__ = "1.0.0"

from .base import BaseSafetyCarModel, PredictionInput, PredictionOutput, ModelConfig
from .xgboost_model import XGBoostSafetyCarModel
from .lightgbm_model import LightGBMSafetyCarModel
from .ensemble_model import EnsembleSafetyCarModel
from .training import ModelTrainer as SafetyCarModelTrainer
from .registry import ModelRegistry as SafetyCarModelRegistry
from .inference import SafetyCarPredictor
from .fallback import FallbackHeuristics as SafetyCarFallbackHeuristics

__all__ = [
    "BaseSafetyCarModel",
    "PredictionInput",
    "PredictionOutput",
    "ModelConfig",
    "XGBoostSafetyCarModel",
    "LightGBMSafetyCarModel",
    "EnsembleSafetyCarModel",
    "SafetyCarModelTrainer",
    "SafetyCarModelRegistry",
    "SafetyCarPredictor",
    "SafetyCarFallbackHeuristics",
]
