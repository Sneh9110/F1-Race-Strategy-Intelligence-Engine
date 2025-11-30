"""
ML Models Package - Predictive models for race strategy

Contains tire degradation, lap time, safety car, and pit stop loss models.
"""

# Tire degradation models
from models.tire_degradation import (
    BaseDegradationModel,
    XGBoostDegradationModel,
    LightGBMDegradationModel,
    EnsembleDegradationModel,
    DegradationPredictor,
    ModelTrainer,
    ModelRegistry,
    FallbackHeuristics
)

__all__ = [
    # Tire degradation
    'BaseDegradationModel',
    'XGBoostDegradationModel',
    'LightGBMDegradationModel',
    'EnsembleDegradationModel',
    'DegradationPredictor',
    'ModelTrainer',
    'ModelRegistry',
    'FallbackHeuristics',
]

__version__ = '1.0.0'
