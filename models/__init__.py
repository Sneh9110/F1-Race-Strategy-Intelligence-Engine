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
    ModelTrainer as DegradationModelTrainer,
    ModelRegistry as DegradationModelRegistry,
    FallbackHeuristics as DegradationFallbackHeuristics
)

# Lap time prediction models
from models.lap_time import (
    BaseLapTimeModel,
    XGBoostLapTimeModel,
    LightGBMLapTimeModel,
    EnsembleLapTimeModel,
    LapTimePredictor,
    ModelTrainer as LapTimeModelTrainer,
    ModelRegistry as LapTimeModelRegistry,
    FallbackHeuristics as LapTimeFallbackHeuristics
)

__all__ = [
    # Tire degradation
    'BaseDegradationModel',
    'XGBoostDegradationModel',
    'LightGBMDegradationModel',
    'EnsembleDegradationModel',
    'DegradationPredictor',
    'DegradationModelTrainer',
    'DegradationModelRegistry',
    'DegradationFallbackHeuristics',
    
    # Lap time prediction
    'BaseLapTimeModel',
    'XGBoostLapTimeModel',
    'LightGBMLapTimeModel',
    'EnsembleLapTimeModel',
    'LapTimePredictor',
    'LapTimeModelTrainer',
    'LapTimeModelRegistry',
    'LapTimeFallbackHeuristics',
]

__version__ = '1.0.0'
