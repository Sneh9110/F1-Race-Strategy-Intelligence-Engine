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

# Safety car probability models
from models.safety_car import (
    BaseSafetyCarModel,
    XGBoostSafetyCarModel,
    LightGBMSafetyCarModel,
    EnsembleSafetyCarModel,
    SafetyCarPredictor,
    SafetyCarModelTrainer,
    SafetyCarModelRegistry,
    SafetyCarFallbackHeuristics
)

# Pit stop loss models
from models.pit_stop_loss import (
    BasePitStopLossModel,
    XGBoostPitStopLossModel,
    LightGBMPitStopLossModel,
    EnsemblePitStopLossModel,
    PitStopLossPredictor,
    PitStopLossModelTrainer,
    PitStopLossModelRegistry,
    PitStopLossFallbackHeuristics
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
    
    # Safety car probability
    'BaseSafetyCarModel',
    'XGBoostSafetyCarModel',
    'LightGBMSafetyCarModel',
    'EnsembleSafetyCarModel',
    'SafetyCarPredictor',
    'SafetyCarModelTrainer',
    'SafetyCarModelRegistry',
    'SafetyCarFallbackHeuristics',
    
    # Pit stop loss
    'BasePitStopLossModel',
    'XGBoostPitStopLossModel',
    'LightGBMPitStopLossModel',
    'EnsemblePitStopLossModel',
    'PitStopLossPredictor',
    'PitStopLossModelTrainer',
    'PitStopLossModelRegistry',
    'PitStopLossFallbackHeuristics',
]

__version__ = '1.1.0'
