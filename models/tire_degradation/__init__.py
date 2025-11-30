"""
Tire Degradation ML Model Package

This package provides production-grade machine learning models for predicting tire
degradation in Formula 1 races. It includes multiple model implementations, training
infrastructure, inference API, and fallback heuristics.

Models:
    - XGBoostDegradationModel: Fast inference optimized model (<200ms)
    - LightGBMDegradationModel: High accuracy model with uncertainty estimation
    - EnsembleDegradationModel: Robust ensemble combining multiple models

Main Components:
    - TireDegradationModel: Base interface for all degradation models
    - DegradationPredictor: Fast inference API with caching and fallback
    - ModelTrainer: Training orchestrator with hyperparameter optimization
    - ModelRegistry: Version management and model lifecycle
    - FallbackHeuristics: Physics-based backup predictions

Model Inputs:
    - tire_compound: Tire compound (SOFT, MEDIUM, HARD)
    - tire_age: Current tire age in laps (0-50)
    - stint_history: Previous stint performance data
    - weather_temp: Track/air temperature (°C)
    - driver_aggression: Driver style metric (0-1)
    - track_name: Circuit identifier

Model Outputs:
    - degradation_curve: Lap-by-lap degradation prediction (List[float])
    - usable_life: Predicted tire usable life in laps (int)
    - dropoff_lap: Predicted cliff/dropoff lap (Optional[int])
    - confidence: Prediction confidence score (0-1)

Usage Example:
    from models.tire_degradation import DegradationPredictor, PredictionInput
    
    predictor = DegradationPredictor(model_version='latest')
    
    input_data = PredictionInput(
        tire_compound='SOFT',
        tire_age=10,
        stint_history=[78.5, 78.7, 78.9, 79.2, 79.5],
        weather_temp=42.0,
        driver_aggression=0.85,
        track_name='Monaco'
    )
    
    result = predictor.predict(input_data)
    print(f"Usable life: {result.usable_life} laps")
    print(f"Dropoff lap: {result.dropoff_lap}")
"""

from models.tire_degradation.base import (
    BaseDegradationModel,
    ModelConfig,
    PredictionInput,
    PredictionOutput
)
from models.tire_degradation.xgboost_model import XGBoostDegradationModel
from models.tire_degradation.lightgbm_model import LightGBMDegradationModel
from models.tire_degradation.ensemble_model import EnsembleDegradationModel
from models.tire_degradation.inference import DegradationPredictor
from models.tire_degradation.training import ModelTrainer
from models.tire_degradation.registry import ModelRegistry
from models.tire_degradation.fallback import FallbackHeuristics

# Alias for backward compatibility
TireDegradationModel = BaseDegradationModel

__all__ = [
    # Base classes
    'BaseDegradationModel',
    'TireDegradationModel',
    'ModelConfig',
    'PredictionInput',
    'PredictionOutput',
    
    # Model implementations
    'XGBoostDegradationModel',
    'LightGBMDegradationModel',
    'EnsembleDegradationModel',
    
    # Infrastructure
    'DegradationPredictor',
    'ModelTrainer',
    'ModelRegistry',
    'FallbackHeuristics',
]

__version__ = '1.0.0'
__author__ = 'F1 Race Strategy Intelligence Team'
