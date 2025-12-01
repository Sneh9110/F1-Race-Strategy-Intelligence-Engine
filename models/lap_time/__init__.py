"""
Lap Time Prediction Models Package

Predicts race pace under varying conditions:
- Clean air (baseline performance)
- Dirty air (traffic penalty)
- Safety car (reduced pace)
- Aging tires (degradation effect)
- Fuel load (weight effect)

Model Inputs:
    - tire_age: Tire age in laps (0-50)
    - tire_compound: Tire compound (SOFT/MEDIUM/HARD/INTERMEDIATE/WET)
    - fuel_load: Fuel load in kg (0-110)
    - traffic_state: Clean air or dirty air condition
    - safety_car_active: Safety car deployment flag
    - weather_temp: Weather temperature in °C
    - track_name: Circuit name
    - driver_number: Driver identifier
    - lap_number: Current lap number

Model Outputs:
    - predicted_lap_time: Predicted lap time in seconds
    - confidence: Prediction confidence (0-1)
    - pace_components: Breakdown of pace factors (base, tire, fuel, traffic, weather, safety car)
    - uncertainty_range: Confidence interval for prediction

Available Models:
    - XGBoostLapTimeModel: Fast inference (<150ms)
    - LightGBMLapTimeModel: High accuracy with uncertainty estimation
    - EnsembleLapTimeModel: Combined predictions with weighted voting

Usage:
    from models.lap_time import LapTimePredictor, PredictionInput
    
    predictor = LapTimePredictor(model_version='latest')
    
    input_data = PredictionInput(
        tire_age=15,
        tire_compound='MEDIUM',
        fuel_load=50.0,
        traffic_state='CLEAN_AIR',
        safety_car_active=False,
        weather_temp=28.0,
        track_name='Monza',
        driver_number=44,
        lap_number=20
    )
    
    output = predictor.predict(input_data)
    print(f"Predicted lap time: {output.predicted_lap_time:.3f}s")
    print(f"Confidence: {output.confidence:.2f}")
"""

from models.lap_time.base import (
    BaseLapTimeModel,
    PredictionInput,
    PredictionOutput,
    ModelConfig,
    RaceCondition
)
from models.lap_time.xgboost_model import XGBoostLapTimeModel
from models.lap_time.lightgbm_model import LightGBMLapTimeModel
from models.lap_time.ensemble_model import EnsembleLapTimeModel
from models.lap_time.inference import LapTimePredictor
from models.lap_time.training import ModelTrainer
from models.lap_time.registry import ModelRegistry
from models.lap_time.fallback import FallbackHeuristics

__all__ = [
    # Base classes
    'BaseLapTimeModel',
    'PredictionInput',
    'PredictionOutput',
    'ModelConfig',
    'RaceCondition',
    
    # Model implementations
    'XGBoostLapTimeModel',
    'LightGBMLapTimeModel',
    'EnsembleLapTimeModel',
    
    # Production components
    'LapTimePredictor',
    'ModelTrainer',
    'ModelRegistry',
    'FallbackHeuristics',
]

__version__ = '1.0.0'
