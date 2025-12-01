"""
Test fixtures and configuration for lap time model tests.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from models.lap_time.base import (
    PredictionInput,
    PredictionOutput,
    RaceCondition,
    ModelConfig
)
from models.lap_time.xgboost_model import XGBoostLapTimeModel
from models.lap_time.lightgbm_model import LightGBMLapTimeModel
from models.lap_time.ensemble_model import EnsembleLapTimeModel


@pytest.fixture
def sample_prediction_input() -> PredictionInput:
    """Create sample prediction input for testing."""
    return PredictionInput(
        tire_age=10,
        tire_compound="SOFT",
        fuel_load=50.0,
        traffic_state=RaceCondition.CLEAN_AIR,
        gap_to_ahead=None,
        safety_car_active=False,
        weather_temp=25.0,
        track_temp=35.0,
        track_name="Monaco",
        driver_number=44,
        lap_number=15,
        session_progress=0.3,
        recent_lap_times=[88.5, 88.7, 88.4, 88.6, 88.5]
    )


@pytest.fixture
def sample_prediction_inputs() -> List[PredictionInput]:
    """Create multiple prediction inputs for batch testing."""
    return [
        PredictionInput(
            tire_age=i * 5,
            tire_compound="SOFT" if i % 2 == 0 else "MEDIUM",
            fuel_load=100.0 - (i * 10),
            traffic_state=RaceCondition.CLEAN_AIR if i % 3 == 0 else RaceCondition.DIRTY_AIR,
            gap_to_ahead=None if i % 3 == 0 else 0.5,
            safety_car_active=i % 5 == 0,
            weather_temp=25.0,
            track_temp=35.0,
            track_name="Monaco",
            driver_number=44,
            lap_number=i + 1,
            session_progress=i / 20.0,
            recent_lap_times=[88.5 + np.random.randn() * 0.3 for _ in range(5)]
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_training_data() -> tuple:
    """Create synthetic training data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    data = pd.DataFrame({
        'tire_age': np.random.randint(0, 40, n_samples),
        'tire_compound_encoded': np.random.randint(0, 3, n_samples),
        'fuel_load': np.random.uniform(0, 110, n_samples),
        'traffic_penalty': np.random.uniform(0, 0.8, n_samples),
        'safety_car_flag': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'weather_temp': np.random.uniform(15, 35, n_samples),
        'track_temp': np.random.uniform(20, 45, n_samples),
        'track_encoded': np.random.randint(0, 10, n_samples),
        'driver_aggression': np.random.uniform(0.5, 1.5, n_samples),
        'degradation_slope': np.random.uniform(0.03, 0.08, n_samples),
        'rolling_avg_pace': np.random.uniform(85, 95, n_samples),
        'sector_consistency': np.random.uniform(0.8, 1.0, n_samples),
        'lap_number': np.random.randint(1, 60, n_samples),
        'session_progress': np.random.uniform(0, 1, n_samples),
    })
    
    # Generate target (synthetic lap times)
    lap_time = (
        90.0 +  # Base lap time
        data['tire_age'] * 0.05 +  # Tire degradation
        data['fuel_load'] * 0.03 +  # Fuel effect
        data['traffic_penalty'] +  # Traffic
        (data['weather_temp'] - 25) * 0.02 +  # Weather
        data['safety_car_flag'] * 30 +  # Safety car
        np.random.randn(n_samples) * 0.5  # Noise
    )
    
    data['lap_time'] = lap_time
    
    # Split data
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    feature_names = [col for col in data.columns if col != 'lap_time']
    
    return train_data, val_data, test_data, feature_names


@pytest.fixture
def model_config() -> ModelConfig:
    """Create model configuration for testing."""
    return ModelConfig(
        name="lap_time_test",
        version="1.0.0",
        model_type="test"
    )


@pytest.fixture
def xgboost_model(model_config) -> XGBoostLapTimeModel:
    """Create XGBoost model instance for testing."""
    return XGBoostLapTimeModel(model_config)


@pytest.fixture
def lightgbm_model(model_config) -> LightGBMLapTimeModel:
    """Create LightGBM model instance for testing."""
    return LightGBMLapTimeModel(model_config)


@pytest.fixture
def ensemble_model(model_config) -> EnsembleLapTimeModel:
    """Create Ensemble model instance for testing."""
    return EnsembleLapTimeModel(model_config)


@pytest.fixture
def trained_xgboost_model(xgboost_model, sample_training_data) -> XGBoostLapTimeModel:
    """Create trained XGBoost model for testing."""
    train_data, val_data, _, _ = sample_training_data
    X_train = train_data.drop('lap_time', axis=1)
    y_train = train_data['lap_time']
    X_val = val_data.drop('lap_time', axis=1)
    y_val = val_data['lap_time']
    
    xgboost_model.train(X_train, y_train, X_val, y_val, n_estimators=50)
    return xgboost_model


@pytest.fixture
def trained_lightgbm_model(lightgbm_model, sample_training_data) -> LightGBMLapTimeModel:
    """Create trained LightGBM model for testing."""
    train_data, val_data, _, _ = sample_training_data
    X_train = train_data.drop('lap_time', axis=1)
    y_train = train_data['lap_time']
    X_val = val_data.drop('lap_time', axis=1)
    y_val = val_data['lap_time']
    
    lightgbm_model.train(X_train, y_train, X_val, y_val, n_estimators=50)
    return lightgbm_model


@pytest.fixture
def trained_ensemble_model(ensemble_model, sample_training_data) -> EnsembleLapTimeModel:
    """Create trained Ensemble model for testing."""
    train_data, val_data, _, _ = sample_training_data
    X_train = train_data.drop('lap_time', axis=1)
    y_train = train_data['lap_time']
    X_val = val_data.drop('lap_time', axis=1)
    y_val = val_data['lap_time']
    
    ensemble_model.train(X_train, y_train, X_val, y_val, n_estimators=50)
    return ensemble_model


@pytest.fixture
def temp_model_dir(tmp_path) -> Path:
    """Create temporary directory for model saving/loading tests."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def temp_registry_dir(tmp_path) -> Path:
    """Create temporary directory for registry tests."""
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    return registry_dir
