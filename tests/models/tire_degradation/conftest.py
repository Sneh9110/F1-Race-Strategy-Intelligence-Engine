"""Test configuration and fixtures for tire degradation models."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

from models.tire_degradation.base import PredictionInput, ModelConfig
from models.tire_degradation.xgboost_model import XGBoostDegradationModel
from models.tire_degradation.lightgbm_model import LightGBMDegradationModel
from models.tire_degradation.ensemble_model import EnsembleDegradationModel
from data_pipeline.schemas.historical_schema import HistoricalStint


@pytest.fixture
def sample_prediction_input() -> PredictionInput:
    """Sample prediction input for testing."""
    return PredictionInput(
        tire_compound='MEDIUM',
        tire_age=10,
        stint_history=[
            {'lap': 1, 'lap_time': 90.5},
            {'lap': 2, 'lap_time': 90.8},
            {'lap': 3, 'lap_time': 91.2}
        ],
        weather_temp=25.0,
        driver_aggression=0.5,
        track_name='Monza'
    )


@pytest.fixture
def sample_training_data() -> pd.DataFrame:
    """Generate sample training data."""
    np.random.seed(42)
    
    n_samples = 100
    data = {
        'tire_compound': np.random.choice(['SOFT', 'MEDIUM', 'HARD'], n_samples),
        'tire_age': np.random.randint(0, 40, n_samples),
        'weather_temp': np.random.uniform(15, 35, n_samples),
        'driver_aggression': np.random.uniform(0.3, 0.8, n_samples),
        'track_name': np.random.choice(['Monza', 'Spa', 'Monaco'], n_samples),
        'avg_lap_time': np.random.uniform(85, 95, n_samples),
        'degradation_rate': np.random.uniform(0.03, 0.15, n_samples),
        'usable_life': np.random.randint(15, 40, n_samples),
        'has_cliff': np.random.choice([True, False], n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_stint_history() -> List[HistoricalStint]:
    """Sample stint history for testing."""
    # This is a simplified version - adjust based on actual schema
    return []


@pytest.fixture
def model_config() -> ModelConfig:
    """Sample model configuration."""
    return ModelConfig(
        version='1.0.0',
        hyperparameters={
            'max_depth': 4,
            'learning_rate': 0.1,
            'n_estimators': 10  # Small for fast testing
        }
    )


@pytest.fixture
def xgboost_model(model_config) -> XGBoostDegradationModel:
    """XGBoost model instance."""
    return XGBoostDegradationModel(model_config)


@pytest.fixture
def lightgbm_model(model_config) -> LightGBMDegradationModel:
    """LightGBM model instance."""
    return LightGBMDegradationModel(model_config)


@pytest.fixture
def ensemble_model(model_config) -> EnsembleDegradationModel:
    """Ensemble model instance."""
    config = ModelConfig(
        version='1.0.0',
        hyperparameters={
            'weights': {'xgboost': 0.5, 'lightgbm': 0.5}
        }
    )
    return EnsembleDegradationModel(config)


@pytest.fixture
def trained_xgboost_model(xgboost_model, sample_training_data) -> XGBoostDegradationModel:
    """Pre-trained XGBoost model."""
    X = sample_training_data.drop(columns=['degradation_rate', 'usable_life', 'has_cliff'])
    y = sample_training_data['degradation_rate']
    
    xgboost_model.train(X, y)
    return xgboost_model


@pytest.fixture
def trained_lightgbm_model(lightgbm_model, sample_training_data) -> LightGBMDegradationModel:
    """Pre-trained LightGBM model."""
    X = sample_training_data.drop(columns=['degradation_rate', 'usable_life', 'has_cliff'])
    y = sample_training_data['degradation_rate']
    
    lightgbm_model.train(X, y)
    return lightgbm_model


@pytest.fixture
def temp_model_dir(tmp_path) -> Path:
    """Temporary directory for model saving/loading."""
    model_dir = tmp_path / 'models'
    model_dir.mkdir()
    return model_dir
