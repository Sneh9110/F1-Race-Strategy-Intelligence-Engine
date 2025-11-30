"""Tests for base tire degradation model."""

import pytest
import numpy as np

from models.tire_degradation.base import (
    BaseDegradationModel,
    PredictionInput,
    PredictionOutput,
    ModelConfig,
    TireCompound
)


def test_tire_compound_enum():
    """Test TireCompound enum."""
    assert TireCompound.SOFT.value == 'SOFT'
    assert TireCompound.MEDIUM.value == 'MEDIUM'
    assert TireCompound.HARD.value == 'HARD'


def test_prediction_input_validation(sample_prediction_input):
    """Test PredictionInput validation."""
    # Valid input
    assert sample_prediction_input.tire_compound == 'MEDIUM'
    assert sample_prediction_input.tire_age == 10
    
    # Invalid tire age
    with pytest.raises(ValueError):
        PredictionInput(
            tire_compound='MEDIUM',
            tire_age=-5,  # Invalid
            stint_history=[],
            weather_temp=25.0,
            driver_aggression=0.5,
            track_name='Monza'
        )
    
    # Invalid temperature
    with pytest.raises(ValueError):
        PredictionInput(
            tire_compound='MEDIUM',
            tire_age=10,
            stint_history=[],
            weather_temp=100.0,  # Invalid
            driver_aggression=0.5,
            track_name='Monza'
        )
    
    # Invalid driver aggression
    with pytest.raises(ValueError):
        PredictionInput(
            tire_compound='MEDIUM',
            tire_age=10,
            stint_history=[],
            weather_temp=25.0,
            driver_aggression=1.5,  # Invalid
            track_name='Monza'
        )


def test_prediction_output_validation():
    """Test PredictionOutput validation."""
    # Valid output
    output = PredictionOutput(
        degradation_curve=[0.1, 0.15, 0.2],
        usable_life=20,
        dropoff_lap=15,
        confidence=0.85
    )
    
    assert len(output.degradation_curve) == 3
    assert output.usable_life == 20
    
    # Invalid confidence
    with pytest.raises(ValueError):
        PredictionOutput(
            degradation_curve=[0.1, 0.15],
            usable_life=20,
            dropoff_lap=15,
            confidence=1.5  # Invalid
        )
    
    # Invalid usable life
    with pytest.raises(ValueError):
        PredictionOutput(
            degradation_curve=[0.1, 0.15],
            usable_life=-5,  # Invalid
            dropoff_lap=15,
            confidence=0.85
        )


def test_model_config():
    """Test ModelConfig."""
    config = ModelConfig(
        version='1.0.0',
        hyperparameters={'max_depth': 6}
    )
    
    assert config.version == '1.0.0'
    assert config.hyperparameters['max_depth'] == 6
    
    # Test to_dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert 'version' in config_dict


def test_base_model_abstract(model_config):
    """Test that BaseDegradationModel cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseDegradationModel(model_config)
