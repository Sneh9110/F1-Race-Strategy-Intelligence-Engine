"""Tests for LightGBM tire degradation model."""

import pytest

from models.tire_degradation.lightgbm_model import LightGBMDegradationModel


def test_lightgbm_model_initialization(lightgbm_model):
    """Test model initialization."""
    assert lightgbm_model.model is None
    assert lightgbm_model.feature_names == []


def test_lightgbm_model_training(lightgbm_model, sample_training_data):
    """Test model training."""
    X = sample_training_data.drop(columns=['degradation_rate', 'usable_life', 'has_cliff'])
    y = sample_training_data['degradation_rate']
    
    metrics = lightgbm_model.train(X, y)
    
    assert lightgbm_model.model is not None
    assert len(lightgbm_model.feature_names) > 0
    assert 'train_samples' in metrics


def test_lightgbm_prediction(trained_lightgbm_model, sample_prediction_input):
    """Test prediction with uncertainty."""
    output = trained_lightgbm_model.predict(sample_prediction_input)
    
    assert output.degradation_rate > 0
    assert len(output.degradation_curve) > 0
    assert 'uncertainty' in output.metadata
    assert 0 <= output.confidence <= 1
