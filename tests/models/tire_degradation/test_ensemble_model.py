"""Tests for ensemble tire degradation model."""

import pytest

from models.tire_degradation.ensemble_model import EnsembleDegradationModel


def test_ensemble_initialization(ensemble_model):
    """Test ensemble initialization."""
    assert 'xgboost' in ensemble_model.models
    assert 'lightgbm' in ensemble_model.models
    assert ensemble_model.weights['xgboost'] + ensemble_model.weights['lightgbm'] == 1.0


def test_ensemble_training(ensemble_model, sample_training_data):
    """Test ensemble training."""
    X = sample_training_data.drop(columns=['degradation_rate', 'usable_life', 'has_cliff'])
    y = sample_training_data['degradation_rate']
    
    metrics = ensemble_model.train(X, y)
    
    assert 'xgboost' in metrics
    assert 'lightgbm' in metrics


def test_ensemble_prediction(ensemble_model, sample_training_data, sample_prediction_input):
    """Test ensemble prediction."""
    X = sample_training_data.drop(columns=['degradation_rate', 'usable_life', 'has_cliff'])
    y = sample_training_data['degradation_rate']
    
    ensemble_model.train(X, y)
    output = ensemble_model.predict(sample_prediction_input)
    
    assert 'ensemble_models' in output.metadata
    assert 'weights' in output.metadata
