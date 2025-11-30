"""Tests for XGBoost tire degradation model."""

import pytest
import numpy as np

from models.tire_degradation.xgboost_model import XGBoostDegradationModel


def test_xgboost_model_initialization(xgboost_model):
    """Test model initialization."""
    assert xgboost_model.model is None
    assert xgboost_model.feature_names == []


def test_xgboost_model_training(xgboost_model, sample_training_data):
    """Test model training."""
    X = sample_training_data.drop(columns=['degradation_rate', 'usable_life', 'has_cliff'])
    y = sample_training_data['degradation_rate']
    
    # Train model
    metrics = xgboost_model.train(X, y)
    
    # Check training completed
    assert xgboost_model.model is not None
    assert len(xgboost_model.feature_names) > 0
    assert 'train_samples' in metrics


def test_xgboost_prediction(trained_xgboost_model, sample_prediction_input):
    """Test prediction."""
    output = trained_xgboost_model.predict(sample_prediction_input)
    
    assert output.degradation_rate > 0
    assert len(output.degradation_curve) > 0
    assert output.usable_life > 0
    assert 0 <= output.confidence <= 1


def test_xgboost_predict_curve(trained_xgboost_model, sample_prediction_input):
    """Test curve prediction."""
    curve = trained_xgboost_model.predict_curve(sample_prediction_input, num_laps=30)
    
    assert len(curve) == 30
    assert all(isinstance(v, float) for v in curve)
    assert all(v >= 0 for v in curve)


def test_xgboost_predict_usable_life(trained_xgboost_model, sample_prediction_input):
    """Test usable life prediction."""
    life = trained_xgboost_model.predict_usable_life(sample_prediction_input)
    
    assert isinstance(life, int)
    assert life > 0


def test_xgboost_save_load(trained_xgboost_model, sample_prediction_input, temp_model_dir):
    """Test model save and load."""
    # Get prediction before save
    output_before = trained_xgboost_model.predict(sample_prediction_input)
    
    # Save model
    trained_xgboost_model.save(temp_model_dir)
    
    # Create new model and load
    from models.tire_degradation.base import ModelConfig
    new_model = XGBoostDegradationModel(ModelConfig(version='1.0.0'))
    new_model.load(temp_model_dir)
    
    # Get prediction after load
    output_after = new_model.predict(sample_prediction_input)
    
    # Predictions should be similar
    assert abs(output_before.degradation_rate - output_after.degradation_rate) < 0.01
