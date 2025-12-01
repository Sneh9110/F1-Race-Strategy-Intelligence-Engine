"""
Tests for LightGBM lap time model.
"""

import pytest
import numpy as np

from models.lap_time.lightgbm_model import LightGBMLapTimeModel


class TestLightGBMLapTimeModel:
    """Test LightGBM model implementation."""
    
    def test_model_initialization(self, lightgbm_model):
        """Test model can be initialized."""
        assert lightgbm_model is not None
        assert lightgbm_model.model is None
    
    def test_model_training(self, lightgbm_model, sample_training_data):
        """Test model training with quantile regression."""
        train_data, val_data, _, _ = sample_training_data
        X_train = train_data.drop('lap_time', axis=1)
        y_train = train_data['lap_time']
        X_val = val_data.drop('lap_time', axis=1)
        y_val = val_data['lap_time']
        
        metrics = lightgbm_model.train(X_train, y_train, X_val, y_val, n_estimators=10)
        
        assert lightgbm_model.model is not None
        assert lightgbm_model.model_lower is not None  # Quantile models
        assert lightgbm_model.model_upper is not None
        assert 'uncertainty_coverage' in metrics
    
    def test_single_prediction_with_uncertainty(self, trained_lightgbm_model, sample_prediction_input):
        """Test prediction with uncertainty range."""
        output = trained_lightgbm_model.predict(sample_prediction_input)
        
        assert output.predicted_lap_time > 60.0
        assert output.uncertainty_range is not None
        lower, upper = output.uncertainty_range
        assert lower < output.predicted_lap_time < upper
    
    def test_batch_prediction(self, trained_lightgbm_model, sample_prediction_inputs):
        """Test batch predictions."""
        outputs = trained_lightgbm_model.predict_batch(sample_prediction_inputs)
        
        assert len(outputs) == len(sample_prediction_inputs)
        for output in outputs:
            assert output.uncertainty_range is not None
    
    def test_model_save_load(self, trained_lightgbm_model, temp_model_dir):
        """Test model persistence including quantile models."""
        trained_lightgbm_model.save(temp_model_dir)
        
        new_model = LightGBMLapTimeModel(trained_lightgbm_model.config)
        new_model.load(temp_model_dir)
        
        assert new_model.model is not None
        assert new_model.model_lower is not None
        assert new_model.model_upper is not None
    
    def test_categorical_features(self, lightgbm_model):
        """Test categorical feature handling."""
        assert isinstance(lightgbm_model.categorical_features, list)
