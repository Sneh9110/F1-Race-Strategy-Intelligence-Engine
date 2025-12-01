"""
Tests for XGBoost lap time model.
"""

import pytest
import numpy as np

from models.lap_time.xgboost_model import XGBoostLapTimeModel


class TestXGBoostLapTimeModel:
    """Test XGBoost model implementation."""
    
    def test_model_initialization(self, xgboost_model):
        """Test model can be initialized."""
        assert xgboost_model is not None
        assert xgboost_model.model is None  # Not trained yet
    
    def test_model_training(self, xgboost_model, sample_training_data):
        """Test model training."""
        train_data, val_data, _, _ = sample_training_data
        X_train = train_data.drop('lap_time', axis=1)
        y_train = train_data['lap_time']
        X_val = val_data.drop('lap_time', axis=1)
        y_val = val_data['lap_time']
        
        metrics = xgboost_model.train(X_train, y_train, X_val, y_val, n_estimators=10)
        
        assert xgboost_model.model is not None
        assert 'train_rmse' in metrics
        assert 'val_rmse' in metrics
        assert metrics['train_rmse'] > 0
    
    def test_single_prediction(self, trained_xgboost_model, sample_prediction_input):
        """Test single prediction."""
        output = trained_xgboost_model.predict(sample_prediction_input)
        
        assert output.predicted_lap_time > 60.0
        assert output.predicted_lap_time < 150.0
        assert 0.0 <= output.confidence <= 1.0
        assert 'base_pace' in output.pace_components
    
    def test_batch_prediction(self, trained_xgboost_model, sample_prediction_inputs):
        """Test batch predictions."""
        outputs = trained_xgboost_model.predict_batch(sample_prediction_inputs)
        
        assert len(outputs) == len(sample_prediction_inputs)
        for output in outputs:
            assert 60.0 < output.predicted_lap_time < 150.0
    
    def test_model_save_load(self, trained_xgboost_model, temp_model_dir):
        """Test model persistence."""
        # Save model
        trained_xgboost_model.save(temp_model_dir)
        
        # Create new model and load
        new_model = XGBoostLapTimeModel(trained_xgboost_model.config)
        new_model.load(temp_model_dir)
        
        assert new_model.model is not None
        assert new_model.feature_names == trained_xgboost_model.feature_names
    
    def test_get_metadata(self, trained_xgboost_model):
        """Test metadata retrieval."""
        metadata = trained_xgboost_model.get_metadata()
        
        assert metadata['model_type'] == 'xgboost'
        assert metadata['trained'] is True
        assert 'feature_count' in metadata
