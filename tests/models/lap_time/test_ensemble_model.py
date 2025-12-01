"""
Tests for ensemble lap time model.
"""

import pytest
import numpy as np

from models.lap_time.ensemble_model import EnsembleLapTimeModel


class TestEnsembleLapTimeModel:
    """Test ensemble model implementation."""
    
    def test_model_initialization(self, ensemble_model):
        """Test ensemble initialization."""
        assert ensemble_model is not None
        assert ensemble_model.xgboost_model is not None
        assert ensemble_model.lightgbm_model is not None
        assert len(ensemble_model.weights) == 2
    
    def test_ensemble_training(self, ensemble_model, sample_training_data):
        """Test ensemble training with weight optimization."""
        train_data, val_data, _, _ = sample_training_data
        X_train = train_data.drop('lap_time', axis=1)
        y_train = train_data['lap_time']
        X_val = val_data.drop('lap_time', axis=1)
        y_val = val_data['lap_time']
        
        metrics = ensemble_model.train(X_train, y_train, X_val, y_val, n_estimators=10)
        
        assert ensemble_model.xgboost_model.model is not None
        assert ensemble_model.lightgbm_model.model is not None
        assert ensemble_model.optimal_weights_found is True
        assert 'ensemble_weights' in metrics
    
    def test_ensemble_prediction(self, trained_ensemble_model, sample_prediction_input):
        """Test ensemble combines both models."""
        output = trained_ensemble_model.predict(sample_prediction_input)
        
        assert output.predicted_lap_time > 60.0
        assert 'xgb_prediction' in output.metadata
        assert 'lgb_prediction' in output.metadata
        assert 'prediction_agreement' in output.metadata
    
    def test_weight_optimization(self, trained_ensemble_model):
        """Test weights are optimized."""
        weights = trained_ensemble_model.weights
        assert len(weights) == 2
        assert abs(sum(weights) - 1.0) < 0.01  # Weights sum to 1
    
    def test_ensemble_save_load(self, trained_ensemble_model, temp_model_dir):
        """Test saving both base models."""
        trained_ensemble_model.save(temp_model_dir)
        
        new_model = EnsembleLapTimeModel(trained_ensemble_model.config)
        new_model.load(temp_model_dir)
        
        assert new_model.xgboost_model.model is not None
        assert new_model.lightgbm_model.model is not None
        assert new_model.weights == trained_ensemble_model.weights
