"""
Tests for model evaluation framework.
"""

import pytest
import numpy as np

from models.lap_time.evaluation import ModelEvaluator


class TestModelEvaluator:
    """Test evaluation metrics and analysis."""
    
    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        evaluator = ModelEvaluator()
        assert evaluator is not None
    
    def test_evaluate_model(self, trained_xgboost_model, sample_training_data):
        """Test comprehensive model evaluation."""
        evaluator = ModelEvaluator()
        train_data, val_data, test_data, _ = sample_training_data
        
        X_test = test_data.drop('lap_time', axis=1)
        y_test = test_data['lap_time']
        
        metrics = evaluator.evaluate_model(
            trained_xgboost_model,
            X_test,
            y_test,
            test_data
        )
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'accuracy_0.5s' in metrics
        assert metrics['rmse'] > 0
    
    def test_compare_models(self, trained_xgboost_model, trained_lightgbm_model, sample_training_data):
        """Test comparing multiple models."""
        evaluator = ModelEvaluator()
        _, _, test_data, _ = sample_training_data
        
        X_test = test_data.drop('lap_time', axis=1)
        y_test = test_data['lap_time']
        
        models = {
            'xgboost': trained_xgboost_model,
            'lightgbm': trained_lightgbm_model,
        }
        
        comparison = evaluator.compare_models(models, X_test, y_test, test_data)
        
        assert len(comparison) == 2
        assert 'mae' in comparison.columns
        assert 'rmse' in comparison.columns
    
    def test_analyze_errors(self, trained_xgboost_model, sample_training_data):
        """Test error analysis."""
        evaluator = ModelEvaluator()
        _, _, test_data, _ = sample_training_data
        
        X_test = test_data.drop('lap_time', axis=1)
        y_test = test_data['lap_time']
        
        analysis = evaluator.analyze_errors(
            trained_xgboost_model,
            X_test,
            y_test,
            test_data,
            top_n=5
        )
        
        assert len(analysis) == 5
        assert 'actual' in analysis.columns
        assert 'predicted' in analysis.columns
        assert 'error' in analysis.columns
