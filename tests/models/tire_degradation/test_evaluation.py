"""Tests for model evaluation."""

import pytest
import numpy as np

from models.tire_degradation.evaluation import ModelEvaluator


def test_evaluator_initialization():
    """Test evaluator initialization."""
    evaluator = ModelEvaluator()
    assert evaluator is not None


def test_evaluate_model(trained_xgboost_model, sample_training_data):
    """Test model evaluation."""
    evaluator = ModelEvaluator()
    
    X = sample_training_data.drop(columns=['degradation_rate', 'usable_life', 'has_cliff'])
    y = sample_training_data['degradation_rate']
    
    metrics = evaluator.evaluate(trained_xgboost_model, X, y)
    
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'r2' in metrics
    assert metrics['mae'] >= 0
    assert metrics['rmse'] >= 0


def test_dtw_distance():
    """Test DTW distance calculation."""
    evaluator = ModelEvaluator()
    
    series1 = [1.0, 2.0, 3.0, 4.0]
    series2 = [1.0, 2.0, 3.0, 4.0]
    
    distance = evaluator._dtw_distance(series1, series2)
    assert distance == 0.0  # Identical series
