"""Tests for data preparation pipeline."""

import pytest
import pandas as pd

from models.tire_degradation.data_preparation import DataPreparationPipeline


def test_pipeline_initialization():
    """Test pipeline initialization."""
    pipeline = DataPreparationPipeline()
    assert pipeline is not None


def test_prepare_training_data(sample_training_data):
    """Test data preparation."""
    pipeline = DataPreparationPipeline()
    
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_training_data(
        sample_training_data,
        target_column='degradation_rate'
    )
    
    # Check splits exist
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    
    # Check target alignment
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)


def test_extract_features_from_stint():
    """Test feature extraction from stint."""
    pipeline = DataPreparationPipeline()
    
    # Create mock stint data
    stint_data = {
        'lap_times': [90.0, 90.5, 91.0, 91.5, 92.0],
        'tire_compound': 'MEDIUM',
        'tire_age': 10,
        'track_name': 'Monza'
    }
    
    features = pipeline.extract_features_from_stint(stint_data)
    
    assert 'avg_lap_time' in features
    assert 'lap_time_std' in features
    assert features['avg_lap_time'] > 0
