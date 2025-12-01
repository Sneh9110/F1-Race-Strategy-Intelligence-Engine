"""
Tests for data preparation pipeline.
"""

import pytest
import numpy as np
import pandas as pd

from models.lap_time.data_preparation import DataPreparationPipeline


class TestDataPreparationPipeline:
    """Test data preparation pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        pipeline = DataPreparationPipeline()
        assert pipeline is not None
        assert pipeline.config.min_samples == 1000
    
    def test_create_features(self, sample_training_data):
        """Test feature creation."""
        train_data, _, _, _ = sample_training_data
        pipeline = DataPreparationPipeline()
        
        # Create features
        features = pipeline.create_features(train_data)
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(train_data)
        assert 'tire_age' in features.columns
        assert 'fuel_load' in features.columns
    
    def test_normalize_features(self, sample_training_data):
        """Test feature normalization."""
        train_data, _, _, _ = sample_training_data
        pipeline = DataPreparationPipeline()
        
        # Normalize features
        X_train = train_data.drop('lap_time', axis=1)
        X_normalized = pipeline.normalize_features(X_train)
        
        assert isinstance(X_normalized, pd.DataFrame)
        assert X_normalized.shape == X_train.shape
        # Check normalization (mean ~0, std ~1)
        assert abs(X_normalized['fuel_load'].mean()) < 0.1
        assert abs(X_normalized['fuel_load'].std() - 1.0) < 0.1
    
    def test_data_quality_checks(self, sample_training_data):
        """Test data quality validation."""
        train_data, _, _, _ = sample_training_data
        pipeline = DataPreparationPipeline()
        
        # Should pass without errors
        pipeline._check_data_quality(train_data, "test_data")
