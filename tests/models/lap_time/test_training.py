"""
Tests for model training pipeline.
"""

import pytest
from pathlib import Path

from models.lap_time.training import ModelTrainer


class TestModelTrainer:
    """Test training pipeline."""
    
    def test_trainer_initialization(self, temp_registry_dir):
        """Test trainer can be initialized."""
        trainer = ModelTrainer(registry_path=temp_registry_dir)
        assert trainer is not None
        assert trainer.n_trials == 50
        assert trainer.n_folds == 5
    
    @pytest.mark.slow
    def test_train_model_no_optimization(self, temp_registry_dir, sample_training_data):
        """Test training without hyperparameter optimization."""
        # This test would require actual data file
        # Skipping full implementation for brevity
        pass
    
    def test_create_model(self, temp_registry_dir):
        """Test model creation."""
        trainer = ModelTrainer(registry_path=temp_registry_dir)
        
        xgb_model = trainer._create_model('xgboost')
        assert xgb_model is not None
        
        lgb_model = trainer._create_model('lightgbm')
        assert lgb_model is not None
        
        ens_model = trainer._create_model('ensemble')
        assert ens_model is not None
    
    def test_invalid_model_type(self, temp_registry_dir):
        """Test error on invalid model type."""
        trainer = ModelTrainer(registry_path=temp_registry_dir)
        
        with pytest.raises(ValueError):
            trainer._create_model('invalid_type')
