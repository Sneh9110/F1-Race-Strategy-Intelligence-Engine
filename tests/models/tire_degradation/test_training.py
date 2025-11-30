"""Tests for model training orchestrator."""

import pytest

from models.tire_degradation.training import ModelTrainer


def test_trainer_initialization():
    """Test trainer initialization."""
    trainer = ModelTrainer(model_type='xgboost')
    assert trainer.model_type == 'xgboost'
    assert trainer.data_pipeline is not None


def test_trainer_train(sample_training_data):
    """Test model training via trainer."""
    trainer = ModelTrainer(model_type='xgboost')
    
    model = trainer.train(
        data=sample_training_data,
        target_column='degradation_rate',
        optimize_hyperparams=False
    )
    
    assert model is not None
    assert model.model is not None
