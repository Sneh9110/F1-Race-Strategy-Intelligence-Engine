"""Tests for model registry."""

import pytest

from models.tire_degradation.registry import ModelRegistry


def test_registry_initialization(temp_model_dir):
    """Test registry initialization."""
    registry = ModelRegistry(registry_dir=temp_model_dir)
    assert registry.registry_dir.exists()
    assert registry.metadata is not None


def test_register_model(temp_model_dir, trained_xgboost_model):
    """Test model registration."""
    registry = ModelRegistry(registry_dir=temp_model_dir)
    
    version = registry.register_model(
        model=trained_xgboost_model,
        version='1.0.0',
        alias='test'
    )
    
    assert version == '1.0.0'
    assert '1.0.0' in registry.metadata['models']
    assert registry.metadata['aliases']['test'] == '1.0.0'


def test_load_model(temp_model_dir, trained_xgboost_model):
    """Test model loading."""
    registry = ModelRegistry(registry_dir=temp_model_dir)
    registry.register_model(trained_xgboost_model, version='1.0.0')
    
    loaded_model = registry.load_model('1.0.0')
    assert loaded_model is not None
    assert loaded_model.model is not None
