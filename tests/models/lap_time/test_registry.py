"""
Tests for model registry.
"""

import pytest
from pathlib import Path

from models.lap_time.registry import ModelRegistry


class TestModelRegistry:
    """Test model registry functionality."""
    
    def test_registry_initialization(self, temp_registry_dir):
        """Test registry can be initialized."""
        registry = ModelRegistry(temp_registry_dir)
        assert registry.registry_path == temp_registry_dir
        assert registry.metadata is not None
    
    def test_register_model(self, temp_registry_dir, trained_xgboost_model):
        """Test model registration."""
        registry = ModelRegistry(temp_registry_dir)
        
        model_info = registry.register_model(
            trained_xgboost_model,
            version="1.0.0",
            metrics={'rmse': 0.5, 'mae': 0.3},
            hyperparameters={'max_depth': 6},
            model_type='xgboost',
            description='Test model'
        )
        
        assert model_info['version'] == "1.0.0"
        assert model_info['model_type'] == 'xgboost'
        assert '1.0.0' in registry.metadata['versions']
    
    def test_load_model(self, temp_registry_dir, trained_xgboost_model):
        """Test loading registered model."""
        registry = ModelRegistry(temp_registry_dir)
        
        # Register model
        registry.register_model(
            trained_xgboost_model,
            version="1.0.0",
            metrics={},
            hyperparameters={},
            model_type='xgboost'
        )
        
        # Load model
        loaded_model = registry.load_model("1.0.0")
        assert loaded_model is not None
        assert loaded_model.config.version == "1.0.0"
    
    def test_promote_model(self, temp_registry_dir, trained_xgboost_model):
        """Test model promotion to alias."""
        registry = ModelRegistry(temp_registry_dir)
        
        registry.register_model(
            trained_xgboost_model,
            version="1.0.0",
            metrics={},
            hyperparameters={},
            model_type='xgboost'
        )
        
        registry.promote_model("1.0.0", "production")
        
        assert registry.metadata['aliases']['production'] == "1.0.0"
    
    def test_list_versions(self, temp_registry_dir, trained_xgboost_model):
        """Test listing all versions."""
        registry = ModelRegistry(temp_registry_dir)
        
        registry.register_model(
            trained_xgboost_model,
            version="1.0.0",
            metrics={},
            hyperparameters={},
            model_type='xgboost'
        )
        
        versions = registry.list_versions()
        assert len(versions) > 0
        assert versions[0]['version'] == "1.0.0"
