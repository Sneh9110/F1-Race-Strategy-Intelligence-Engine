"""
Model registry for versioning, storage, and deployment management.

Handles model versioning, metadata tracking, and A/B testing.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import semver

from models.tire_degradation.base import BaseDegradationModel, ModelConfig
from models.tire_degradation.xgboost_model import XGBoostDegradationModel
from models.tire_degradation.lightgbm_model import LightGBMDegradationModel
from models.tire_degradation.ensemble_model import EnsembleDegradationModel
from app.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ModelRegistry:
    """
    Model registry for version control and deployment.
    
    Features:
    - Semantic versioning
    - Model metadata tracking
    - A/B testing support
    - Easy model promotion (dev -> staging -> prod)
    """
    
    def __init__(self, registry_dir: Optional[Path] = None):
        """
        Initialize model registry.
        
        Args:
            registry_dir: Directory for model storage
        """
        self.registry_dir = registry_dir or Path(settings.BASE_DIR) / 'models' / 'saved' / 'tire_degradation'
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.registry_dir / 'registry_metadata.json'
        self.metadata = self._load_metadata()
        
        # Model class mapping
        self.model_classes = {
            'xgboost': XGBoostDegradationModel,
            'lightgbm': LightGBMDegradationModel,
            'ensemble': EnsembleDegradationModel
        }
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load registry metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        
        return {
            'models': {},
            'aliases': {
                'latest': None,
                'production': None,
                'staging': None
            }
        }
    
    def _save_metadata(self) -> None:
        """Save registry metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(
        self,
        model: BaseDegradationModel,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
        alias: Optional[str] = None
    ) -> str:
        """
        Register a new model version.
        
        Args:
            model: Trained model to register
            version: Semantic version (e.g., '1.0.0')
            metadata: Additional metadata
            alias: Optional alias ('production', 'staging', etc.)
        
        Returns:
            Model ID
        """
        # Validate semantic version
        try:
            semver.VersionInfo.parse(version)
        except ValueError:
            raise ValueError(f"Invalid semantic version: {version}")
        
        # Check if version exists
        if version in self.metadata['models']:
            logger.warning(f"Version {version} already exists. Overwriting.")
        
        # Create model directory
        model_dir = self.registry_dir / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save(model_dir)
        
        # Store metadata
        model_metadata = {
            'version': version,
            'model_type': model.__class__.__name__,
            'registered_at': datetime.now().isoformat(),
            'model_config': model.config.to_dict(),
            'performance_metrics': model.metadata.get('performance_metrics', {}),
            'path': str(model_dir)
        }
        
        if metadata:
            model_metadata.update(metadata)
        
        self.metadata['models'][version] = model_metadata
        
        # Set alias if provided
        if alias:
            self.set_alias(alias, version)
        
        # Update 'latest' alias
        self.set_alias('latest', version)
        
        self._save_metadata()
        
        logger.info(f"Model registered: version={version}, alias={alias}")
        return version
    
    def load_model(
        self,
        version: str = 'latest',
        alias: Optional[str] = None
    ) -> BaseDegradationModel:
        """
        Load a model by version or alias.
        
        Args:
            version: Model version or alias
            alias: Alternative alias lookup
        
        Returns:
            Loaded model
        """
        # Resolve alias
        if alias:
            version = self.metadata['aliases'].get(alias, version)
        elif version in self.metadata['aliases']:
            version = self.metadata['aliases'][version]
        
        # Get model metadata
        if version not in self.metadata['models']:
            raise ValueError(f"Model version '{version}' not found in registry")
        
        model_meta = self.metadata['models'][version]
        model_dir = Path(model_meta['path'])
        
        # Determine model type
        model_type = model_meta['model_type']
        if model_type not in self.model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load config
        config_path = model_dir / 'config.json'
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)
        
        # Instantiate and load model
        model_class = self.model_classes[model_type]
        model = model_class(config)
        model.load(model_dir)
        
        logger.info(f"Model loaded: version={version}, type={model_type}")
        return model
    
    def set_alias(self, alias: str, version: str) -> None:
        """
        Set an alias to a specific version.
        
        Args:
            alias: Alias name ('production', 'staging', 'latest', etc.)
            version: Model version
        """
        if version not in self.metadata['models']:
            raise ValueError(f"Model version '{version}' not found")
        
        self.metadata['aliases'][alias] = version
        self._save_metadata()
        
        logger.info(f"Alias '{alias}' set to version '{version}'")
    
    def get_alias(self, alias: str) -> Optional[str]:
        """
        Get version for an alias.
        
        Args:
            alias: Alias name
        
        Returns:
            Version or None
        """
        return self.metadata['aliases'].get(alias)
    
    def list_models(self, sort_by: str = 'registered_at') -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Args:
            sort_by: Field to sort by
        
        Returns:
            List of model metadata
        """
        models = list(self.metadata['models'].values())
        
        if sort_by in ['registered_at', 'version']:
            models.sort(key=lambda x: x.get(sort_by, ''), reverse=True)
        
        return models
    
    def get_model_info(self, version: str) -> Dict[str, Any]:
        """
        Get metadata for a specific model version.
        
        Args:
            version: Model version
        
        Returns:
            Model metadata
        """
        if version in self.metadata['aliases']:
            version = self.metadata['aliases'][version]
        
        if version not in self.metadata['models']:
            raise ValueError(f"Model version '{version}' not found")
        
        return self.metadata['models'][version]
    
    def delete_model(self, version: str, force: bool = False) -> None:
        """
        Delete a model version.
        
        Args:
            version: Model version to delete
            force: Force deletion even if aliased
        """
        if version not in self.metadata['models']:
            raise ValueError(f"Model version '{version}' not found")
        
        # Check if model is aliased
        aliased = [alias for alias, v in self.metadata['aliases'].items() if v == version]
        if aliased and not force:
            raise ValueError(f"Model is aliased as {aliased}. Use force=True to delete.")
        
        # Remove from disk
        model_dir = Path(self.metadata['models'][version]['path'])
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from metadata
        del self.metadata['models'][version]
        
        # Clear aliases
        for alias in aliased:
            self.metadata['aliases'][alias] = None
        
        self._save_metadata()
        
        logger.info(f"Model version '{version}' deleted")
    
    def promote_model(self, version: str, from_env: str, to_env: str) -> None:
        """
        Promote model between environments.
        
        Args:
            version: Model version
            from_env: Source environment ('staging')
            to_env: Target environment ('production')
        """
        if version not in self.metadata['models']:
            raise ValueError(f"Model version '{version}' not found")
        
        # Verify source
        if self.metadata['aliases'].get(from_env) != version:
            logger.warning(f"Version {version} is not in {from_env}")
        
        # Promote
        self.set_alias(to_env, version)
        
        logger.info(f"Model {version} promoted from {from_env} to {to_env}")
    
    def compare_models(
        self,
        version1: str,
        version2: str,
        metric: str = 'rmse'
    ) -> Dict[str, Any]:
        """
        Compare performance metrics of two models.
        
        Args:
            version1: First model version
            version2: Second model version
            metric: Metric to compare
        
        Returns:
            Comparison results
        """
        info1 = self.get_model_info(version1)
        info2 = self.get_model_info(version2)
        
        metrics1 = info1.get('performance_metrics', {})
        metrics2 = info2.get('performance_metrics', {})
        
        comparison = {
            'version1': {
                'version': version1,
                'metrics': metrics1
            },
            'version2': {
                'version': version2,
                'metrics': metrics2
            }
        }
        
        # Compare specific metric
        if metric in metrics1 and metric in metrics2:
            val1 = metrics1[metric]
            val2 = metrics2[metric]
            comparison['winner'] = version1 if val1 < val2 else version2
            comparison['difference'] = abs(val1 - val2)
            comparison['improvement_pct'] = ((val1 - val2) / val1) * 100
        
        return comparison
