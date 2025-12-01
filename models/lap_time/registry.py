"""
Model Registry for Version Management

Manages model versions, metadata, and deployment.
Supports semantic versioning and A/B testing.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import shutil
from datetime import datetime

from .base import BaseLapTimeModel, ModelConfig
from .xgboost_model import XGBoostLapTimeModel
from .lightgbm_model import LightGBMLapTimeModel
from .ensemble_model import EnsembleLapTimeModel
from config.settings import Settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing model versions.
    
    Features:
    - Semantic versioning (major.minor.patch)
    - Model aliases (latest, production, staging)
    - Metadata tracking (metrics, hyperparameters, training date)
    - Model promotion workflow
    - A/B testing support
    
    Attributes:
        registry_path: Path to registry directory
        metadata: Registry metadata
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = registry_path or Path("models/registry/lap_time")
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.registry_path / "registry.json"
        self.metadata = self._load_metadata()
    
    def register_model(
        self,
        model: BaseLapTimeModel,
        version: str,
        metrics: Dict[str, float],
        hyperparameters: Dict,
        model_type: str,
        description: Optional[str] = None
    ) -> Dict:
        """
        Register a new model version.
        
        Args:
            model: Trained model instance
            version: Version string (e.g., "1.0.0")
            metrics: Training metrics
            hyperparameters: Model hyperparameters
            model_type: Type of model (xgboost, lightgbm, ensemble)
            description: Optional description
            
        Returns:
            Model information dictionary
        """
        logger.info(f"Registering model version {version}")
        
        # Validate version format
        self._validate_version(version)
        
        # Create version directory
        version_path = self.registry_path / version
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save(version_path)
        
        # Create model metadata
        model_info = {
            'version': version,
            'model_type': model_type,
            'registered_at': datetime.now().isoformat(),
            'metrics': metrics,
            'hyperparameters': hyperparameters,
            'description': description or f"{model_type} model v{version}",
            'path': str(version_path),
        }
        
        # Update registry metadata
        if 'versions' not in self.metadata:
            self.metadata['versions'] = {}
        
        self.metadata['versions'][version] = model_info
        
        # Update 'latest' alias
        self.metadata['aliases'] = self.metadata.get('aliases', {})
        self.metadata['aliases']['latest'] = version
        
        # Save metadata
        self._save_metadata()
        
        logger.info(f"Model {version} registered successfully")
        return model_info
    
    def load_model(self, version: str = "latest") -> BaseLapTimeModel:
        """
        Load model by version or alias.
        
        Args:
            version: Version string, or alias ('latest', 'production', 'staging')
            
        Returns:
            Loaded model instance
        """
        # Resolve alias if needed
        resolved_version = self._resolve_version(version)
        
        if resolved_version not in self.metadata.get('versions', {}):
            raise ValueError(f"Model version {resolved_version} not found in registry")
        
        model_info = self.metadata['versions'][resolved_version]
        model_path = Path(model_info['path'])
        model_type = model_info['model_type']
        
        # Create model instance
        config = ModelConfig(
            name=f"lap_time_{model_type}",
            version=resolved_version,
            model_type=model_type
        )
        
        if model_type == "xgboost":
            model = XGBoostLapTimeModel(config)
        elif model_type == "lightgbm":
            model = LightGBMLapTimeModel(config)
        elif model_type == "ensemble":
            model = EnsembleLapTimeModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load model weights
        model.load(model_path)
        
        logger.info(f"Loaded model {resolved_version} ({model_type})")
        return model
    
    def promote_model(self, version: str, alias: str = "production") -> None:
        """
        Promote model version to alias (e.g., production, staging).
        
        Args:
            version: Version to promote
            alias: Target alias
        """
        resolved_version = self._resolve_version(version)
        
        if resolved_version not in self.metadata.get('versions', {}):
            raise ValueError(f"Model version {resolved_version} not found")
        
        # Update alias
        self.metadata['aliases'][alias] = resolved_version
        self._save_metadata()
        
        logger.info(f"Promoted model {resolved_version} to {alias}")
    
    def list_versions(self) -> List[Dict]:
        """
        List all registered model versions.
        
        Returns:
            List of model information dictionaries
        """
        versions = []
        for version, info in self.metadata.get('versions', {}).items():
            # Add alias information
            aliases = [
                alias for alias, ver in self.metadata.get('aliases', {}).items()
                if ver == version
            ]
            info_copy = info.copy()
            info_copy['aliases'] = aliases
            versions.append(info_copy)
        
        # Sort by registration date (newest first)
        versions.sort(key=lambda x: x.get('registered_at', ''), reverse=True)
        return versions
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """
        Compare metrics between two model versions.
        
        Args:
            version1: First version
            version2: Second version
            
        Returns:
            Comparison dictionary
        """
        v1 = self._resolve_version(version1)
        v2 = self._resolve_version(version2)
        
        if v1 not in self.metadata.get('versions', {}):
            raise ValueError(f"Version {v1} not found")
        if v2 not in self.metadata.get('versions', {}):
            raise ValueError(f"Version {v2} not found")
        
        info1 = self.metadata['versions'][v1]
        info2 = self.metadata['versions'][v2]
        
        comparison = {
            'version1': v1,
            'version2': v2,
            'metrics_v1': info1.get('metrics', {}),
            'metrics_v2': info2.get('metrics', {}),
            'improvements': {},
        }
        
        # Calculate improvements
        for metric in info1.get('metrics', {}):
            if metric in info2.get('metrics', {}):
                val1 = info1['metrics'][metric]
                val2 = info2['metrics'][metric]
                
                # For error metrics (RMSE, MAE), negative change is improvement
                if metric.lower() in ['rmse', 'mae', 'mape']:
                    improvement = (val1 - val2) / val1 if val1 != 0 else 0
                else:  # For R², accuracy, positive change is improvement
                    improvement = (val2 - val1) / abs(val1) if val1 != 0 else 0
                
                comparison['improvements'][metric] = improvement
        
        return comparison
    
    def delete_version(self, version: str, force: bool = False) -> None:
        """
        Delete model version from registry.
        
        Args:
            version: Version to delete
            force: Whether to force deletion even if it has aliases
        """
        resolved_version = self._resolve_version(version)
        
        if resolved_version not in self.metadata.get('versions', {}):
            raise ValueError(f"Version {resolved_version} not found")
        
        # Check if version has aliases
        aliases = [
            alias for alias, ver in self.metadata.get('aliases', {}).items()
            if ver == resolved_version
        ]
        
        if aliases and not force:
            raise ValueError(
                f"Cannot delete version {resolved_version} - it has aliases: {aliases}. "
                "Use force=True to delete anyway."
            )
        
        # Remove model files
        model_path = Path(self.metadata['versions'][resolved_version]['path'])
        if model_path.exists():
            shutil.rmtree(model_path)
        
        # Remove from metadata
        del self.metadata['versions'][resolved_version]
        
        # Remove aliases
        for alias in aliases:
            del self.metadata['aliases'][alias]
        
        self._save_metadata()
        
        logger.info(f"Deleted model version {resolved_version}")
    
    def _resolve_version(self, version: str) -> str:
        """
        Resolve version alias to actual version.
        
        Args:
            version: Version or alias
            
        Returns:
            Actual version string
        """
        # Check if it's an alias
        if version in self.metadata.get('aliases', {}):
            return self.metadata['aliases'][version]
        
        # Otherwise return as-is
        return version
    
    def _validate_version(self, version: str) -> None:
        """
        Validate version string format.
        
        Args:
            version: Version string
            
        Raises:
            ValueError: If version format is invalid
        """
        # Allow semantic versioning (e.g., "1.0.0") or timestamped (e.g., "1.0.20240115_143022")
        parts = version.split('.')
        if len(parts) < 2:
            raise ValueError(f"Invalid version format: {version}. Expected format: major.minor[.patch]")
    
    def _load_metadata(self) -> Dict:
        """
        Load registry metadata from disk.
        
        Returns:
            Metadata dictionary
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load registry metadata: {e}")
        
        # Return empty metadata
        return {
            'versions': {},
            'aliases': {},
            'created_at': datetime.now().isoformat(),
        }
    
    def _save_metadata(self) -> None:
        """Save registry metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry metadata: {e}")
