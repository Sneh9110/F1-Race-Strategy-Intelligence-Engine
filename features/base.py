"""
Base classes and abstractions for feature engineering.

This module provides the foundational architecture for all feature calculators,
including abstract base classes, configuration dataclasses, and common functionality
for validation, caching, and error handling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import hashlib
import json
import time

import pandas as pd
import numpy as np

from app.utils.logger import get_logger
from app.utils.validators import validate_dataframe, validate_numeric_range

logger = get_logger(__name__)


@dataclass
class FeatureConfig:
    """
    Configuration for feature calculation.
    
    Attributes:
        window_size: Rolling window size for time-series features
        thresholds: Dictionary of threshold values for validation
        cache_ttl: Cache time-to-live in seconds
        fallback_enabled: Whether to use fallback values on error
        min_data_points: Minimum data points required for computation
        outlier_removal: Whether to remove outliers before computation
        outlier_threshold: Z-score threshold for outlier detection
    """
    window_size: int = 5
    thresholds: Dict[str, float] = field(default_factory=dict)
    cache_ttl: int = 300
    fallback_enabled: bool = True
    min_data_points: int = 3
    outlier_removal: bool = False
    outlier_threshold: float = 3.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'window_size': self.window_size,
            'thresholds': self.thresholds,
            'cache_ttl': self.cache_ttl,
            'fallback_enabled': self.fallback_enabled,
            'min_data_points': self.min_data_points,
            'outlier_removal': self.outlier_removal,
            'outlier_threshold': self.outlier_threshold
        }


@dataclass
class FeatureResult:
    """
    Result of feature computation.
    
    Attributes:
        features: Computed features as DataFrame
        metadata: Feature metadata (computation time, version, etc.)
        warnings: List of warning messages
        errors: List of error messages
        computation_time_ms: Time taken for computation in milliseconds
    """
    features: pd.DataFrame
    metadata: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    computation_time_ms: float = 0.0
    
    @property
    def success(self) -> bool:
        """Check if computation was successful."""
        return len(self.errors) == 0 and not self.features.empty
    
    @property
    def has_warnings(self) -> bool:
        """Check if computation has warnings."""
        return len(self.warnings) > 0


class BaseFeature(ABC):
    """
    Abstract base class for all feature calculators.
    
    Provides common functionality for feature computation including:
    - Input validation
    - Error handling with fallback values
    - Logging with correlation IDs
    - Performance metrics tracking
    - Caching support
    
    Subclasses must implement:
    - _calculate(): Core feature calculation logic
    - _get_dependencies(): Feature dependencies for DAG
    """
    
    def __init__(
        self,
        name: str,
        version: str,
        config: Optional[FeatureConfig] = None,
        correlation_id: Optional[str] = None
    ):
        """
        Initialize feature calculator.
        
        Args:
            name: Feature name (e.g., 'stint_summary', 'degradation_slope')
            version: Feature version (semantic versioning: 'v1.0.0')
            config: Feature-specific configuration
            correlation_id: Optional correlation ID for request tracking
        """
        self.name = name
        self.version = version
        self.config = config or FeatureConfig()
        self.correlation_id = correlation_id or self._generate_correlation_id()
        self.logger = get_logger(f"{__name__}.{name}")
        
    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID for tracking."""
        timestamp = datetime.utcnow().isoformat()
        unique_str = f"{self.name}_{timestamp}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    def compute(self, data: Any, **kwargs) -> FeatureResult:
        """
        Compute features from input data.
        
        Main entry point for feature calculation. Handles:
        - Input validation
        - Error handling
        - Performance tracking
        - Logging
        
        Args:
            data: Input data (type depends on feature)
            **kwargs: Additional parameters for feature calculation
            
        Returns:
            FeatureResult with computed features and metadata
        """
        start_time = time.time()
        
        self.logger.info(
            f"Computing feature '{self.name}' v{self.version}",
            extra={
                'correlation_id': self.correlation_id,
                'feature_name': self.name,
                'feature_version': self.version
            }
        )
        
        warnings = []
        errors = []
        
        try:
            # Validate input
            validation_result = self._validate_input(data)
            if not validation_result['valid']:
                errors.extend(validation_result['errors'])
                if not self.config.fallback_enabled:
                    raise ValueError(f"Input validation failed: {validation_result['errors']}")
                warnings.append("Input validation failed, using fallback")
            
            # Calculate features
            features_df = self._calculate(data, **kwargs)
            
            # Validate output
            output_validation = self.validate(features_df)
            if not output_validation['valid']:
                warnings.extend(output_validation['warnings'])
                if output_validation['errors']:
                    errors.extend(output_validation['errors'])
            
            # Build metadata
            computation_time_ms = (time.time() - start_time) * 1000
            metadata = self._build_metadata(features_df, computation_time_ms)
            
            self.logger.info(
                f"Feature '{self.name}' computed successfully",
                extra={
                    'correlation_id': self.correlation_id,
                    'feature_name': self.name,
                    'computation_time_ms': computation_time_ms,
                    'num_records': len(features_df)
                }
            )
            
            return FeatureResult(
                features=features_df,
                metadata=metadata,
                warnings=warnings,
                errors=errors,
                computation_time_ms=computation_time_ms
            )
            
        except Exception as e:
            computation_time_ms = (time.time() - start_time) * 1000
            error_msg = f"Error computing feature '{self.name}': {str(e)}"
            errors.append(error_msg)
            
            self.logger.error(
                error_msg,
                extra={
                    'correlation_id': self.correlation_id,
                    'feature_name': self.name,
                    'error': str(e)
                },
                exc_info=True
            )
            
            # Return fallback if enabled
            if self.config.fallback_enabled:
                fallback_df = self._get_fallback_features()
                warnings.append(f"Using fallback features due to error: {str(e)}")
                return FeatureResult(
                    features=fallback_df,
                    metadata={'fallback': True, 'error': str(e)},
                    warnings=warnings,
                    errors=errors,
                    computation_time_ms=computation_time_ms
                )
            else:
                # Return empty result
                return FeatureResult(
                    features=pd.DataFrame(),
                    metadata={},
                    warnings=warnings,
                    errors=errors,
                    computation_time_ms=computation_time_ms
                )
    
    @abstractmethod
    def _calculate(self, data: Any, **kwargs) -> pd.DataFrame:
        """
        Core feature calculation logic.
        
        Must be implemented by subclasses.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with computed features
        """
        pass
    
    @abstractmethod
    def _get_dependencies(self) -> List[str]:
        """
        Get list of feature dependencies.
        
        Returns:
            List of feature names that this feature depends on
        """
        pass
    
    def _validate_input(self, data: Any) -> Dict[str, Any]:
        """
        Validate input data.
        
        Args:
            data: Input data to validate
            
        Returns:
            Dictionary with validation result
        """
        errors = []
        
        # Check if data is None
        if data is None:
            errors.append("Input data is None")
            return {'valid': False, 'errors': errors}
        
        # Check if DataFrame
        if isinstance(data, pd.DataFrame):
            if data.empty:
                errors.append("Input DataFrame is empty")
            if len(data) < self.config.min_data_points:
                errors.append(
                    f"Insufficient data points: {len(data)} < {self.config.min_data_points}"
                )
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def validate(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate computed features.
        
        Args:
            features: Computed features DataFrame
            
        Returns:
            Dictionary with validation result
        """
        warnings = []
        errors = []
        
        # Check if empty
        if features.empty:
            errors.append("Computed features DataFrame is empty")
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        # Check for NaN values
        nan_counts = features.isnull().sum()
        if nan_counts.sum() > 0:
            warnings.append(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
        
        # Check for infinite values
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(features[col]).any():
                warnings.append(f"Infinite values found in column '{col}'")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get feature metadata.
        
        Returns:
            Dictionary with feature information
        """
        return {
            'name': self.name,
            'version': self.version,
            'dependencies': self._get_dependencies(),
            'config': self.config.to_dict(),
            'correlation_id': self.correlation_id
        }
    
    def cache_key(self, inputs: Dict[str, Any]) -> str:
        """
        Generate cache key for feature computation.
        
        Args:
            inputs: Dictionary of input parameters
            
        Returns:
            Cache key string
        """
        # Sort inputs for consistent hashing
        sorted_inputs = json.dumps(inputs, sort_keys=True)
        hash_str = f"{self.name}_{self.version}_{sorted_inputs}"
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    def _build_metadata(
        self,
        features_df: pd.DataFrame,
        computation_time_ms: float
    ) -> Dict[str, Any]:
        """
        Build metadata for computed features.
        
        Args:
            features_df: Computed features DataFrame
            computation_time_ms: Computation time in milliseconds
            
        Returns:
            Metadata dictionary
        """
        metadata = self.get_metadata()
        metadata.update({
            'computation_timestamp': datetime.utcnow().isoformat(),
            'computation_time_ms': computation_time_ms,
            'num_records': len(features_df),
            'columns': list(features_df.columns),
            'statistics': self._calculate_statistics(features_df)
        })
        return metadata
    
    def _calculate_statistics(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistics for features.
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats[col] = {
                'min': float(features_df[col].min()),
                'max': float(features_df[col].max()),
                'mean': float(features_df[col].mean()),
                'std': float(features_df[col].std()),
                'null_count': int(features_df[col].isnull().sum())
            }
        
        return stats
    
    def _get_fallback_features(self) -> pd.DataFrame:
        """
        Get fallback features when computation fails.
        
        Returns:
            Empty DataFrame with expected schema
        """
        return pd.DataFrame()
    
    def _remove_outliers(self, data: pd.Series) -> pd.Series:
        """
        Remove outliers using z-score method.
        
        Args:
            data: Series with numeric data
            
        Returns:
            Series with outliers removed (replaced with NaN)
        """
        if not self.config.outlier_removal:
            return data
        
        z_scores = np.abs((data - data.mean()) / data.std())
        return data.mask(z_scores > self.config.outlier_threshold)
