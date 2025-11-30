"""
Base classes and abstractions for tire degradation models.

Defines the common interface that all degradation models must implement,
along with configuration and I/O data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, validator

from app.utils.validators import validate_numeric_range
from config.settings import settings


class TireCompound(str, Enum):
    """Tire compound types."""
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"


@dataclass
class ModelConfig:
    """
    Configuration for degradation model training and inference.
    
    Attributes:
        model_type: Type of model (xgboost, lightgbm, ensemble)
        hyperparameters: Model-specific hyperparameters
        training_config: Training settings (epochs, batch_size, etc.)
        inference_config: Inference settings (batch_size, cache_ttl, etc.)
        fallback_config: Fallback behavior settings
        version: Model version (semantic versioning)
    """
    model_type: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    inference_config: Dict[str, Any] = field(default_factory=dict)
    fallback_config: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'training_config': self.training_config,
            'inference_config': self.inference_config,
            'fallback_config': self.fallback_config,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**data)


class PredictionInput(BaseModel):
    """
    Input data for tire degradation prediction.
    
    All inputs are validated to ensure they are within reasonable ranges
    for F1 racing scenarios.
    """
    tire_compound: TireCompound = Field(..., description="Tire compound type")
    tire_age: int = Field(..., ge=0, le=50, description="Current tire age in laps")
    stint_history: List[float] = Field(
        default_factory=list,
        description="Historical lap times for current stint"
    )
    weather_temp: float = Field(
        ...,
        ge=-10.0,
        le=60.0,
        description="Track or air temperature in Celsius"
    )
    driver_aggression: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Driver aggression score (0=conservative, 1=aggressive)"
    )
    track_name: str = Field(..., description="Circuit/track identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")
    
    # Optional contextual features
    track_temp: Optional[float] = Field(
        None,
        ge=-10.0,
        le=80.0,
        description="Track temperature (if different from weather_temp)"
    )
    fuel_load: Optional[float] = Field(
        None,
        ge=0.0,
        le=150.0,
        description="Current fuel load in kg"
    )
    stint_number: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="Stint number in current session"
    )
    
    @validator('stint_history')
    def validate_stint_history(cls, v):
        """Validate stint history lap times are reasonable."""
        if v:
            if any(t < 50.0 or t > 200.0 for t in v):
                raise ValueError("Lap times must be between 50.0 and 200.0 seconds")
        return v
    
    class Config:
        use_enum_values = True


class PredictionOutput(BaseModel):
    """
    Output from tire degradation prediction.
    
    Includes degradation curve, usable life, dropoff prediction,
    and confidence metrics.
    """
    degradation_curve: List[float] = Field(
        ...,
        description="Lap-by-lap degradation prediction (seconds added per lap)"
    )
    usable_life: int = Field(
        ...,
        ge=0,
        description="Predicted usable tire life in laps"
    )
    dropoff_lap: Optional[int] = Field(
        None,
        description="Predicted lap where tire performance cliff occurs"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (model version, computation time, etc.)"
    )
    
    # Optional detailed outputs
    degradation_rate: Optional[float] = Field(
        None,
        description="Average degradation rate (seconds per lap)"
    )
    confidence_intervals: Optional[Dict[str, List[float]]] = Field(
        None,
        description="Confidence intervals for curve (lower, upper bounds)"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        None,
        description="Feature importance scores"
    )
    
    @validator('degradation_curve')
    def validate_curve(cls, v):
        """Validate degradation curve values are reasonable."""
        if v:
            if len(v) > 100:
                raise ValueError("Degradation curve cannot exceed 100 laps")
            if any(d < -5.0 or d > 10.0 for d in v):
                raise ValueError("Degradation values must be between -5.0 and 10.0 seconds")
        return v


class BaseDegradationModel(ABC):
    """
    Abstract base class for tire degradation prediction models.
    
    All degradation models must implement this interface to ensure
    consistency and interoperability.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize model with configuration.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.metadata = {
            'model_type': config.model_type,
            'version': config.version,
            'trained': False,
            'training_date': None,
            'feature_names': [],
            'performance_metrics': {}
        }
    
    @abstractmethod
    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model on provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training arguments
        
        Returns:
            Dictionary with training metrics and history
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Make prediction for single input.
        
        Args:
            input_data: Input features
        
        Returns:
            Prediction output with degradation curve and metadata
        """
        pass
    
    @abstractmethod
    def predict_curve(
        self,
        input_data: PredictionInput,
        num_laps: int = 50
    ) -> List[float]:
        """
        Predict degradation curve for specified number of laps.
        
        Args:
            input_data: Input features
            num_laps: Number of laps to predict
        
        Returns:
            List of degradation values (seconds per lap)
        """
        pass
    
    @abstractmethod
    def predict_usable_life(self, input_data: PredictionInput) -> int:
        """
        Predict usable tire life in laps.
        
        Args:
            input_data: Input features
        
        Returns:
            Predicted usable life in laps
        """
        pass
    
    @abstractmethod
    def predict_dropoff_lap(self, input_data: PredictionInput) -> Optional[int]:
        """
        Predict lap where tire performance cliff/dropoff occurs.
        
        Args:
            input_data: Input features
        
        Returns:
            Predicted dropoff lap, or None if no cliff expected
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load model from disk.
        
        Args:
            path: Directory path to load model from
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary with model metadata
        """
        return self.metadata.copy()
    
    def _validate_input(self, input_data: PredictionInput) -> None:
        """
        Validate input data.
        
        Args:
            input_data: Input to validate
        
        Raises:
            ValueError: If input is invalid
        """
        # Tire age validation
        if not 0 <= input_data.tire_age <= 50:
            raise ValueError(f"Tire age must be 0-50, got {input_data.tire_age}")
        
        # Temperature validation
        if not -10 <= input_data.weather_temp <= 60:
            raise ValueError(
                f"Weather temperature must be -10 to 60°C, got {input_data.weather_temp}"
            )
        
        # Driver aggression validation
        if not 0 <= input_data.driver_aggression <= 1:
            raise ValueError(
                f"Driver aggression must be 0-1, got {input_data.driver_aggression}"
            )
        
        # Stint history validation
        if input_data.stint_history:
            if len(input_data.stint_history) > 100:
                raise ValueError(
                    f"Stint history too long: {len(input_data.stint_history)} laps"
                )
            if any(t < 50.0 or t > 200.0 for t in input_data.stint_history):
                raise ValueError("Lap times must be between 50.0 and 200.0 seconds")
    
    def _extract_features(self, input_data: PredictionInput) -> Dict[str, Any]:
        """
        Extract features from input data for model consumption.
        
        Args:
            input_data: Raw input data
        
        Returns:
            Dictionary of extracted features
        """
        features = {
            'tire_compound': input_data.tire_compound,
            'tire_age': input_data.tire_age,
            'weather_temp': input_data.weather_temp,
            'driver_aggression': input_data.driver_aggression,
            'track_name': input_data.track_name,
        }
        
        # Add stint history features
        if input_data.stint_history:
            features['stint_length'] = len(input_data.stint_history)
            features['avg_lap_time'] = sum(input_data.stint_history) / len(input_data.stint_history)
            features['lap_time_std'] = (
                sum((t - features['avg_lap_time']) ** 2 for t in input_data.stint_history)
                / len(input_data.stint_history)
            ) ** 0.5
            features['recent_pace'] = input_data.stint_history[-1] if input_data.stint_history else 0.0
        else:
            features['stint_length'] = 0
            features['avg_lap_time'] = 0.0
            features['lap_time_std'] = 0.0
            features['recent_pace'] = 0.0
        
        # Add optional features
        if input_data.track_temp is not None:
            features['track_temp'] = input_data.track_temp
        
        if input_data.fuel_load is not None:
            features['fuel_load'] = input_data.fuel_load
        
        if input_data.stint_number is not None:
            features['stint_number'] = input_data.stint_number
        
        return features
