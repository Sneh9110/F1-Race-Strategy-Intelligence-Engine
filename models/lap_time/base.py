"""
Base classes and abstractions for lap time prediction models.

Defines the common interface that all lap time models must implement,
along with configuration and I/O data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, validator

from app.utils.validators import validate_numeric_range
from config.settings import settings


class RaceCondition(str, Enum):
    """Race condition types."""
    CLEAN_AIR = "CLEAN_AIR"
    DIRTY_AIR = "DIRTY_AIR"
    SAFETY_CAR = "SAFETY_CAR"
    VIRTUAL_SAFETY_CAR = "VIRTUAL_SAFETY_CAR"
    NORMAL = "NORMAL"


class TireCompound(str, Enum):
    """Tire compound types."""
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"


class PredictionInput(BaseModel):
    """
    Input data for lap time prediction.
    
    Attributes:
        tire_age: Tire age in laps (0-50)
        tire_compound: Tire compound type
        fuel_load: Current fuel load in kg (0-110)
        traffic_state: Traffic condition (CLEAN_AIR or DIRTY_AIR)
        gap_to_ahead: Gap to car ahead in seconds (optional)
        safety_car_active: Whether safety car is deployed
        weather_temp: Weather temperature in °C
        track_temp: Track temperature in °C (optional)
        track_name: Circuit name
        driver_number: Driver number
        lap_number: Current lap number
        session_progress: Race progress (0-1)
        stint_number: Current stint number (optional)
        recent_lap_times: Last 5 lap times in seconds
    """
    tire_age: int = Field(..., ge=0, le=50, description="Tire age in laps")
    tire_compound: TireCompound = Field(..., description="Tire compound")
    fuel_load: float = Field(..., ge=0.0, le=110.0, description="Fuel load in kg")
    traffic_state: RaceCondition = Field(default=RaceCondition.CLEAN_AIR)
    gap_to_ahead: Optional[float] = Field(None, ge=0.0, description="Gap to car ahead in seconds")
    safety_car_active: bool = Field(default=False, description="Safety car deployed")
    weather_temp: float = Field(..., description="Weather temperature in °C")
    track_temp: Optional[float] = Field(None, description="Track temperature in °C")
    track_name: str = Field(..., description="Circuit name")
    driver_number: int = Field(..., ge=1, le=99, description="Driver number")
    lap_number: int = Field(..., ge=1, description="Current lap number")
    session_progress: float = Field(default=0.5, ge=0.0, le=1.0, description="Session progress")
    stint_number: Optional[int] = Field(None, ge=1, description="Stint number")
    recent_lap_times: List[float] = Field(default_factory=list, description="Recent lap times")
    
    @validator('recent_lap_times')
    def validate_lap_times(cls, v):
        """Validate lap times are in realistic range."""
        if v:
            for lap_time in v:
                if not (60.0 <= lap_time <= 150.0):
                    raise ValueError(f"Lap time {lap_time} out of range (60-150s)")
        return v
    
    @validator('gap_to_ahead')
    def validate_gap(cls, v, values):
        """Validate gap is reasonable for dirty air."""
        if v is not None and v < 0:
            raise ValueError("Gap to ahead cannot be negative")
        if values.get('traffic_state') == RaceCondition.DIRTY_AIR and v is None:
            # Default gap for dirty air if not specified
            return 0.8
        return v
    
    class Config:
        use_enum_values = True


class PredictionOutput(BaseModel):
    """
    Output from lap time prediction.
    
    Attributes:
        predicted_lap_time: Predicted lap time in seconds
        confidence: Prediction confidence (0-1)
        pace_components: Breakdown of pace factors
        uncertainty_range: Optional confidence interval (lower, upper)
        metadata: Additional prediction metadata
    """
    predicted_lap_time: float = Field(..., ge=60.0, le=150.0, description="Predicted lap time")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    pace_components: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown: base_pace, tire_effect, fuel_effect, traffic_penalty, weather_adjustment, safety_car_factor"
    )
    uncertainty_range: Optional[Tuple[float, float]] = Field(
        None, description="Confidence interval (lower, upper)"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('predicted_lap_time')
    def validate_lap_time(cls, v):
        """Validate lap time is realistic."""
        if not (60.0 <= v <= 150.0):
            raise ValueError(f"Predicted lap time {v} out of realistic range (60-150s)")
        return v
    
    @validator('uncertainty_range')
    def validate_uncertainty(cls, v):
        """Validate uncertainty range is sensible."""
        if v is not None:
            lower, upper = v
            if lower < 60.0 or upper > 150.0:
                raise ValueError(f"Uncertainty range ({lower}, {upper}) out of bounds")
            if lower >= upper:
                raise ValueError(f"Lower bound {lower} must be less than upper bound {upper}")
        return v
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class ModelConfig:
    """
    Configuration for lap time model training and inference.
    
    Attributes:
        model_type: Type of model (xgboost, lightgbm, ensemble)
        hyperparameters: Model-specific hyperparameters
        training_config: Training settings
        inference_config: Inference settings
        fallback_config: Fallback behavior settings
        version: Model version
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


class BaseLapTimeModel(ABC):
    """
    Abstract base class for lap time prediction models.
    
    All lap time models must implement this interface.
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
            'trained': False,
            'model_type': config.model_type,
            'version': config.version
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
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets (lap times)
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training arguments
        
        Returns:
            Training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Make single prediction.
        
        Args:
            input_data: Prediction input
        
        Returns:
            Prediction output
        """
        pass
    
    @abstractmethod
    def predict_batch(
        self,
        inputs: List[PredictionInput]
    ) -> List[PredictionOutput]:
        """
        Make batch predictions.
        
        Args:
            inputs: List of prediction inputs
        
        Returns:
            List of prediction outputs
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
            Metadata dictionary
        """
        return self.metadata.copy()
    
    def _validate_input(self, input_data: PredictionInput) -> None:
        """
        Validate prediction input.
        
        Args:
            input_data: Input to validate
        
        Raises:
            ValueError: If input is invalid
        """
        # Pydantic already validates, but add custom checks
        if input_data.tire_age < 0 or input_data.tire_age > 50:
            raise ValueError(f"Tire age {input_data.tire_age} out of range (0-50)")
        
        if input_data.fuel_load < 0 or input_data.fuel_load > 110:
            raise ValueError(f"Fuel load {input_data.fuel_load} out of range (0-110)")
        
        if input_data.traffic_state == RaceCondition.DIRTY_AIR and input_data.gap_to_ahead is None:
            raise ValueError("Gap to ahead required for dirty air condition")
    
    def _extract_features(self, input_data: PredictionInput) -> Dict[str, Any]:
        """
        Extract features from prediction input.
        
        Args:
            input_data: Prediction input
        
        Returns:
            Feature dictionary
        """
        features = {
            'tire_age': input_data.tire_age,
            'tire_compound': input_data.tire_compound,
            'fuel_load': input_data.fuel_load,
            'traffic_state': input_data.traffic_state,
            'gap_to_ahead': input_data.gap_to_ahead or 999.0,  # Large value for clean air
            'safety_car_active': 1 if input_data.safety_car_active else 0,
            'weather_temp': input_data.weather_temp,
            'track_temp': input_data.track_temp or input_data.weather_temp + 10,
            'track_name': input_data.track_name,
            'driver_number': input_data.driver_number,
            'lap_number': input_data.lap_number,
            'session_progress': input_data.session_progress,
            'stint_number': input_data.stint_number or 1,
        }
        
        # Add derived features
        if input_data.recent_lap_times:
            features['avg_recent_pace'] = sum(input_data.recent_lap_times) / len(input_data.recent_lap_times)
            features['pace_consistency'] = max(input_data.recent_lap_times) - min(input_data.recent_lap_times)
        else:
            features['avg_recent_pace'] = None
            features['pace_consistency'] = None
        
        return features
