"""Request/response schemas for prediction endpoints."""

from typing import Any
from pydantic import BaseModel, Field
from datetime import datetime

from api.schemas.common import MetadataBase, ModelStats


# Lap Time Prediction Schemas
class LapTimePredictionRequest(BaseModel):
    """Request for lap time prediction."""
    
    circuit_name: str = Field(..., description="Circuit name")
    driver: str = Field(..., description="Driver name")
    team: str = Field(..., description="Team name")
    tire_compound: str = Field(..., description="Tire compound (SOFT, MEDIUM, HARD)")
    tire_age: int = Field(..., ge=0, description="Tire age in laps")
    fuel_load: float = Field(..., ge=0, le=110, description="Fuel load in kg")
    track_temp: float = Field(..., ge=-10, le=60, description="Track temperature in Celsius")
    air_temp: float = Field(..., ge=-10, le=50, description="Air temperature in Celsius")
    weather_condition: str = Field(default="Dry", description="Weather condition")
    
    class Config:
        json_schema_extra = {
            "example": {
                "circuit_name": "Monaco",
                "driver": "Max Verstappen",
                "team": "Red Bull Racing",
                "tire_compound": "SOFT",
                "tire_age": 5,
                "fuel_load": 80.0,
                "track_temp": 35.0,
                "air_temp": 25.0,
                "weather_condition": "Dry"
            }
        }


class LapTimePredictionResponse(BaseModel):
    """Response for lap time prediction."""
    
    predicted_lap_time: float = Field(..., description="Predicted lap time in seconds")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    metadata: MetadataBase


# Tire Degradation Prediction Schemas
class TireDegradationRequest(BaseModel):
    """Request for tire degradation prediction."""
    
    circuit_name: str
    tire_compound: str
    laps: int = Field(..., ge=1, le=100)
    track_temp: float
    fuel_load: float
    downforce_level: str = Field(default="Medium", description="HIGH, MEDIUM, LOW")
    
    class Config:
        json_schema_extra = {
            "example": {
                "circuit_name": "Silverstone",
                "tire_compound": "MEDIUM",
                "laps": 15,
                "track_temp": 30.0,
                "fuel_load": 75.0,
                "downforce_level": "Medium"
            }
        }


class TireDegradationResponse(BaseModel):
    """Response for tire degradation prediction."""
    
    degradation_per_lap: float = Field(..., description="% degradation per lap")
    total_degradation: float = Field(..., description="Total % degradation")
    remaining_performance: float = Field(..., ge=0, le=100, description="Remaining tire performance %")
    metadata: MetadataBase


# Safety Car Prediction Schemas
class SafetyCarRequest(BaseModel):
    """Request for safety car probability prediction."""
    
    circuit_name: str
    lap: int = Field(..., ge=1)
    total_laps: int = Field(..., ge=1)
    weather_condition: str
    incidents_so_far: int = Field(default=0, ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "circuit_name": "Baku",
                "lap": 35,
                "total_laps": 51,
                "weather_condition": "Dry",
                "incidents_so_far": 1
            }
        }


class SafetyCarResponse(BaseModel):
    """Response for safety car probability."""
    
    probability: float = Field(..., ge=0, le=1, description="Probability of safety car deployment")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH")
    metadata: MetadataBase


# Pit Stop Loss Prediction Schemas
class PitStopLossRequest(BaseModel):
    """Request for pit stop time loss prediction."""
    
    circuit_name: str
    pit_lane_type: str = Field(default="Standard", description="Standard or Short")
    traffic_density: float = Field(default=0.5, ge=0, le=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "circuit_name": "Monza",
                "pit_lane_type": "Standard",
                "traffic_density": 0.3
            }
        }


class PitStopLossResponse(BaseModel):
    """Response for pit stop time loss."""
    
    time_loss: float = Field(..., description="Expected time loss in seconds")
    range_min: float = Field(..., description="Minimum expected time loss")
    range_max: float = Field(..., description="Maximum expected time loss")
    metadata: MetadataBase


# Batch Prediction Schemas
class BatchPredictionRequest(BaseModel):
    """Generic batch prediction request."""
    
    predictions: list[dict[str, Any]] = Field(..., min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    """Generic batch prediction response."""
    
    results: list[dict[str, Any]]
    total: int
    successful: int
    failed: int
    metadata: MetadataBase


# Model Statistics Schema
class PredictionModelStats(ModelStats):
    """Extended model statistics for prediction models."""
    
    total_predictions: int = Field(default=0)
    cache_hit_rate: float = Field(default=0.0, ge=0, le=1)
    avg_latency_ms: float = Field(default=0.0)
    error_rate: float = Field(default=0.0, ge=0, le=1)
