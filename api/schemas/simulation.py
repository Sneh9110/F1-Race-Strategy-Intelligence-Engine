"""Simulation and strategy schemas."""

from typing import Any
from pydantic import BaseModel, Field
from datetime import datetime

from api.schemas.common import MetadataBase


# Simulation Request/Response Schemas
class StrategySimulationRequest(BaseModel):
    """Request for race strategy simulation."""
    
    circuit_name: str
    total_laps: int = Field(..., ge=1, le=100)
    starting_tire: str
    fuel_load: float = Field(..., ge=0, le=110)
    weather_condition: str = "Dry"
    pit_stops: list[dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "circuit_name": "Monaco",
                "total_laps": 78,
                "starting_tire": "SOFT",
                "fuel_load": 105.0,
                "weather_condition": "Dry",
                "pit_stops": [
                    {"lap": 25, "tire": "MEDIUM"},
                    {"lap": 50, "tire": "MEDIUM"}
                ]
            }
        }


class StrategySimulationResponse(BaseModel):
    """Response for race strategy simulation."""
    
    total_race_time: float = Field(..., description="Total race time in seconds")
    final_position: int = Field(..., ge=1, description="Predicted final position")
    pit_stop_count: int
    tire_strategy: list[str]
    lap_times: list[float] = Field(default_factory=list)
    metadata: MetadataBase


class CompareStrategiesRequest(BaseModel):
    """Request to compare multiple strategies."""
    
    circuit_name: str
    total_laps: int
    strategies: list[dict[str, Any]] = Field(..., min_length=2, max_length=5)
    
    class Config:
        json_schema_extra = {
            "example": {
                "circuit_name": "Silverstone",
                "total_laps": 52,
                "strategies": [
                    {"name": "One-stop", "pit_stops": [{"lap": 26, "tire": "HARD"}]},
                    {"name": "Two-stop", "pit_stops": [{"lap": 18, "tire": "MEDIUM"}, {"lap": 35, "tire": "SOFT"}]}
                ]
            }
        }


class CompareStrategiesResponse(BaseModel):
    """Response for strategy comparison."""
    
    best_strategy: str
    comparisons: list[dict[str, Any]]
    time_differences: dict[str, float]
    metadata: MetadataBase


# Decision Engine Schemas
class DecisionRequest(BaseModel):
    """Request for strategy recommendation."""
    
    circuit_name: str
    current_lap: int
    total_laps: int
    current_position: int
    current_tire: str
    tire_age: int
    fuel_remaining: float
    gap_to_leader: float = 0.0
    gap_to_next: float = 0.0
    weather_condition: str = "Dry"
    safety_car_deployed: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "circuit_name": "Spa",
                "current_lap": 20,
                "total_laps": 44,
                "current_position": 3,
                "current_tire": "MEDIUM",
                "tire_age": 18,
                "fuel_remaining": 60.0,
                "gap_to_leader": 8.5,
                "gap_to_next": 2.3,
                "weather_condition": "Dry",
                "safety_car_deployed": False
            }
        }


class DecisionResponse(BaseModel):
    """Response with strategy recommendation."""
    
    recommendation: str = Field(..., description="Recommended action")
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str = Field(..., description="Explanation of recommendation")
    alternative_options: list[str] = Field(default_factory=list)
    risk_assessment: str = Field(default="MEDIUM")
    metadata: MetadataBase


class DecisionModule(BaseModel):
    """Decision module information."""
    
    name: str
    priority: int
    enabled: bool
    description: str
