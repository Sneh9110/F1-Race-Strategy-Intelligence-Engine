"""
Weather Data Schema - Track and atmospheric conditions

Defines schemas for real-time weather observations and forecasts.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class TrackCondition(str, Enum):
    """Track surface condition categories."""

    DRY = "DRY"
    DAMP = "DAMP"
    WET = "WET"
    SOAKING = "SOAKING"


class WeatherData(BaseModel):
    """
    Real-time weather observation data.

    Captured from track sensors and weather stations.
    """

    timestamp: datetime = Field(description="UTC timestamp of observation")
    track_temp_celsius: float = Field(ge=10.0, le=60.0, description="Track surface temperature in °C")
    air_temp_celsius: float = Field(ge=0.0, le=50.0, description="Air temperature in °C")
    humidity_percent: float = Field(ge=0.0, le=100.0, description="Relative humidity percentage")
    wind_speed_kmh: float = Field(ge=0.0, le=100.0, description="Wind speed in km/h")
    wind_direction_degrees: float = Field(ge=0.0, lt=360.0, description="Wind direction in degrees (0=N)")
    rainfall_mm: float = Field(ge=0.0, description="Rainfall in mm (0 = no rain)")
    pressure_hpa: float = Field(ge=950.0, le=1050.0, description="Atmospheric pressure in hPa")
    track_condition: TrackCondition = Field(description="Track surface condition")

    @field_validator("track_condition")
    @classmethod
    def validate_condition_with_rain(cls, v: TrackCondition, info) -> TrackCondition:
        """Validate track condition is consistent with rainfall."""
        rainfall = info.data.get("rainfall_mm", 0.0)

        # If significant rainfall, track should not be dry
        if rainfall > 1.0 and v == TrackCondition.DRY:
            raise ValueError(f"Track cannot be DRY with {rainfall}mm rainfall")

        # If no rain, track should not be soaking
        if rainfall == 0.0 and v == TrackCondition.SOAKING:
            raise ValueError("Track cannot be SOAKING with 0mm rainfall")

        return v

    @field_validator("track_temp_celsius")
    @classmethod
    def validate_track_warmer_than_air(cls, v: float, info) -> float:
        """Validate track temperature is typically warmer than air temperature."""
        air_temp = info.data.get("air_temp_celsius")
        if air_temp and v < air_temp - 5.0:
            # Track temp can be slightly cooler but not significantly
            raise ValueError(
                f"Track temp ({v}°C) unusually colder than air temp ({air_temp}°C)"
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-05-26T14:23:00Z",
                "track_temp_celsius": 47.5,
                "air_temp_celsius": 29.2,
                "humidity_percent": 48.5,
                "wind_speed_kmh": 12.3,
                "wind_direction_degrees": 185.0,
                "rainfall_mm": 0.0,
                "pressure_hpa": 1013.2,
                "track_condition": "DRY",
            }
        }


class WeatherForecast(BaseModel):
    """
    Weather forecast prediction for future time window.

    Used for strategy planning and risk assessment.
    """

    forecast_time: datetime = Field(description="UTC timestamp this forecast is for")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="When forecast was generated")
    predicted_track_temp_celsius: float = Field(ge=10.0, le=60.0, description="Predicted track temperature")
    predicted_air_temp_celsius: float = Field(ge=0.0, le=50.0, description="Predicted air temperature")
    predicted_rainfall_mm: float = Field(ge=0.0, description="Predicted rainfall amount")
    rain_probability_percent: float = Field(ge=0.0, le=100.0, description="Probability of rain")
    predicted_track_condition: TrackCondition = Field(description="Predicted track condition")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Forecast confidence (0-1)")

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float, info) -> float:
        """Lower confidence for longer-term forecasts."""
        forecast_time = info.data.get("forecast_time")
        generated_at = info.data.get("generated_at")

        if forecast_time and generated_at:
            hours_ahead = (forecast_time - generated_at).total_seconds() / 3600
            # Confidence should decrease with longer forecast window
            if hours_ahead > 2 and v > 0.9:
                raise ValueError(f"Confidence too high ({v}) for forecast {hours_ahead:.1f}h ahead")

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "forecast_time": "2024-05-26T15:00:00Z",
                "generated_at": "2024-05-26T14:23:00Z",
                "predicted_track_temp_celsius": 48.2,
                "predicted_air_temp_celsius": 30.1,
                "predicted_rainfall_mm": 0.0,
                "rain_probability_percent": 15.0,
                "predicted_track_condition": "DRY",
                "confidence_score": 0.85,
            }
        }


class WeatherSession(BaseModel):
    """
    Complete weather data for an entire session.

    Aggregates observations and forecasts for the session duration.
    """

    session_id: str = Field(description="Unique session identifier")
    track_name: str = Field(description="Circuit name")
    session_start_time: datetime = Field(description="Session start UTC timestamp")
    observations: List[WeatherData] = Field(default_factory=list, description="Historical observations")
    forecasts: List[WeatherForecast] = Field(default_factory=list, description="Future forecasts")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "2024_MONACO_RACE",
                "track_name": "Circuit de Monaco",
                "session_start_time": "2024-05-26T13:00:00Z",
                "observations": [],
                "forecasts": [],
            }
        }


# Utility functions for weather data
def get_average_track_temp(observations: List[WeatherData]) -> float:
    """Calculate average track temperature from observations."""
    if not observations:
        return 0.0
    return sum(obs.track_temp_celsius for obs in observations) / len(observations)


def is_rain_likely(forecasts: List[WeatherForecast], threshold: float = 0.5) -> bool:
    """Check if rain is likely based on forecasts."""
    if not forecasts:
        return False
    avg_rain_prob = sum(f.rain_probability_percent for f in forecasts) / len(forecasts)
    return avg_rain_prob >= (threshold * 100)


def get_track_condition_changes(observations: List[WeatherData]) -> List[tuple]:
    """Identify when track condition changed during session."""
    changes = []
    prev_condition = None

    for obs in observations:
        if prev_condition and obs.track_condition != prev_condition:
            changes.append((obs.timestamp, prev_condition, obs.track_condition))
        prev_condition = obs.track_condition

    return changes
