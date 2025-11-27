"""
Telemetry Data Schema - High-frequency driver and car telemetry

Defines schemas for speed, throttle, brake, tire temperatures, and position data.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class DRSStatus(str, Enum):
    """DRS (Drag Reduction System) activation status."""

    CLOSED = "CLOSED"
    AVAILABLE = "AVAILABLE"
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"


class TirePosition(str, Enum):
    """Tire position on car."""

    FRONT_LEFT = "FL"
    FRONT_RIGHT = "FR"
    REAR_LEFT = "RL"
    REAR_RIGHT = "RR"


class TelemetryPoint(BaseModel):
    """
    Single telemetry data point captured at high frequency (typically 10-60 Hz).

    Represents instantaneous car state at a specific point on track.
    """

    timestamp: datetime = Field(description="UTC timestamp of telemetry capture")
    driver_number: int = Field(ge=1, le=99, description="Driver racing number")
    speed_kmh: float = Field(ge=0.0, le=380.0, description="Speed in km/h")
    throttle_percent: float = Field(ge=0.0, le=100.0, description="Throttle position (0-100%)")
    brake_percent: float = Field(ge=0.0, le=100.0, description="Brake pressure (0-100%)")
    gear: int = Field(ge=0, le=8, description="Current gear (0=neutral/reverse, 1-8=forward)")
    rpm: int = Field(ge=0, le=15000, description="Engine RPM")
    drs_status: DRSStatus = Field(description="DRS activation status")

    # Tire temperatures (Celsius)
    tire_temp_fl: float = Field(ge=50.0, le=130.0, description="Front left tire temp (°C)")
    tire_temp_fr: float = Field(ge=50.0, le=130.0, description="Front right tire temp (°C)")
    tire_temp_rl: float = Field(ge=50.0, le=130.0, description="Rear left tire temp (°C)")
    tire_temp_rr: float = Field(ge=50.0, le=130.0, description="Rear right tire temp (°C)")

    # Brake temperatures (Celsius)
    brake_temp_fl: float = Field(ge=100.0, le=1200.0, description="Front left brake temp (°C)")
    brake_temp_fr: float = Field(ge=100.0, le=1200.0, description="Front right brake temp (°C)")
    brake_temp_rl: float = Field(ge=100.0, le=1200.0, description="Rear left brake temp (°C)")
    brake_temp_rr: float = Field(ge=100.0, le=1200.0, description="Rear right brake temp (°C)")

    # Vehicle state
    fuel_remaining_kg: Optional[float] = Field(None, ge=0.0, le=110.0, description="Fuel remaining (kg)")

    # Position coordinates (for track mapping)
    position_x: Optional[float] = Field(None, description="X coordinate on track (meters)")
    position_y: Optional[float] = Field(None, description="Y coordinate on track (meters)")

    @field_validator("throttle_percent")
    @classmethod
    def validate_throttle_brake(cls, v: float, info) -> float:
        """Validate throttle and brake are not both high simultaneously."""
        brake = info.data.get("brake_percent", 0.0)
        # Allow small overlap for transition, but not full application
        if v > 50.0 and brake > 50.0:
            raise ValueError("Throttle and brake cannot both be >50% simultaneously")
        return v

    @field_validator("rpm")
    @classmethod
    def validate_rpm_with_gear(cls, v: int, info) -> int:
        """Validate RPM is reasonable for current gear."""
        gear = info.data.get("gear", 1)
        speed = info.data.get("speed_kmh", 0.0)

        # In neutral/reverse, RPM should be low
        if gear == 0 and v > 8000:
            raise ValueError(f"RPM too high ({v}) for neutral/reverse gear")

        # At speed, RPM should not be at redline in low gears
        if speed > 200 and gear < 4 and v > 13000:
            raise ValueError(f"RPM ({v}) too high for gear {gear} at speed {speed} km/h")

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-05-26T14:23:45.123Z",
                "driver_number": 1,
                "speed_kmh": 312.5,
                "throttle_percent": 100.0,
                "brake_percent": 0.0,
                "gear": 8,
                "rpm": 12500,
                "drs_status": "ACTIVE",
                "tire_temp_fl": 95.2,
                "tire_temp_fr": 98.1,
                "tire_temp_rl": 92.3,
                "tire_temp_rr": 94.5,
                "brake_temp_fl": 650.0,
                "brake_temp_fr": 680.0,
                "brake_temp_rl": 450.0,
                "brake_temp_rr": 470.0,
                "fuel_remaining_kg": 109.8,
                "position_x": 1234.5,
                "position_y": 678.9,
            }
        }


class TelemetrySession(BaseModel):
    """
    Complete telemetry data for a driver in a session.

    Aggregates all high-frequency telemetry points for analysis.
    """

    session_id: str = Field(description="Unique session identifier")
    driver_number: int = Field(ge=1, le=99, description="Driver racing number")
    lap_number: Optional[int] = Field(None, description="Specific lap number (None = entire session)")
    telemetry_points: List[TelemetryPoint] = Field(
        default_factory=list, description="High-frequency telemetry data points"
    )
    sample_rate_hz: float = Field(default=10.0, description="Telemetry sampling frequency (Hz)")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "2024_MONACO_RACE",
                "driver_number": 1,
                "lap_number": 23,
                "telemetry_points": [],
                "sample_rate_hz": 10.0,
            }
        }


class TelemetryLapSummary(BaseModel):
    """
    Aggregated telemetry summary for a complete lap.

    Useful for analysis without processing full high-frequency data.
    """

    driver_number: int = Field(ge=1, le=99, description="Driver racing number")
    lap_number: int = Field(ge=1, description="Lap number")
    max_speed_kmh: float = Field(ge=0.0, le=380.0, description="Maximum speed achieved")
    avg_speed_kmh: float = Field(ge=0.0, description="Average speed over lap")
    max_rpm: int = Field(ge=0, le=15000, description="Maximum RPM")
    avg_throttle_percent: float = Field(ge=0.0, le=100.0, description="Average throttle application")
    avg_brake_percent: float = Field(ge=0.0, le=100.0, description="Average brake application")
    drs_activations: int = Field(ge=0, description="Number of DRS activations")

    # Temperature averages
    avg_tire_temp_fl: float = Field(description="Average front left tire temperature")
    avg_tire_temp_fr: float = Field(description="Average front right tire temperature")
    avg_tire_temp_rl: float = Field(description="Average rear left tire temperature")
    avg_tire_temp_rr: float = Field(description="Average rear right tire temperature")

    max_brake_temp_fl: float = Field(description="Maximum front left brake temperature")
    max_brake_temp_fr: float = Field(description="Maximum front right brake temperature")

    fuel_consumed_kg: Optional[float] = Field(None, description="Fuel consumed during lap")

    class Config:
        json_schema_extra = {
            "example": {
                "driver_number": 1,
                "lap_number": 23,
                "max_speed_kmh": 312.5,
                "avg_speed_kmh": 185.3,
                "max_rpm": 12800,
                "avg_throttle_percent": 65.2,
                "avg_brake_percent": 8.5,
                "drs_activations": 2,
                "avg_tire_temp_fl": 95.2,
                "avg_tire_temp_fr": 97.8,
                "avg_tire_temp_rl": 91.5,
                "avg_tire_temp_rr": 93.7,
                "max_brake_temp_fl": 720.0,
                "max_brake_temp_fr": 750.0,
                "fuel_consumed_kg": 1.8,
            }
        }


# Utility functions for telemetry data
def calculate_lap_summary(telemetry_points: List[TelemetryPoint]) -> Dict[str, float]:
    """Calculate summary statistics from telemetry points."""
    if not telemetry_points:
        return {}

    return {
        "max_speed": max(p.speed_kmh for p in telemetry_points),
        "avg_speed": sum(p.speed_kmh for p in telemetry_points) / len(telemetry_points),
        "max_rpm": max(p.rpm for p in telemetry_points),
        "avg_throttle": sum(p.throttle_percent for p in telemetry_points) / len(telemetry_points),
        "avg_brake": sum(p.brake_percent for p in telemetry_points) / len(telemetry_points),
    }


def detect_drs_activations(telemetry_points: List[TelemetryPoint]) -> int:
    """Count number of DRS activations in telemetry data."""
    activations = 0
    prev_status = DRSStatus.CLOSED

    for point in telemetry_points:
        if point.drs_status == DRSStatus.ACTIVE and prev_status != DRSStatus.ACTIVE:
            activations += 1
        prev_status = point.drs_status

    return activations


def get_fuel_consumption(telemetry_points: List[TelemetryPoint]) -> Optional[float]:
    """Calculate fuel consumption from telemetry data."""
    if not telemetry_points or telemetry_points[0].fuel_remaining_kg is None:
        return None

    valid_points = [p for p in telemetry_points if p.fuel_remaining_kg is not None]
    if len(valid_points) < 2:
        return None

    return valid_points[0].fuel_remaining_kg - valid_points[-1].fuel_remaining_kg
