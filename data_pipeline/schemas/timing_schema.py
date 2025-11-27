"""
FIA Timing Data Schema - Real-time race timing and lap data structures

Defines comprehensive schemas for lap times, sector times, and session timing data.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class TrackStatus(str, Enum):
    """Track status flags."""

    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    SAFETY_CAR = "SAFETY_CAR"
    VIRTUAL_SAFETY_CAR = "VIRTUAL_SAFETY_CAR"


class TireCompound(str, Enum):
    """F1 tire compounds."""

    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"


class TimingPoint(BaseModel):
    """
    Real-time timing point data for a single driver at a specific moment.

    Captured during live timing feed ingestion.
    """

    timestamp: datetime = Field(description="UTC timestamp of timing point")
    driver_number: int = Field(ge=1, le=99, description="Driver racing number")
    position: int = Field(ge=1, le=20, description="Current race position")
    lap_number: int = Field(ge=0, description="Current lap number (0 = out lap)")
    lap_time: Optional[float] = Field(None, ge=30.0, le=150.0, description="Lap time in seconds")
    sector_1_time: Optional[float] = Field(None, ge=5.0, le=60.0, description="Sector 1 time in seconds")
    sector_2_time: Optional[float] = Field(None, ge=5.0, le=60.0, description="Sector 2 time in seconds")
    sector_3_time: Optional[float] = Field(None, ge=5.0, le=60.0, description="Sector 3 time in seconds")
    speed_trap: Optional[float] = Field(None, ge=0.0, le=380.0, description="Speed trap km/h")
    gap_to_leader: float = Field(ge=0.0, description="Gap to race leader in seconds")
    interval_to_ahead: Optional[float] = Field(None, ge=0.0, description="Interval to car ahead in seconds")

    @field_validator("gap_to_leader")
    @classmethod
    def validate_gap(cls, v: float, info) -> float:
        """Validate gap is zero for leader (position 1)."""
        if info.data.get("position") == 1 and v > 0.01:
            raise ValueError("Leader must have zero gap to leader")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-05-26T14:23:45.123Z",
                "driver_number": 1,
                "position": 1,
                "lap_number": 23,
                "lap_time": 83.456,
                "sector_1_time": 28.123,
                "sector_2_time": 30.234,
                "sector_3_time": 25.099,
                "speed_trap": 312.5,
                "gap_to_leader": 0.0,
                "interval_to_ahead": None,
            }
        }


class LapData(BaseModel):
    """
    Complete lap data including timing, tire information, and pit stops.

    Represents a single completed lap for a driver.
    """

    lap_number: int = Field(ge=1, description="Lap number")
    lap_time: float = Field(ge=30.0, le=150.0, description="Total lap time in seconds")
    sector_1_time: float = Field(ge=5.0, le=60.0, description="Sector 1 time in seconds")
    sector_2_time: float = Field(ge=5.0, le=60.0, description="Sector 2 time in seconds")
    sector_3_time: float = Field(ge=5.0, le=60.0, description="Sector 3 time in seconds")
    tire_compound: TireCompound = Field(description="Tire compound used")
    tire_age: int = Field(ge=0, description="Tire age in laps")
    pit_in_time: Optional[float] = Field(None, description="Time entered pit lane (seconds)")
    pit_out_time: Optional[float] = Field(None, description="Time exited pit lane (seconds)")
    is_personal_best: bool = Field(default=False, description="Personal best lap for driver")
    track_status: TrackStatus = Field(default=TrackStatus.GREEN, description="Track status during lap")

    @field_validator("lap_time")
    @classmethod
    def validate_sector_sum(cls, v: float, info) -> float:
        """Validate sector times sum to lap time within tolerance."""
        s1 = info.data.get("sector_1_time")
        s2 = info.data.get("sector_2_time")
        s3 = info.data.get("sector_3_time")

        if s1 and s2 and s3:
            sector_sum = s1 + s2 + s3
            if abs(v - sector_sum) > 0.1:  # 100ms tolerance
                raise ValueError(
                    f"Sector sum ({sector_sum:.3f}s) doesn't match lap time ({v:.3f}s)"
                )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "lap_number": 23,
                "lap_time": 83.456,
                "sector_1_time": 28.123,
                "sector_2_time": 30.234,
                "sector_3_time": 25.099,
                "tire_compound": "SOFT",
                "tire_age": 5,
                "pit_in_time": None,
                "pit_out_time": None,
                "is_personal_best": True,
                "track_status": "GREEN",
            }
        }


class SessionTiming(BaseModel):
    """
    Complete timing data for an entire session.

    Aggregates all lap data for all drivers in a session.
    """

    session_id: str = Field(description="Unique session identifier (e.g., '2024_MONACO_RACE')")
    session_type: str = Field(description="Session type (FP1/FP2/FP3/QUALIFYING/SPRINT/RACE)")
    track_name: str = Field(description="Circuit name")
    session_start_time: datetime = Field(description="Session start UTC timestamp")
    lap_data: List[LapData] = Field(default_factory=list, description="List of all lap data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Data capture timestamp")

    @field_validator("session_type")
    @classmethod
    def validate_session_type(cls, v: str) -> str:
        """Validate session type is recognized."""
        valid_types = ["FP1", "FP2", "FP3", "QUALIFYING", "SPRINT", "RACE"]
        if v.upper() not in valid_types:
            raise ValueError(f"Invalid session type. Must be one of: {valid_types}")
        return v.upper()

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "2024_MONACO_RACE",
                "session_type": "RACE",
                "track_name": "Circuit de Monaco",
                "session_start_time": "2024-05-26T13:00:00Z",
                "lap_data": [],
                "timestamp": "2024-05-26T15:30:00Z",
            }
        }


# Utility functions for timing data
def calculate_average_lap_time(lap_data: List[LapData]) -> float:
    """Calculate average lap time from lap data."""
    if not lap_data:
        return 0.0
    return sum(lap.lap_time for lap in lap_data) / len(lap_data)


def get_fastest_lap(lap_data: List[LapData]) -> Optional[LapData]:
    """Get fastest lap from lap data."""
    if not lap_data:
        return None
    return min(lap_data, key=lambda x: x.lap_time)


def filter_clean_laps(lap_data: List[LapData]) -> List[LapData]:
    """Filter laps completed under green flag conditions."""
    return [lap for lap in lap_data if lap.track_status == TrackStatus.GREEN and not lap.pit_in_time]
