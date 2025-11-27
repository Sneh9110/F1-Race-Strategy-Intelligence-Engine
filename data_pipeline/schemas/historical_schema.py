"""
Historical Race Data Schema - Past race results and strategy analysis

Defines schemas for historical race outcomes, stint analysis, and strategy patterns.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from datetime import datetime, date
from enum import Enum


class WeatherCondition(str, Enum):
    """Historical weather classification."""

    DRY = "DRY"
    MIXED = "MIXED"
    WET = "WET"


class TireCompound(str, Enum):
    """F1 tire compounds."""

    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"


class HistoricalRace(BaseModel):
    """
    Complete historical race metadata and outcomes.

    Represents a single completed F1 race with all key information.
    """

    race_id: str = Field(description="Unique race identifier (e.g., '2023_ABU_DHABI_RACE')")
    year: int = Field(ge=1950, le=2100, description="Race year")
    round_number: int = Field(ge=1, le=24, description="Round number in championship")
    track_name: str = Field(description="Circuit name")
    race_date: date = Field(description="Race date")
    winner_driver_number: int = Field(ge=1, le=99, description="Winning driver number")
    winner_name: str = Field(description="Winning driver name")
    total_laps: int = Field(ge=1, description="Total laps completed")
    race_duration_seconds: float = Field(ge=0.0, description="Total race duration in seconds")
    weather_conditions: WeatherCondition = Field(description="Overall weather classification")
    safety_car_laps: List[int] = Field(default_factory=list, description="Laps with safety car")
    vsc_laps: List[int] = Field(default_factory=list, description="Laps with virtual safety car")
    red_flag_laps: List[int] = Field(default_factory=list, description="Laps with red flag")

    @field_validator("race_duration_seconds")
    @classmethod
    def validate_race_duration(cls, v: float, info) -> float:
        """Validate race duration is reasonable."""
        total_laps = info.data.get("total_laps", 0)
        if total_laps > 0:
            avg_lap_time = v / total_laps
            # Average lap time should be between 60s and 180s
            if avg_lap_time < 60 or avg_lap_time > 180:
                raise ValueError(
                    f"Average lap time {avg_lap_time:.1f}s unrealistic for race duration"
                )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "race_id": "2023_ABU_DHABI_RACE",
                "year": 2023,
                "round_number": 22,
                "track_name": "Yas Marina Circuit",
                "race_date": "2023-11-26",
                "winner_driver_number": 1,
                "winner_name": "Max Verstappen",
                "total_laps": 58,
                "race_duration_seconds": 5232.0,
                "weather_conditions": "DRY",
                "safety_car_laps": [15, 16, 17, 18],
                "vsc_laps": [],
                "red_flag_laps": [],
            }
        }


class HistoricalStint(BaseModel):
    """
    Detailed stint data for a driver in a historical race.

    Represents continuous running on one set of tires.
    """

    driver_number: int = Field(ge=1, le=99, description="Driver racing number")
    driver_name: str = Field(description="Driver name")
    stint_number: int = Field(ge=1, description="Stint number (1st, 2nd, etc.)")
    tire_compound: TireCompound = Field(description="Tire compound used")
    start_lap: int = Field(ge=1, description="First lap of stint")
    end_lap: int = Field(ge=1, description="Last lap of stint")
    lap_times: List[float] = Field(description="Lap times during stint (seconds)")
    avg_pace: float = Field(ge=0.0, description="Average lap time (seconds)")
    degradation_rate: float = Field(description="Pace degradation per lap (seconds/lap)")
    pit_stop_duration: Optional[float] = Field(None, ge=0.0, description="Pit stop duration if applicable")

    @field_validator("end_lap")
    @classmethod
    def validate_end_after_start(cls, v: int, info) -> int:
        """Validate end lap is after start lap."""
        start_lap = info.data.get("start_lap")
        if start_lap and v < start_lap:
            raise ValueError(f"End lap ({v}) must be >= start lap ({start_lap})")
        return v

    @field_validator("lap_times")
    @classmethod
    def validate_lap_times_length(cls, v: List[float], info) -> List[float]:
        """Validate lap times match stint length."""
        start_lap = info.data.get("start_lap")
        end_lap = info.data.get("end_lap")

        if start_lap and end_lap:
            expected_laps = end_lap - start_lap + 1
            if len(v) != expected_laps:
                raise ValueError(
                    f"Expected {expected_laps} lap times, got {len(v)}"
                )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "driver_number": 1,
                "driver_name": "Max Verstappen",
                "stint_number": 2,
                "tire_compound": "HARD",
                "start_lap": 19,
                "end_lap": 58,
                "lap_times": [91.2, 90.8, 90.5, 90.7],  # Truncated for example
                "avg_pace": 90.8,
                "degradation_rate": 0.015,
                "pit_stop_duration": 2.3,
            }
        }


class HistoricalStrategy(BaseModel):
    """
    Complete race strategy for a driver.

    Aggregates all stints and provides overall strategy classification.
    """

    race_id: str = Field(description="Race identifier")
    driver_number: int = Field(ge=1, le=99, description="Driver racing number")
    driver_name: str = Field(description="Driver name")
    team_name: str = Field(description="Team name")
    final_position: int = Field(ge=1, description="Final race position")
    num_pit_stops: int = Field(ge=0, description="Total number of pit stops")
    stints: List[HistoricalStint] = Field(description="List of all stints")
    total_race_time: float = Field(ge=0.0, description="Total race time in seconds")
    strategy_type: str = Field(description="Strategy classification (e.g., '1-STOP', '2-STOP')")

    @field_validator("num_pit_stops")
    @classmethod
    def validate_stops_match_stints(cls, v: int, info) -> int:
        """Validate number of stops matches stint count."""
        stints = info.data.get("stints", [])
        if stints:
            expected_stops = len(stints) - 1  # Number of stops = stints - 1
            if v != expected_stops:
                raise ValueError(
                    f"Number of stops ({v}) doesn't match stints ({len(stints)})"
                )
        return v

    @field_validator("strategy_type")
    @classmethod
    def validate_strategy_type(cls, v: str, info) -> str:
        """Auto-generate strategy type if not provided."""
        num_stops = info.data.get("num_pit_stops", 0)
        expected_type = f"{num_stops}-STOP"
        if v != expected_type:
            return expected_type
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "race_id": "2023_ABU_DHABI_RACE",
                "driver_number": 1,
                "driver_name": "Max Verstappen",
                "team_name": "Red Bull Racing",
                "final_position": 1,
                "num_pit_stops": 1,
                "stints": [],
                "total_race_time": 5232.0,
                "strategy_type": "1-STOP",
            }
        }


class HistoricalRaceAnalysis(BaseModel):
    """
    Comprehensive analysis of a historical race.

    Aggregates all driver strategies and key race metrics.
    """

    race: HistoricalRace = Field(description="Race metadata")
    strategies: List[HistoricalStrategy] = Field(description="All driver strategies")
    fastest_lap_time: float = Field(description="Fastest lap of the race")
    fastest_lap_driver: str = Field(description="Driver who set fastest lap")
    avg_pit_stop_duration: float = Field(description="Average pit stop duration")
    most_common_strategy: str = Field(description="Most popular strategy type")

    class Config:
        json_schema_extra = {
            "example": {
                "race": {},
                "strategies": [],
                "fastest_lap_time": 85.3,
                "fastest_lap_driver": "Max Verstappen",
                "avg_pit_stop_duration": 2.4,
                "most_common_strategy": "1-STOP",
            }
        }


# Utility functions for historical data
def calculate_stint_degradation(lap_times: List[float]) -> float:
    """Calculate tire degradation rate from lap times."""
    if len(lap_times) < 2:
        return 0.0

    # Simple linear regression slope
    n = len(lap_times)
    sum_x = sum(range(n))
    sum_y = sum(lap_times)
    sum_xy = sum(i * lap_times[i] for i in range(n))
    sum_x2 = sum(i * i for i in range(n))

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    return slope


def get_optimal_stint_length(stints: List[HistoricalStint], compound: TireCompound) -> Dict[str, float]:
    """Analyze optimal stint length for tire compound."""
    relevant_stints = [s for s in stints if s.tire_compound == compound]

    if not relevant_stints:
        return {}

    stint_lengths = [s.end_lap - s.start_lap + 1 for s in relevant_stints]
    avg_paces = [s.avg_pace for s in relevant_stints]

    return {
        "avg_stint_length": sum(stint_lengths) / len(stint_lengths),
        "avg_pace": sum(avg_paces) / len(avg_paces),
        "sample_size": len(relevant_stints),
    }


def classify_strategy_effectiveness(strategy: HistoricalStrategy) -> str:
    """Classify strategy effectiveness based on outcome."""
    # Simple classification based on final position
    if strategy.final_position <= 3:
        return "EFFECTIVE"
    elif strategy.final_position <= 10:
        return "MODERATE"
    else:
        return "INEFFECTIVE"
