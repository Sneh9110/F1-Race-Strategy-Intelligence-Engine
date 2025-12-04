"""
Pydantic schemas for race simulation input/output.

Defines comprehensive data structures for simulation configuration, driver states,
race results, strategy options, and Monte Carlo analysis outputs.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field, validator


class TireCompound(str, Enum):
    """Tire compound types."""
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"


class PaceTarget(str, Enum):
    """Race pace strategy targets."""
    AGGRESSIVE = "AGGRESSIVE"
    CONSERVATIVE = "CONSERVATIVE"
    BALANCED = "BALANCED"


class TrafficState(str, Enum):
    """Air condition behind other cars."""
    CLEAN_AIR = "CLEAN_AIR"
    DIRTY_AIR = "DIRTY_AIR"


class DriverState(BaseModel):
    """Current state of a driver during race simulation."""
    driver_number: int = Field(..., ge=1, le=99, description="Driver number")
    current_position: int = Field(..., ge=1, le=20, description="Current race position")
    tire_compound: TireCompound = Field(..., description="Current tire compound")
    tire_age: int = Field(..., ge=0, le=50, description="Tire age in laps")
    fuel_load: float = Field(..., ge=0.0, le=110.0, description="Current fuel load in kg")
    gap_to_ahead: Optional[float] = Field(None, description="Gap to car ahead in seconds")
    gap_to_behind: Optional[float] = Field(None, description="Gap to car behind in seconds")
    recent_lap_times: List[float] = Field(default_factory=list, description="Last 5 lap times")
    pit_stops_completed: int = Field(0, ge=0, le=5, description="Number of pit stops completed")
    stint_number: int = Field(1, ge=1, le=6, description="Current stint number")
    cumulative_race_time: float = Field(0.0, ge=0.0, description="Total race time so far")
    
    class Config:
        use_enum_values = True


class RaceConfig(BaseModel):
    """Race configuration and environment."""
    track_name: str = Field(..., description="Circuit name (e.g., 'Monaco', 'Monza')")
    total_laps: int = Field(..., ge=1, le=100, description="Total race laps")
    current_lap: int = Field(1, ge=1, description="Starting lap for simulation")
    weather_temp: float = Field(25.0, ge=-10.0, le=50.0, description="Ambient temperature in °C")
    track_temp: float = Field(35.0, ge=0.0, le=70.0, description="Track surface temperature in °C")
    session_id: Optional[str] = Field(None, description="Session identifier")
    grid_positions: Dict[int, int] = Field(..., description="Starting grid: driver_number -> position")
    safety_car_active: bool = Field(False, description="Is safety car currently deployed")
    vsc_active: bool = Field(False, description="Is virtual safety car active")
    
    @validator("current_lap")
    def current_lap_within_race(cls, v, values):
        if "total_laps" in values and v > values["total_laps"]:
            raise ValueError(f"current_lap {v} cannot exceed total_laps {values['total_laps']}")
        return v


class StrategyOption(BaseModel):
    """Pit stop strategy option."""
    pit_laps: List[int] = Field(..., description="Laps on which to pit (e.g., [20, 40] for 2-stop)")
    tire_sequence: List[TireCompound] = Field(..., description="Tire compounds per stint")
    target_pace: PaceTarget = Field(PaceTarget.BALANCED, description="Pace target")
    
    @validator("tire_sequence")
    def tire_sequence_length(cls, v, values):
        if "pit_laps" in values:
            expected_length = len(values["pit_laps"]) + 1
            if len(v) != expected_length:
                raise ValueError(
                    f"tire_sequence length {len(v)} must equal pit_laps length + 1 ({expected_length})"
                )
        return v
    
    @validator("pit_laps")
    def pit_laps_ordered(cls, v):
        if len(v) > 1 and v != sorted(v):
            raise ValueError("pit_laps must be in ascending order")
        return v
    
    class Config:
        use_enum_values = True


class SimulationInput(BaseModel):
    """Complete input for race simulation."""
    race_config: RaceConfig = Field(..., description="Race configuration")
    drivers: List[DriverState] = Field(..., min_items=1, max_items=20, description="Driver states")
    strategy_to_evaluate: Optional[StrategyOption] = Field(None, description="Baseline strategy (used if per_driver_strategies not specified)")
    per_driver_strategies: Optional[Dict[int, StrategyOption]] = Field(None, description="Per-driver strategy mapping (driver_number -> strategy)")
    monte_carlo_runs: int = Field(1, ge=1, le=10000, description="Number of MC runs")
    enable_what_if: bool = Field(False, description="Enable what-if scenario analysis")
    what_if_scenario: Optional[Dict[str, Any]] = Field(None, description="Scenario parameters")
    what_if_params: Optional[Dict[str, Any]] = Field(None, description="What-if parameters for scenario injection")
    
    @validator("drivers")
    def unique_driver_numbers(cls, v):
        driver_numbers = [d.driver_number for d in v]
        if len(driver_numbers) != len(set(driver_numbers)):
            raise ValueError("Driver numbers must be unique")
        return v
    
    @validator("drivers")
    def valid_positions(cls, v):
        """Validate positions are unique and in reasonable range.
        
        Note: Non-consecutive positions are allowed to support mid-race scenarios
        where drivers may have DNF'd or not yet started.
        """
        positions = [d.current_position for d in v]
        if len(positions) != len(set(positions)):
            raise ValueError("Driver positions must be unique")
        if any(p < 1 or p > 20 for p in positions):
            raise ValueError("Positions must be in range 1-20")
        return v
    
    @validator("strategy_to_evaluate", always=True)
    def validate_strategy(cls, v, values):
        """Ensure at least one strategy source is provided."""
        per_driver = values.get("per_driver_strategies")
        if v is None and (per_driver is None or len(per_driver) == 0):
            raise ValueError("Must provide either strategy_to_evaluate or per_driver_strategies")
        return v


class LapResult(BaseModel):
    """Result of a single lap for a driver."""
    lap_number: int = Field(..., ge=1, description="Lap number")
    lap_time: float = Field(..., ge=0.0, description="Lap time in seconds")
    tire_age: int = Field(..., ge=0, le=50, description="Tire age at lap end")
    fuel_load: float = Field(..., ge=0.0, le=110.0, description="Fuel load at lap end")
    position: int = Field(..., ge=1, le=20, description="Position at lap end")
    gap_to_leader: float = Field(..., ge=0.0, description="Gap to leader in seconds")
    tire_compound: TireCompound = Field(..., description="Tire compound used")
    safety_car_active: bool = Field(False, description="Was SC active this lap")
    
    class Config:
        use_enum_values = True


class StintResult(BaseModel):
    """Result of a stint (between pit stops)."""
    stint_number: int = Field(..., ge=1, description="Stint number")
    start_lap: int = Field(..., ge=1, description="First lap of stint")
    end_lap: int = Field(..., ge=1, description="Last lap of stint")
    tire_compound: TireCompound = Field(..., description="Tire compound used")
    avg_lap_time: float = Field(..., ge=0.0, description="Average lap time in seconds")
    degradation_rate: float = Field(..., description="Degradation in s/lap")
    pit_lap: Optional[int] = Field(None, description="Lap of pit stop ending this stint")
    
    class Config:
        use_enum_values = True


class PitStopInfo(BaseModel):
    """Pit stop details."""
    lap: int = Field(..., ge=1, description="Lap number of pit stop")
    duration: float = Field(..., ge=0.0, description="Pit stop duration in seconds")
    loss: float = Field(..., ge=0.0, description="Time loss vs staying out")
    old_compound: TireCompound = Field(..., description="Compound before pit")
    new_compound: TireCompound = Field(..., description="Compound after pit")
    
    class Config:
        use_enum_values = True


class DriverSimulationResult(BaseModel):
    """Simulation result for a single driver."""
    driver_number: int = Field(..., description="Driver number")
    final_position: int = Field(..., ge=1, le=20, description="Final race position")
    total_race_time: float = Field(..., ge=0.0, description="Total race time in seconds")
    laps: List[LapResult] = Field(..., description="Lap-by-lap results")
    stints: List[StintResult] = Field(..., description="Stint summaries")
    pit_stops: List[PitStopInfo] = Field(..., description="Pit stop details")
    win_probability: float = Field(0.0, ge=0.0, le=1.0, description="Win probability (MC)")
    podium_probability: float = Field(0.0, ge=0.0, le=1.0, description="Podium probability (MC)")


class StrategyRanking(BaseModel):
    """Ranking of a strategy option."""
    strategy_id: str = Field(..., description="Strategy identifier")
    expected_position: float = Field(..., ge=1.0, le=20.0, description="Expected finish position")
    win_probability: float = Field(..., ge=0.0, le=1.0, description="Win probability")
    total_time: float = Field(..., ge=0.0, description="Expected total race time")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (1 - CV)")
    risk_score: float = Field(..., ge=0.0, description="Risk score (higher = riskier)")
    pit_laps: List[int] = Field(..., description="Pit stop laps")
    tire_sequence: List[TireCompound] = Field(..., description="Tire compound sequence")
    
    class Config:
        use_enum_values = True


class SimulationOutput(BaseModel):
    """Complete simulation output."""
    race_config: RaceConfig = Field(..., description="Race configuration used")
    results: List[DriverSimulationResult] = Field(..., description="Driver results")
    strategy_rankings: List[StrategyRanking] = Field(
        default_factory=list, description="Strategy rankings"
    )
    optimal_strategy: Optional[StrategyRanking] = Field(None, description="Best strategy found")
    monte_carlo_stats: Optional[Dict[str, Any]] = Field(None, description="MC statistics")
    what_if_analysis: Optional[Dict[str, Any]] = Field(None, description="What-if analysis results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Simulation metadata")
