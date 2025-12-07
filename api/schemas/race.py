"""Race state schemas for HTTP and WebSocket communication."""

from typing import Optional, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime

from api.schemas.common import MetadataBase


# Race State Schemas
class DriverState(BaseModel):
    """Individual driver state during race."""
    
    driver_number: int
    driver_name: str
    team: str
    position: int
    tire_compound: str
    tire_age: int
    pit_stops: int
    gap_to_leader: Optional[float] = None
    gap_to_next: Optional[float] = None
    last_lap_time: Optional[float] = None
    status: str = "RACING"  # RACING, PIT, OUT, DNF


class RaceState(BaseModel):
    """Current race state."""
    
    session_id: str
    circuit_name: str
    session_type: str = "RACE"
    current_lap: int
    total_laps: int
    safety_car_deployed: bool = False
    weather_condition: str = "Dry"
    track_temp: float
    air_temp: float
    drivers: list[DriverState]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RaceStateRequest(BaseModel):
    """Request to create/update race state."""
    
    session_id: str
    circuit_name: str
    session_type: str = "RACE"
    current_lap: int
    total_laps: int
    safety_car_deployed: bool = False
    weather_condition: str = "Dry"
    track_temp: float = 25.0
    air_temp: float = 20.0
    drivers: list[DriverState]


class RaceStateResponse(BaseModel):
    """Response with race state."""
    
    session_id: str
    circuit_name: str
    session_type: str
    current_lap: int
    total_laps: int
    safety_car_deployed: bool
    weather_condition: str
    track_temp: float
    air_temp: float
    drivers: list[DriverState]
    timestamp: datetime
    metadata: MetadataBase


# WebSocket Message Types
class LapCompletedData(BaseModel):
    """Lap completed event data."""
    driver_number: int
    lap: int
    lap_time: float
    tire_compound: str
    tire_age: int


class PitStopData(BaseModel):
    """Pit stop event data."""
    driver_number: int
    lap: int
    pit_duration: float
    tire_compound_in: str
    tire_compound_out: str


class SafetyCarData(BaseModel):
    """Safety car event data."""
    deployed: bool
    lap: int
    reason: Optional[str] = None


class PositionChangeData(BaseModel):
    """Position change event data."""
    driver_number: int
    old_position: int
    new_position: int
    lap: int


class WebSocketMessage(BaseModel):
    """WebSocket message wrapper."""
    
    type: Literal["LAP_COMPLETED", "PIT_STOP", "SAFETY_CAR", "POSITION_CHANGE", "RACE_STATE_UPDATE"]
    data: Union[LapCompletedData, PitStopData, SafetyCarData, PositionChangeData, RaceState]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
