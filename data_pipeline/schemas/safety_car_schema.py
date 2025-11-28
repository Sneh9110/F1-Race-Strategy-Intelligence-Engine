"""
Safety Car Event Schema - Race interruptions and incidents

Defines schemas for safety car deployments, incidents, and track disruptions.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """Types of race control events."""

    SAFETY_CAR = "SAFETY_CAR"
    VIRTUAL_SAFETY_CAR = "VIRTUAL_SAFETY_CAR"
    RED_FLAG = "RED_FLAG"
    YELLOW_FLAG = "YELLOW_FLAG"
    DOUBLE_YELLOW_FLAG = "DOUBLE_YELLOW_FLAG"


class IncidentType(str, Enum):
    """Types of track incidents."""

    COLLISION = "COLLISION"
    SPIN = "SPIN"
    DEBRIS = "DEBRIS"
    MECHANICAL_FAILURE = "MECHANICAL_FAILURE"
    OFF_TRACK = "OFF_TRACK"
    BARRIER_CONTACT = "BARRIER_CONTACT"
    OTHER = "OTHER"


class IncidentSeverity(str, Enum):
    """Severity classification of incidents."""

    LOW = "LOW"  # Minor, quick recovery
    MEDIUM = "MEDIUM"  # Marshalling required
    HIGH = "HIGH"  # Safety car deployed
    CRITICAL = "CRITICAL"  # Red flag required


class SafetyCarEvent(BaseModel):
    """
    Safety car or track interruption event.

    Represents any race control intervention affecting racing conditions.
    """

    event_id: str = Field(description="Unique event identifier")
    event_type: EventType = Field(description="Type of safety/control event")
    start_lap: int = Field(ge=1, description="Lap when event started")
    end_lap: int = Field(ge=1, description="Lap when event ended")
    start_time: datetime = Field(description="Event start timestamp")
    end_time: datetime = Field(description="Event end timestamp")
    reason: str = Field(description="Reason for event deployment")
    affected_sectors: List[int] = Field(
        default_factory=list, description="Track sectors affected (1, 2, 3)"
    )
    deployment_duration_seconds: float = Field(ge=0.0, description="Total duration in seconds")

    @field_validator("end_lap")
    @classmethod
    def validate_end_after_start(cls, v: int, info) -> int:
        """Validate end lap is after or equal to start lap."""
        start_lap = info.data.get("start_lap")
        if start_lap and v < start_lap:
            raise ValueError(f"End lap ({v}) must be >= start lap ({start_lap})")
        return v

    @field_validator("deployment_duration_seconds")
    @classmethod
    def validate_duration_with_laps(cls, v: float, info) -> float:
        """Validate duration is reasonable for lap count."""
        start_lap = info.data.get("start_lap")
        end_lap = info.data.get("end_lap")

        if start_lap and end_lap:
            lap_count = end_lap - start_lap + 1
            # Assume typical lap time 60-150s
            min_duration = lap_count * 60
            max_duration = lap_count * 150

            if v < min_duration * 0.5 or v > max_duration * 1.5:
                raise ValueError(
                    f"Duration {v}s unrealistic for {lap_count} laps"
                )
        return v

    @field_validator("affected_sectors")
    @classmethod
    def validate_sectors(cls, v: List[int]) -> List[int]:
        """Validate sector numbers are valid."""
        for sector in v:
            if sector not in [1, 2, 3]:
                raise ValueError(f"Invalid sector number: {sector}. Must be 1, 2, or 3")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "2024_MONACO_SC_001",
                "event_type": "SAFETY_CAR",
                "start_lap": 12,
                "end_lap": 15,
                "start_time": "2024-05-26T13:25:30Z",
                "end_time": "2024-05-26T13:31:45Z",
                "reason": "Collision at Turn 4 - debris on track",
                "affected_sectors": [1, 2],
                "deployment_duration_seconds": 375.0,
            }
        }


class IncidentLog(BaseModel):
    """
    Detailed incident information triggering race control action.

    Records what happened, where, and who was involved.
    """

    incident_id: str = Field(description="Unique incident identifier")
    lap_number: int = Field(ge=1, description="Lap when incident occurred")
    turn_number: Optional[int] = Field(None, ge=1, le=25, description="Turn/corner number")
    sector_number: int = Field(ge=1, le=3, description="Track sector (1, 2, or 3)")
    drivers_involved: List[int] = Field(description="Driver numbers involved in incident")
    incident_type: IncidentType = Field(description="Type of incident")
    severity: IncidentSeverity = Field(description="Incident severity classification")
    marshalling_required: bool = Field(description="Whether track marshals needed")
    safety_car_triggered: bool = Field(description="Whether incident triggered safety car")
    description: str = Field(description="Detailed incident description")
    investigation_notes: Optional[str] = Field(None, description="Stewards investigation notes")

    @field_validator("safety_car_triggered")
    @classmethod
    def validate_sc_with_severity(cls, v: bool, info) -> bool:
        """Validate safety car deployment is consistent with severity."""
        severity = info.data.get("severity")

        # High/Critical severity should typically trigger safety car
        if severity in [IncidentSeverity.HIGH, IncidentSeverity.CRITICAL] and not v:
            # Warning but allow it (stewards might decide differently)
            pass

        # Low severity should not trigger safety car
        if severity == IncidentSeverity.LOW and v:
            raise ValueError("Low severity incident should not trigger safety car")

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "incident_id": "2024_MONACO_INC_001",
                "lap_number": 12,
                "turn_number": 4,
                "sector_number": 1,
                "drivers_involved": [14, 31],
                "incident_type": "COLLISION",
                "severity": "HIGH",
                "marshalling_required": True,
                "safety_car_triggered": True,
                "description": "Alonso and Ocon collided at Turn 4 apex. Both cars stopped on track.",
                "investigation_notes": "Racing incident, no further action",
            }
        }


class SafetyCarSession(BaseModel):
    """
    Complete safety car and incident data for a session.

    Aggregates all events and incidents for strategic analysis.
    """

    session_id: str = Field(description="Unique session identifier")
    track_name: str = Field(description="Circuit name")
    events: List[SafetyCarEvent] = Field(default_factory=list, description="All safety car events")
    incidents: List[IncidentLog] = Field(default_factory=list, description="All recorded incidents")
    total_sc_laps: int = Field(default=0, description="Total laps under safety car")
    total_vsc_laps: int = Field(default=0, description="Total laps under VSC")
    total_red_flag_laps: int = Field(default=0, description="Total laps under red flag")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "2024_MONACO_RACE",
                "track_name": "Circuit de Monaco",
                "events": [],
                "incidents": [],
                "total_sc_laps": 4,
                "total_vsc_laps": 2,
                "total_red_flag_laps": 0,
            }
        }


class SafetyCarImpact(BaseModel):
    """
    Strategic impact analysis of safety car deployment.

    Quantifies how safety car affected race outcomes and strategies.
    """

    event_id: str = Field(description="Safety car event identifier")
    lap_time_delta: float = Field(description="Difference vs racing lap (seconds)")
    field_bunching_effect: float = Field(
        ge=0.0, le=1.0, description="How much field compressed (0-1)"
    )
    pit_window_created: bool = Field(description="Whether event created strategic pit opportunity")
    drivers_who_pitted: List[int] = Field(
        default_factory=list, description="Drivers who pitted during event"
    )
    position_changes: int = Field(description="Number of position changes during event")
    strategic_advantage_gained: List[int] = Field(
        default_factory=list, description="Drivers who gained strategic advantage"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "2024_MONACO_SC_001",
                "lap_time_delta": 15.3,
                "field_bunching_effect": 0.75,
                "pit_window_created": True,
                "drivers_who_pitted": [16, 4, 11],
                "position_changes": 5,
                "strategic_advantage_gained": [16, 11],
            }
        }


# Utility functions for safety car data
def calculate_total_neutralized_laps(events: List[SafetyCarEvent]) -> Dict[str, int]:
    """Calculate total laps under each type of neutralization."""
    totals = {
        "SAFETY_CAR": 0,
        "VIRTUAL_SAFETY_CAR": 0,
        "RED_FLAG": 0,
    }

    for event in events:
        if event.event_type in totals:
            lap_count = event.end_lap - event.start_lap + 1
            totals[event.event_type] += lap_count

    return totals


def get_safety_car_probability(track_name: str, historical_events: List[SafetyCarEvent]) -> float:
    """
    Calculate historical safety car probability for a track.

    Returns probability (0-1) based on historical data.
    """
    track_events = [e for e in historical_events if e.event_type == EventType.SAFETY_CAR]

    if not track_events:
        return 0.0

    # Count unique sessions with safety cars
    sessions = set(e.event_id.split("_SC_")[0] for e in track_events)

    # This is simplified - would need total session count for accurate probability
    return min(len(sessions) / 10, 1.0)  # Assume 10 historical sessions


def identify_high_risk_sectors(incidents: List[IncidentLog]) -> List[tuple]:
    """Identify sectors with highest incident frequency."""
    sector_counts = {1: 0, 2: 0, 3: 0}

    for incident in incidents:
        sector_counts[incident.sector_number] += 1

    # Sort by count, descending
    sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_sectors
