"""Pydantic schemas for Decision Engine I/O."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator, model_validator


class DecisionAction(str, Enum):
    """Possible decision actions."""
    PIT_NOW = "pit_now"
    STAY_OUT = "stay_out"
    SWITCH_TO_ONE_STOP = "switch_to_one_stop"
    SWITCH_TO_TWO_STOP = "switch_to_two_stop"
    SWITCH_TO_THREE_STOP = "switch_to_three_stop"
    OFFSET_STRATEGY = "offset_strategy"
    PIT_UNDER_SC = "pit_under_sc"
    STAY_OUT_SC = "stay_out_sc"
    SWITCH_TO_INTERS = "switch_to_inters"
    SWITCH_TO_WETS = "switch_to_wets"
    AGGRESSIVE_PACE = "aggressive_pace"
    CONSERVATIVE_PACE = "conservative_pace"
    UNDERCUT_NOW = "undercut_now"
    OVERCUT_NOW = "overcut_now"
    NO_ACTION = "no_action"


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TrafficLight(str, Enum):
    """Visual recommendation status."""
    RED = "red"
    AMBER = "amber"
    GREEN = "green"


class DecisionCategory(str, Enum):
    """Category of decision."""
    PIT_TIMING = "pit_timing"
    STRATEGY_CONVERSION = "strategy_conversion"
    OFFSET_STRATEGY = "offset_strategy"
    SAFETY_CAR = "safety_car"
    RAIN = "rain"
    PACE_ADJUSTMENT = "pace_adjustment"
    UNDERCUT_OVERCUT = "undercut_overcut"


class DecisionContext(BaseModel):
    """Current race state for decision making."""
    session_id: str
    lap_number: int = Field(ge=1)
    driver_number: int = Field(ge=1)
    track_name: str
    total_laps: int = Field(ge=1)
    
    current_position: int = Field(ge=1, le=20)
    tire_age: int = Field(ge=0)
    tire_compound: str
    fuel_load: float = Field(ge=0.0)
    stint_number: int = Field(ge=1)
    pit_stops_completed: int = Field(ge=0)
    
    gap_to_ahead: Optional[float] = Field(default=None, ge=0.0)
    gap_to_behind: Optional[float] = Field(default=None, ge=0.0)
    
    safety_car_active: bool = False
    weather_temp: Optional[float] = None
    track_temp: Optional[float] = None
    
    recent_lap_times: List[float] = Field(default_factory=list)
    
    @validator('recent_lap_times')
    def validate_lap_times(cls, v):
        """Validate lap times are positive."""
        if any(lt <= 0 for lt in v):
            raise ValueError("Lap times must be positive")
        return v


class SimulationContext(BaseModel):
    """Simulation outputs for decision making."""
    optimal_strategy: Optional[Dict[str, Any]] = None
    strategy_rankings: List[Dict[str, Any]] = Field(default_factory=list)
    win_probability: float = Field(ge=0.0, le=1.0)
    expected_position: float = Field(ge=1.0, le=20.0)
    monte_carlo_stats: Optional[Dict[str, Any]] = None
    what_if_analysis: Optional[Dict[str, Any]] = None


class RivalContext(BaseModel):
    """Competitor state for decision making."""
    rival_driver_number: int = Field(ge=1)
    rival_position: int = Field(ge=1, le=20)
    rival_tire_compound: str
    rival_tire_age: int = Field(ge=0)
    rival_pit_stops: int = Field(ge=0)
    gap_to_rival: float = Field(ge=0.0)


class DecisionInput(BaseModel):
    """Complete input for decision engine."""
    context: DecisionContext
    simulation_context: Optional[SimulationContext] = None
    rival_contexts: List[RivalContext] = Field(default_factory=list)
    feature_data: Dict[str, Any] = Field(default_factory=dict)


class AlternativeOption(BaseModel):
    """Alternative action with expected outcome."""
    action: DecisionAction
    expected_outcome: str
    confidence: float = Field(ge=0.0, le=1.0)
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)


class DecisionReasoning(BaseModel):
    """Explainability for decision."""
    primary_factors: List[str] = Field(default_factory=list)
    rule_triggers: List[str] = Field(default_factory=list)
    model_contributions: Dict[str, float] = Field(default_factory=dict)
    risk_assessment: str = ""
    opportunity_assessment: str = ""


class DecisionRecommendation(BaseModel):
    """Main decision recommendation output."""
    action: DecisionAction
    category: DecisionCategory
    confidence: ConfidenceLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    traffic_light: TrafficLight
    
    reasoning: DecisionReasoning
    alternatives: List[AlternativeOption] = Field(default_factory=list)
    
    expected_gain_seconds: float = Field(ge=-30.0, le=30.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    priority: int = Field(ge=1, le=10)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('expected_gain_seconds')
    def validate_expected_gain(cls, v):
        """Validate expected gain is realistic."""
        if abs(v) > 30:
            raise ValueError("Expected gain must be between -30 and +30 seconds")
        return v
    
    @model_validator(mode='after')
    def validate_sc_actions(self):
        """Validate safety car actions require SC active."""
        if self.action in [DecisionAction.PIT_UNDER_SC, DecisionAction.STAY_OUT_SC]:
            # Note: Can't access context here, validation done in module
            pass
        return self


class DecisionOutput(BaseModel):
    """Batch decision output."""
    recommendations: List[DecisionRecommendation] = Field(default_factory=list)
    session_id: str
    lap_number: int = Field(ge=1)
    computation_time_ms: float = Field(ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
