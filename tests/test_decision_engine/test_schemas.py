"""Tests for decision engine schemas."""

import pytest
from pydantic import ValidationError

from decision_engine.schemas import (
    DecisionContext,
    DecisionInput,
    DecisionRecommendation,
    DecisionOutput,
    DecisionAction,
    DecisionCategory,
    ConfidenceLevel,
    TrafficLight,
)


def test_decision_context_valid():
    """Test valid decision context creation."""
    context = DecisionContext(
        session_id='test',
        lap_number=25,
        driver_number=44,
        track_name='Monaco',
        total_laps=78,
        current_position=3,
        tire_age=20,
        tire_compound='MEDIUM',
        fuel_load=85.0,
        stint_number=2,
        pit_stops_completed=1,
    )
    
    assert context.session_id == 'test'
    assert context.lap_number == 25


def test_decision_context_invalid_lap():
    """Test negative lap number raises error."""
    with pytest.raises(ValidationError):
        DecisionContext(
            session_id='test',
            lap_number=-1,  # Invalid
            driver_number=44,
            track_name='Monaco',
            total_laps=78,
            current_position=3,
            tire_age=20,
            tire_compound='MEDIUM',
            fuel_load=85.0,
            stint_number=2,
            pit_stops_completed=1,
        )


def test_decision_context_invalid_position():
    """Test position > 20 raises error."""
    with pytest.raises(ValidationError):
        DecisionContext(
            session_id='test',
            lap_number=25,
            driver_number=44,
            track_name='Monaco',
            total_laps=78,
            current_position=21,  # Invalid
            tire_age=20,
            tire_compound='MEDIUM',
            fuel_load=85.0,
            stint_number=2,
            pit_stops_completed=1,
        )


def test_decision_input_valid(sample_decision_input):
    """Test valid decision input."""
    assert sample_decision_input.context is not None
    assert sample_decision_input.simulation_context is not None


def test_decision_recommendation_valid():
    """Test valid recommendation creation."""
    from decision_engine.schemas import DecisionReasoning
    
    rec = DecisionRecommendation(
        action=DecisionAction.PIT_NOW,
        category=DecisionCategory.PIT_TIMING,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.85,
        traffic_light=TrafficLight.GREEN,
        reasoning=DecisionReasoning(),
        expected_gain_seconds=3.5,
        risk_score=0.3,
        priority=9,
    )
    
    assert rec.action == DecisionAction.PIT_NOW
    assert rec.confidence_score == 0.85


def test_decision_output_valid():
    """Test valid decision output."""
    output = DecisionOutput(
        recommendations=[],
        session_id='test',
        lap_number=25,
        computation_time_ms=150.0,
    )
    
    assert output.session_id == 'test'
    assert output.computation_time_ms == 150.0


def test_enums():
    """Test enum values."""
    assert DecisionAction.PIT_NOW.value == 'pit_now'
    assert ConfidenceLevel.HIGH.value == 'high'
    assert TrafficLight.GREEN.value == 'green'
    assert DecisionCategory.PIT_TIMING.value == 'pit_timing'
