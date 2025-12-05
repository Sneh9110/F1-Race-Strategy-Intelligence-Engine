"""Pytest fixtures for decision engine tests."""

import pytest
from datetime import datetime
from typing import Dict, Any

from decision_engine.schemas import (
    DecisionContext,
    SimulationContext,
    RivalContext,
    DecisionInput,
)


@pytest.fixture
def sample_decision_context() -> DecisionContext:
    """Basic decision context."""
    return DecisionContext(
        session_id='test_session',
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
        gap_to_ahead=2.5,
        gap_to_behind=3.2,
        safety_car_active=False,
        weather_temp=25.0,
        track_temp=35.0,
        recent_lap_times=[90.5, 90.8, 91.2, 91.5, 91.8],
    )


@pytest.fixture
def sample_simulation_context() -> SimulationContext:
    """Simulation context with strategy rankings."""
    return SimulationContext(
        optimal_strategy={
            'pit_laps': [35],
            'tire_sequence': ['MEDIUM', 'SOFT'],
        },
        strategy_rankings=[
            {
                'strategy_id': 'strat_1_stop',
                'expected_position': 2.5,
                'win_probability': 0.15,
                'pit_laps': [35],
            },
            {
                'strategy_id': 'strat_2_stop',
                'expected_position': 3.2,
                'win_probability': 0.10,
                'pit_laps': [20, 45],
            },
        ],
        win_probability=0.15,
        expected_position=2.5,
    )


@pytest.fixture
def sample_rival_context() -> RivalContext:
    """Rival context for undercut/overcut tests."""
    return RivalContext(
        rival_driver_number=1,
        rival_position=2,
        rival_tire_compound='SOFT',
        rival_tire_age=25,
        rival_pit_stops=1,
        gap_to_rival=2.3,
    )


@pytest.fixture
def sample_decision_input(
    sample_decision_context,
    sample_simulation_context,
    sample_rival_context
) -> DecisionInput:
    """Complete decision input."""
    return DecisionInput(
        context=sample_decision_context,
        simulation_context=sample_simulation_context,
        rival_contexts=[sample_rival_context],
        feature_data={},
    )


@pytest.fixture
def sample_decision_input_with_simulation(
    sample_decision_context,
    sample_simulation_context
) -> DecisionInput:
    """Decision input with simulation context."""
    return DecisionInput(
        context=sample_decision_context,
        simulation_context=sample_simulation_context,
        rival_contexts=[],
        feature_data={},
    )


@pytest.fixture
def sc_decision_context(sample_decision_context) -> DecisionContext:
    """Context with safety car active."""
    context = sample_decision_context.copy()
    context.safety_car_active = True
    context.lap_number = 30
    context.tire_age = 18
    return context


@pytest.fixture
def tire_cliff_context(sample_decision_context) -> DecisionContext:
    """Context with tire cliff."""
    context = sample_decision_context.copy()
    context.tire_age = 28
    context.recent_lap_times = [90.5, 90.8, 91.2, 92.5, 94.8]  # Sudden drop
    return context


@pytest.fixture
def decision_engine_config() -> str:
    """Test config path."""
    return 'config/decision_engine.yaml'


def create_decision_context(**kwargs) -> DecisionContext:
    """Factory for custom decision context."""
    defaults = {
        'session_id': 'test',
        'lap_number': 25,
        'driver_number': 44,
        'track_name': 'Monaco',
        'total_laps': 78,
        'current_position': 3,
        'tire_age': 20,
        'tire_compound': 'MEDIUM',
        'fuel_load': 85.0,
        'stint_number': 2,
        'pit_stops_completed': 1,
        'recent_lap_times': [90.5, 90.8, 91.2],
    }
    defaults.update(kwargs)
    return DecisionContext(**defaults)


def assert_recommendation(rec, action, min_confidence=0.0):
    """Assert recommendation properties."""
    assert rec is not None
    assert rec.action == action
    assert rec.confidence_score >= min_confidence
    assert 0.0 <= rec.risk_score <= 1.0
    assert 1 <= rec.priority <= 10
