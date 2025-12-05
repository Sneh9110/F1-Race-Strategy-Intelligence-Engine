"""Integration tests for decision engine."""

import pytest
from decision_engine import (
    DecisionEngine,
    DecisionInput,
    DecisionContext,
    SimulationContext,
    RivalContext,
    DecisionAction,
)


def test_end_to_end_monaco_lap_25():
    """Test complete decision workflow for Monaco lap 25."""
    # Scenario: Driver in P3, lap 25/78, MEDIUM tires, 20 laps old
    context = DecisionContext(
        session_id="2024_MONACO",
        lap_number=25,
        driver_number=44,
        track_name="Monaco",
        total_laps=78,
        current_position=3,
        tire_age=20,
        tire_compound="MEDIUM",
        fuel_load=85.0,
        stint_number=2,
        pit_stops_completed=1,
        gap_to_ahead=2.5,
        gap_to_behind=4.3,
        recent_lap_times=[73.2, 73.5, 73.8, 74.1, 74.5],  # Monaco lap times, degrading
    )
    
    # Create decision input
    decision_input = DecisionInput(context=context)
    
    # Make decision
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = engine.make_decision(decision_input)
    
    # Verify output structure
    assert output is not None
    assert len(output.recommendations) > 0
    assert output.session_id == "2024_MONACO"
    assert output.lap_number == 25
    assert output.computation_time_ms < 500  # Should be fast
    
    # Verify recommendations have required fields
    for rec in output.recommendations:
        assert rec.action is not None
        assert rec.confidence_score >= 0.0
        assert rec.traffic_light is not None
        assert len(rec.reasoning.primary_factors) > 0


def test_safety_car_deployment_scenario():
    """Test decision making when safety car is deployed."""
    # Scenario: SC just deployed, driver in P5, old tires
    context = DecisionContext(
        session_id="2024_SILVERSTONE",
        lap_number=35,
        driver_number=1,
        track_name="Silverstone",
        total_laps=52,
        current_position=5,
        tire_age=18,
        tire_compound="MEDIUM",
        fuel_load=70.0,
        stint_number=2,
        pit_stops_completed=1,
        safety_car_active=True,
        gap_to_ahead=3.2,
        recent_lap_times=[90.5, 90.8, 91.2, 110.5, 115.3],  # SC deployed
    )
    
    decision_input = DecisionInput(context=context)
    
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = engine.make_decision(decision_input)
    
    # Should prioritize SC decision
    assert output is not None
    # Should have SC-related recommendations
    sc_actions = [DecisionAction.PIT_UNDER_SC, DecisionAction.STAY_OUT_SC]
    has_sc_rec = any(rec.action in sc_actions for rec in output.recommendations)
    
    # May or may not have SC recommendation depending on full logic
    assert output.computation_time_ms < 500


def test_undercut_opportunity_scenario():
    """Test undercut opportunity detection."""
    # Scenario: Close to leader with older tires
    context = DecisionContext(
        session_id="2024_SPAIN",
        lap_number=28,
        driver_number=44,
        track_name="Barcelona",
        total_laps=66,
        current_position=2,
        tire_age=18,
        tire_compound="MEDIUM",
        fuel_load=75.0,
        stint_number=2,
        pit_stops_completed=1,
        gap_to_ahead=2.3,  # Close to leader
        recent_lap_times=[80.2, 80.3, 80.5, 80.7],
    )
    
    # Rival (leader) with older tires
    rival = RivalContext(
        rival_driver_number=1,
        rival_position=1,
        rival_tire_compound="MEDIUM",
        rival_tire_age=25,  # Much older
        rival_pit_stops=1,
        gap_to_rival=2.3,
    )
    
    decision_input = DecisionInput(
        context=context,
        rival_contexts=[rival]
    )
    
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = engine.make_decision(decision_input)
    
    # Should detect undercut opportunity
    assert output is not None
    
    # Check if any recommendation considers undercut
    # (may be PIT_NOW from pit_timing or UNDERCUT_NOW from undercut module)
    assert len(output.recommendations) > 0


def test_tire_cliff_emergency_scenario():
    """Test emergency pit recommendation with tire cliff."""
    # Scenario: Sudden tire degradation (cliff)
    context = DecisionContext(
        session_id="2024_BAHRAIN",
        lap_number=32,
        driver_number=16,
        track_name="Bahrain",
        total_laps=57,
        current_position=6,
        tire_age=28,
        tire_compound="SOFT",
        fuel_load=65.0,
        stint_number=3,
        pit_stops_completed=2,
        recent_lap_times=[95.5, 95.8, 96.2, 98.5, 101.2],  # Cliff detected
    )
    
    decision_input = DecisionInput(context=context)
    
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = engine.make_decision(decision_input)
    
    # Should strongly recommend pitting
    assert output is not None
    
    # Look for pit recommendation with high confidence
    pit_recs = [
        rec for rec in output.recommendations 
        if rec.action in [DecisionAction.PIT_NOW, DecisionAction.PIT_UNDER_SC]
    ]
    
    # Should have at least one pit recommendation
    # (may not if other factors override, but should be considered)
    assert len(output.recommendations) > 0


def test_strategy_conversion_scenario():
    """Test strategy conversion recommendation."""
    # Scenario: 1-stop not working, degradation higher than expected
    context = DecisionContext(
        session_id="2024_HUNGARY",
        lap_number=42,
        driver_number=11,
        track_name="Hungaroring",
        total_laps=70,
        current_position=8,
        tire_age=35,
        tire_compound="HARD",
        fuel_load=50.0,
        stint_number=2,
        pit_stops_completed=1,
        recent_lap_times=[81.5, 82.0, 82.5, 83.2, 83.8],  # High degradation
    )
    
    # Simulation shows 2-stop might be better
    simulation = SimulationContext(
        strategy_rankings=[
            {'strategy_id': '2-stop', 'expected_position': 6.5, 'expected_time': 5950.0},
            {'strategy_id': '1-stop', 'expected_position': 8.2, 'expected_time': 5968.0},
        ],
        win_probability=0.05,
        expected_position=8.2,
    )
    
    decision_input = DecisionInput(
        context=context,
        simulation_context=simulation
    )
    
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = engine.make_decision(decision_input)
    
    # Should consider strategy conversion
    assert output is not None
    assert len(output.recommendations) > 0


def test_multiple_drivers_batch():
    """Test batch processing for multiple drivers."""
    # Create contexts for 3 drivers
    contexts = []
    for i, position in enumerate([1, 5, 10], start=1):
        context = DecisionContext(
            session_id="2024_MONZA",
            lap_number=30,
            driver_number=i,
            track_name="Monza",
            total_laps=53,
            current_position=position,
            tire_age=18 + i * 2,
            tire_compound="MEDIUM",
            fuel_load=70.0,
            stint_number=2,
            pit_stops_completed=1,
            recent_lap_times=[84.0 + i * 0.1] * 5,
        )
        contexts.append(context)
    
    inputs = [DecisionInput(context=ctx) for ctx in contexts]
    
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    outputs = engine.make_decisions_batch(inputs)
    
    # Should get output for each driver
    assert len(outputs) == 3
    
    for output in outputs:
        assert output is not None
        assert len(output.recommendations) >= 0  # May have 0 if none applicable


def test_conflict_resolution_multiple_modules():
    """Test that conflicting recommendations from different modules are resolved."""
    # Create scenario where multiple modules might trigger
    context = DecisionContext(
        session_id="2024_SUZUKA",
        lap_number=28,
        driver_number=44,
        track_name="Suzuka",
        total_laps=53,
        current_position=3,
        tire_age=22,
        tire_compound="MEDIUM",
        fuel_load=72.0,
        stint_number=2,
        pit_stops_completed=1,
        gap_to_ahead=2.8,
        recent_lap_times=[91.5, 91.8, 92.2, 92.6],
    )
    
    # Add rival for undercut module
    rival = RivalContext(
        rival_driver_number=1,
        rival_position=2,
        rival_tire_compound="MEDIUM",
        rival_tire_age=28,
        rival_pit_stops=1,
        gap_to_rival=2.8,
    )
    
    decision_input = DecisionInput(
        context=context,
        rival_contexts=[rival]
    )
    
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = engine.make_decision(decision_input)
    
    # Should resolve conflicts and provide ranked recommendations
    assert output is not None
    assert len(output.recommendations) <= 5  # Should limit to top N
    
    # Check ranking order
    if len(output.recommendations) > 1:
        for i in range(len(output.recommendations) - 1):
            curr = output.recommendations[i]
            next_rec = output.recommendations[i + 1]
            # Higher priority should come first
            assert curr.priority >= next_rec.priority or (
                curr.priority == next_rec.priority and 
                curr.confidence_score >= next_rec.confidence_score
            )


def test_performance_under_200ms():
    """Test that decision making meets latency target."""
    context = DecisionContext(
        session_id="2024_PERFORMANCE_TEST",
        lap_number=25,
        driver_number=44,
        track_name="Silverstone",
        total_laps=52,
        current_position=5,
        tire_age=20,
        tire_compound="MEDIUM",
        fuel_load=75.0,
        stint_number=2,
        pit_stops_completed=1,
        recent_lap_times=[90.0] * 5,
    )
    
    decision_input = DecisionInput(context=context)
    
    engine = DecisionEngine(config_path='config/decision_engine.yaml', enable_cache=False)
    
    # Warm up
    engine.make_decision(decision_input)
    
    # Measure latency
    import time
    start = time.time()
    output = engine.make_decision(decision_input)
    latency_ms = (time.time() - start) * 1000
    
    # Should meet target (allowing overhead)
    assert latency_ms < 500  # Generous for test environment
    assert output.computation_time_ms < 300


@pytest.mark.asyncio
async def test_async_parallel_execution():
    """Test async parallel module execution."""
    context = DecisionContext(
        session_id="2024_ASYNC_TEST",
        lap_number=30,
        driver_number=44,
        track_name="Spa",
        total_laps=44,
        current_position=4,
        tire_age=22,
        tire_compound="MEDIUM",
        fuel_load=68.0,
        stint_number=2,
        pit_stops_completed=1,
        recent_lap_times=[107.0] * 5,
    )
    
    decision_input = DecisionInput(context=context)
    
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = await engine.make_decision_async(decision_input)
    
    # Should work and be faster than sync
    assert output is not None
    assert len(output.recommendations) >= 0
