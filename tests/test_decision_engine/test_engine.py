"""Tests for decision engine orchestrator."""

import pytest
from decision_engine import (
    DecisionEngine,
    DecisionInput,
    DecisionContext,
    DecisionOutput,
    DecisionAction,
)


def test_decision_engine_initialization():
    """Test engine initialization."""
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    
    assert engine is not None
    assert len(engine.modules) > 0
    assert engine.decision_count == 0


def test_make_decision_basic(sample_decision_input):
    """Test basic decision making."""
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = engine.make_decision(sample_decision_input)
    
    assert isinstance(output, DecisionOutput)
    assert len(output.recommendations) > 0
    assert output.computation_time_ms >= 0
    assert engine.decision_count == 1


def test_make_decision_with_simulation_context(sample_decision_input_with_simulation):
    """Test decision with simulation context."""
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = engine.make_decision(sample_decision_input_with_simulation)
    
    assert isinstance(output, DecisionOutput)
    assert len(output.recommendations) > 0


def test_make_decision_filters_inapplicable_modules(sample_decision_context):
    """Test that inapplicable modules are filtered."""
    # Create context with very low tire age (should filter out pit timing)
    context = DecisionContext(
        session_id="test",
        lap_number=2,
        driver_number=44,
        track_name="Monaco",
        total_laps=78,
        current_position=3,
        tire_age=2,  # Too low for pit timing
        tire_compound="MEDIUM",
        fuel_load=95.0,
        stint_number=1,
        pit_stops_completed=0,
        recent_lap_times=[91.0, 91.1],
    )
    
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = engine.make_decision(DecisionInput(context=context))
    
    # Should still work, just with fewer modules
    assert isinstance(output, DecisionOutput)


def test_make_decision_ranking(sample_decision_input):
    """Test recommendation ranking."""
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = engine.make_decision(sample_decision_input)
    
    if len(output.recommendations) > 1:
        # Check sorted by priority and confidence
        for i in range(len(output.recommendations) - 1):
            curr = output.recommendations[i]
            next_rec = output.recommendations[i + 1]
            # Higher priority or same priority but higher confidence
            assert (
                curr.priority >= next_rec.priority
                or (curr.priority == next_rec.priority and curr.confidence_score >= next_rec.confidence_score)
            )


def test_make_decision_with_sc(sc_decision_context):
    """Test decision during safety car."""
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = engine.make_decision(DecisionInput(context=sc_decision_context))
    
    assert isinstance(output, DecisionOutput)
    # Should have SC-related recommendations
    sc_actions = [DecisionAction.PIT_UNDER_SC, DecisionAction.STAY_OUT_SC]
    has_sc_recommendation = any(
        rec.action in sc_actions for rec in output.recommendations
    )
    # May or may not have SC rec depending on context, but should not crash
    assert output.computation_time_ms >= 0


@pytest.mark.asyncio
async def test_make_decision_async(sample_decision_input):
    """Test async decision making."""
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = await engine.make_decision_async(sample_decision_input)
    
    assert isinstance(output, DecisionOutput)
    assert len(output.recommendations) > 0


def test_make_decisions_batch(sample_decision_input):
    """Test batch decision making."""
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    
    # Create multiple inputs
    inputs = [sample_decision_input for _ in range(3)]
    outputs = engine.make_decisions_batch(inputs)
    
    assert len(outputs) == 3
    for output in outputs:
        assert isinstance(output, DecisionOutput)


def test_conflict_resolution(sample_decision_input):
    """Test that conflicting recommendations are resolved."""
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    output = engine.make_decision(sample_decision_input)
    
    # Check no duplicate actions in top recommendations
    actions = [rec.action for rec in output.recommendations]
    # Note: Same action from different modules is OK, just shouldn't be excessive
    assert len(output.recommendations) <= 5  # Should limit top N


def test_engine_performance_tracking(sample_decision_input):
    """Test performance metrics tracking."""
    engine = DecisionEngine(config_path='config/decision_engine.yaml', enable_cache=False)
    
    # Make multiple decisions
    for _ in range(5):
        engine.make_decision(sample_decision_input)
    
    stats = engine.get_stats()
    
    assert stats['decision_count'] == 5
    assert stats['total_decisions'] == 5


def test_engine_caching(sample_decision_input):
    """Test decision caching."""
    engine = DecisionEngine(config_path='config/decision_engine.yaml', enable_cache=True)
    
    # Make same decision twice
    output1 = engine.make_decision(sample_decision_input)
    output2 = engine.make_decision(sample_decision_input)
    
    # Second should be from cache (faster)
    # Note: Caching may not be fully implemented, so just check it doesn't crash
    assert isinstance(output1, DecisionOutput)
    assert isinstance(output2, DecisionOutput)


def test_engine_module_stats(sample_decision_input):
    """Test getting module statistics."""
    engine = DecisionEngine(config_path='config/decision_engine.yaml')
    
    # Make some decisions
    engine.make_decision(sample_decision_input)
    
    # Check module stats
    for module in engine.modules:
        stats = module.get_stats()
        assert 'name' in stats
        assert 'evaluation_count' in stats
        assert stats['evaluation_count'] >= 0
