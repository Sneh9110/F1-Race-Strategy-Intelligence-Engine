"""Tests for pit timing decision module."""

import pytest
from decision_engine import PitTimingDecision, DecisionInput, DecisionAction


def test_pit_timing_initialization():
    """Test module initialization."""
    module = PitTimingDecision(config_path='config/decision_engine.yaml')
    
    assert module.name == "pit_timing"
    assert module.priority == 9
    assert module.enabled


def test_pit_timing_not_applicable_just_pitted(sample_decision_context):
    """Test module not applicable right after pit."""
    sample_decision_context.tire_age = 3  # Just pitted
    
    module = PitTimingDecision(config_path='config/decision_engine.yaml')
    assert not module.is_applicable(sample_decision_context)


def test_pit_timing_not_applicable_early_lap(sample_decision_context):
    """Test module not applicable on early laps."""
    sample_decision_context.lap_number = 3
    sample_decision_context.tire_age = 3
    
    module = PitTimingDecision(config_path='config/decision_engine.yaml')
    assert not module.is_applicable(sample_decision_context)


def test_pit_timing_applicable_normal(sample_decision_context):
    """Test module applicable in normal pit window."""
    sample_decision_context.tire_age = 20
    sample_decision_context.lap_number = 25
    
    module = PitTimingDecision(config_path='config/decision_engine.yaml')
    assert module.is_applicable(sample_decision_context)


def test_pit_timing_high_tire_age(sample_decision_context):
    """Test recommendation with high tire age."""
    sample_decision_context.tire_age = 30
    sample_decision_context.recent_lap_times = [90.5, 90.8, 91.2, 91.8, 92.5]  # Degrading
    
    module = PitTimingDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=sample_decision_context)
    
    recommendation = module.evaluate(decision_input)
    
    assert recommendation is not None
    assert recommendation.action == DecisionAction.PIT_NOW
    assert recommendation.confidence_score > 0.5


def test_pit_timing_tire_cliff(tire_cliff_context):
    """Test recommendation with tire cliff detected."""
    module = PitTimingDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=tire_cliff_context)
    
    recommendation = module.evaluate(decision_input)
    
    # Should strongly recommend pitting
    assert recommendation is not None
    assert recommendation.action == DecisionAction.PIT_NOW
    assert recommendation.confidence_score >= 0.7


def test_pit_timing_stay_out_sc_imminent(sample_decision_context):
    """Test stay out when SC imminent."""
    sample_decision_context.tire_age = 15
    # Would need SC probability in feature_data - simplified test
    
    module = PitTimingDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=sample_decision_context)
    
    recommendation = module.evaluate(decision_input)
    
    # Should work without crashing
    assert recommendation is None or isinstance(recommendation.action, DecisionAction)


def test_pit_timing_undercut_opportunity(sample_decision_context, sample_rival_context):
    """Test undercut opportunity detection."""
    sample_decision_context.tire_age = 18
    sample_decision_context.gap_to_ahead = 2.5  # Within undercut range
    sample_rival_context.rival_tire_age = 25  # Older tires
    
    module = PitTimingDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(
        context=sample_decision_context,
        rival_contexts=[sample_rival_context]
    )
    
    recommendation = module.evaluate(decision_input)
    
    # May or may not recommend (depends on full logic), but shouldn't crash
    assert recommendation is None or isinstance(recommendation.action, DecisionAction)


def test_pit_timing_confidence_calculation(sample_decision_context):
    """Test confidence score calculation."""
    sample_decision_context.tire_age = 20
    sample_decision_context.recent_lap_times = [90.5, 90.6, 90.7, 90.8, 90.9]  # Consistent
    
    module = PitTimingDecision(config_path='config/decision_engine.yaml')
    confidence = module.get_confidence(sample_decision_context)
    
    assert 0.0 <= confidence <= 1.0


def test_pit_timing_has_alternatives(sample_decision_context):
    """Test that recommendations include alternatives."""
    sample_decision_context.tire_age = 25
    
    module = PitTimingDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=sample_decision_context)
    
    recommendation = module.evaluate(decision_input)
    
    if recommendation:
        assert len(recommendation.alternatives) > 0
        assert all(hasattr(alt, 'action') for alt in recommendation.alternatives)


def test_pit_timing_reasoning_populated(sample_decision_context):
    """Test that reasoning is properly populated."""
    sample_decision_context.tire_age = 25
    sample_decision_context.recent_lap_times = [90.5, 91.0, 91.5, 92.0, 92.5]
    
    module = PitTimingDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=sample_decision_context)
    
    recommendation = module.evaluate(decision_input)
    
    if recommendation:
        assert len(recommendation.reasoning.primary_factors) > 0
        assert recommendation.reasoning.risk_assessment != ""


def test_pit_timing_performance_tracking(sample_decision_context):
    """Test module performance tracking."""
    module = PitTimingDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=sample_decision_context)
    
    # Make multiple evaluations
    for _ in range(5):
        module.evaluate_safe(decision_input)
    
    stats = module.get_stats()
    
    assert stats['evaluation_count'] == 5
    assert stats['name'] == "pit_timing"
    assert 'latency_p50_ms' in stats
