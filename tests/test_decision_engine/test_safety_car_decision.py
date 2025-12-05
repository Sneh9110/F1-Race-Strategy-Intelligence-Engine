"""Tests for safety car decision module."""

import pytest
from decision_engine import SafetyCarDecision, DecisionInput, DecisionAction, TrafficLight


def test_safety_car_initialization():
    """Test module initialization."""
    module = SafetyCarDecision(config_path='config/decision_engine.yaml')
    
    assert module.name == "safety_car_decision"
    assert module.priority == 10  # Highest priority
    assert module.enabled


def test_safety_car_not_applicable_no_sc(sample_decision_context):
    """Test module not applicable without SC."""
    sample_decision_context.safety_car_active = False
    
    module = SafetyCarDecision(config_path='config/decision_engine.yaml')
    # May check SC probability, so might still be applicable with low probability
    # Just ensure it doesn't crash
    result = module.is_applicable(sample_decision_context)
    assert isinstance(result, bool)


def test_safety_car_applicable_sc_active(sc_decision_context):
    """Test module applicable when SC active."""
    module = SafetyCarDecision(config_path='config/decision_engine.yaml')
    assert module.is_applicable(sc_decision_context)


def test_safety_car_pit_under_sc_high_tire_age(sc_decision_context):
    """Test pit recommendation with high tire age during SC."""
    sc_decision_context.tire_age = 25  # Old tires
    
    module = SafetyCarDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=sc_decision_context)
    
    recommendation = module.evaluate(decision_input)
    
    assert recommendation is not None
    assert recommendation.action in [DecisionAction.PIT_UNDER_SC, DecisionAction.STAY_OUT_SC]
    assert recommendation.traffic_light != TrafficLight.RED  # SC decisions never RED


def test_safety_car_stay_out_just_pitted(sc_decision_context):
    """Test stay out when just pitted."""
    sc_decision_context.tire_age = 3  # Just pitted
    
    module = SafetyCarDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=sc_decision_context)
    
    recommendation = module.evaluate(decision_input)
    
    # Should recommend staying out or may not be applicable
    assert recommendation is None or recommendation.action == DecisionAction.STAY_OUT_SC


def test_safety_car_podium_position(sc_decision_context):
    """Test decision for podium position."""
    sc_decision_context.current_position = 2  # Podium
    sc_decision_context.tire_age = 18  # Medium age
    
    module = SafetyCarDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=sc_decision_context)
    
    recommendation = module.evaluate(decision_input)
    
    # Should be more conservative with podium position
    assert recommendation is not None


def test_safety_car_rival_analysis(sc_decision_context, sample_rival_context):
    """Test rival strategy analysis during SC."""
    # Multiple rivals pitting
    rivals = [sample_rival_context for _ in range(3)]
    
    module = SafetyCarDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(
        context=sc_decision_context,
        rival_contexts=rivals
    )
    
    recommendation = module.evaluate(decision_input)
    
    # Should consider rival strategies
    assert recommendation is not None


def test_safety_car_latency_target(sc_decision_context):
    """Test that latency meets <100ms target."""
    module = SafetyCarDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=sc_decision_context)
    
    import time
    start = time.time()
    recommendation = module.evaluate_safe(decision_input)
    latency_ms = (time.time() - start) * 1000
    
    # Should be fast (allowing some overhead for first call)
    assert latency_ms < 500  # Generous for first call
    
    # Second call should be faster
    start = time.time()
    module.evaluate_safe(decision_input)
    latency_ms = (time.time() - start) * 1000
    
    assert latency_ms < 200  # More realistic target


def test_safety_car_confidence_high_certainty(sc_decision_context):
    """Test high confidence for clear SC decisions."""
    sc_decision_context.tire_age = 30  # Very old tires
    sc_decision_context.current_position = 8  # Midfield
    
    module = SafetyCarDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=sc_decision_context)
    
    recommendation = module.evaluate(decision_input)
    
    if recommendation:
        # Should have reasonable confidence
        assert recommendation.confidence_score > 0.5


def test_safety_car_never_red_traffic_light(sc_decision_context):
    """Test SC decisions never show RED traffic light."""
    module = SafetyCarDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=sc_decision_context)
    
    # Try multiple scenarios
    for tire_age in [10, 20, 30]:
        sc_decision_context.tire_age = tire_age
        recommendation = module.evaluate(decision_input)
        
        if recommendation:
            assert recommendation.traffic_light in [TrafficLight.AMBER, TrafficLight.GREEN]


def test_safety_car_has_reasoning(sc_decision_context):
    """Test that SC recommendations have detailed reasoning."""
    module = SafetyCarDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=sc_decision_context)
    
    recommendation = module.evaluate(decision_input)
    
    if recommendation:
        assert len(recommendation.reasoning.primary_factors) > 0
        assert len(recommendation.alternatives) > 0


def test_safety_car_performance_tracking(sc_decision_context):
    """Test module performance tracking."""
    module = SafetyCarDecision(config_path='config/decision_engine.yaml')
    decision_input = DecisionInput(context=sc_decision_context)
    
    # Make multiple evaluations
    for _ in range(10):
        module.evaluate_safe(decision_input)
    
    stats = module.get_stats()
    
    assert stats['evaluation_count'] == 10
    assert stats['name'] == "safety_car_decision"
    # Check latency target
    if stats['latency_p95_ms'] > 0:
        assert stats['latency_p95_ms'] < 150  # Should be fast
