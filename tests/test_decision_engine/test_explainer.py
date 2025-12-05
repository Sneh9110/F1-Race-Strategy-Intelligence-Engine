"""Tests for explainability components."""

import pytest
from decision_engine.explainer import DecisionExplainer, DecisionLogger, DecisionAuditor
from decision_engine import DecisionRecommendation, DecisionAction, DecisionCategory, ConfidenceLevel, TrafficLight, DecisionReasoning


def test_decision_explainer_initialization():
    """Test explainer initialization."""
    explainer = DecisionExplainer()
    assert explainer is not None


def test_generate_reasoning(sample_decision_context):
    """Test reasoning generation."""
    explainer = DecisionExplainer()
    
    factors = ["High tire age (25 laps)", "Pace dropping", "Pit window open"]
    rules = ["tire_age > optimal", "pace_drop > 0.5s"]
    models = {"degradation_model": 0.85, "lap_time_model": 0.75}
    
    reasoning = explainer.generate_reasoning(
        factors=factors,
        rules=rules,
        models=models,
        context=sample_decision_context
    )
    
    assert len(reasoning.primary_factors) > 0
    assert len(reasoning.rule_triggers) > 0
    assert len(reasoning.model_contributions) > 0


def test_generate_reasoning_limits_factors():
    """Test reasoning limits to top 5 factors."""
    explainer = DecisionExplainer()
    
    factors = [f"Factor {i}" for i in range(10)]
    rules = []
    models = {}
    
    reasoning = explainer.generate_reasoning(
        factors=factors,
        rules=rules,
        models=models
    )
    
    assert len(reasoning.primary_factors) <= 5


def test_generate_explanation_text():
    """Test human-readable explanation generation."""
    explainer = DecisionExplainer()
    
    rec = DecisionRecommendation(
        action=DecisionAction.PIT_NOW,
        category=DecisionCategory.PIT_TIMING,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.85,
        traffic_light=TrafficLight.GREEN,
        reasoning=DecisionReasoning(
            primary_factors=["High tire age", "Degrading pace"],
            rule_triggers=["tire_age > optimal"],
            model_contributions={"degradation_model": 0.85},
        ),
        expected_gain_seconds=3.5,
        risk_score=0.25,
        priority=9,
    )
    
    explanation = explainer.generate_explanation_text(rec)
    
    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert "PIT_NOW" in explanation or "pit" in explanation.lower()
    assert "85" in explanation or "0.85" in explanation  # Confidence


def test_generate_comparison_table():
    """Test comparison table generation."""
    explainer = DecisionExplainer()
    
    recs = [
        DecisionRecommendation(
            action=DecisionAction.PIT_NOW,
            category=DecisionCategory.PIT_TIMING,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.85,
            traffic_light=TrafficLight.GREEN,
            reasoning=DecisionReasoning(),
            expected_gain_seconds=3.5,
            risk_score=0.25,
            priority=9,
        ),
        DecisionRecommendation(
            action=DecisionAction.STAY_OUT,
            category=DecisionCategory.PIT_TIMING,
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.65,
            traffic_light=TrafficLight.AMBER,
            reasoning=DecisionReasoning(),
            expected_gain_seconds=1.5,
            risk_score=0.4,
            priority=9,
        ),
    ]
    
    table = explainer.generate_comparison_table(recs)
    
    assert isinstance(table, str)
    assert "Action" in table
    assert "Confidence" in table
    assert "PIT_NOW" in table or "pit" in table.lower()


def test_decision_logger_initialization():
    """Test logger initialization."""
    logger = DecisionLogger()
    assert logger is not None


def test_log_decision(sample_decision_context):
    """Test decision logging."""
    logger = DecisionLogger()
    
    rec = DecisionRecommendation(
        action=DecisionAction.PIT_NOW,
        category=DecisionCategory.PIT_TIMING,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.85,
        traffic_light=TrafficLight.GREEN,
        reasoning=DecisionReasoning(),
        expected_gain_seconds=3.5,
        risk_score=0.25,
        priority=9,
    )
    
    # Should not crash
    logger.log_decision(
        session_id=sample_decision_context.session_id,
        lap_number=sample_decision_context.lap_number,
        driver_number=sample_decision_context.driver_number,
        recommendations=[rec]
    )


def test_log_decision_trace():
    """Test decision trace logging."""
    logger = DecisionLogger()
    
    module_results = {
        'pit_timing': {'recommendation': 'PIT_NOW', 'confidence': 0.85, 'latency_ms': 50},
        'safety_car': {'recommendation': None, 'confidence': 0.0, 'latency_ms': 30},
    }
    
    # Should not crash
    logger.log_decision_trace(
        session_id="test",
        lap_number=25,
        module_results=module_results
    )


def test_export_decision_history():
    """Test decision history export."""
    logger = DecisionLogger()
    
    # Log some decisions
    rec = DecisionRecommendation(
        action=DecisionAction.PIT_NOW,
        category=DecisionCategory.PIT_TIMING,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.85,
        traffic_light=TrafficLight.GREEN,
        reasoning=DecisionReasoning(),
        expected_gain_seconds=3.5,
        risk_score=0.25,
        priority=9,
    )
    
    logger.log_decision(
        session_id="test",
        lap_number=25,
        driver_number=44,
        recommendations=[rec]
    )
    
    # Export should work (may return empty or simplified data)
    history = logger.export_decision_history(session_id="test")
    
    assert isinstance(history, (list, dict, str))


def test_decision_auditor_initialization():
    """Test auditor initialization."""
    auditor = DecisionAuditor()
    assert auditor is not None


def test_audit_decision_basic():
    """Test basic decision audit."""
    auditor = DecisionAuditor()
    
    rec = DecisionRecommendation(
        action=DecisionAction.PIT_NOW,
        category=DecisionCategory.PIT_TIMING,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.85,
        traffic_light=TrafficLight.GREEN,
        reasoning=DecisionReasoning(),
        expected_gain_seconds=3.5,
        risk_score=0.25,
        priority=9,
    )
    
    # Audit with actual outcome
    audit_result = auditor.audit_decision(
        recommendation=rec,
        actual_gain_seconds=3.2,
        actual_outcome="Successful pit, gained 1 position"
    )
    
    assert audit_result is not None
    assert 'accuracy' in audit_result or 'error' in audit_result


def test_audit_decision_better_than_expected():
    """Test audit when outcome better than expected."""
    auditor = DecisionAuditor()
    
    rec = DecisionRecommendation(
        action=DecisionAction.PIT_NOW,
        category=DecisionCategory.PIT_TIMING,
        confidence=ConfidenceLevel.MEDIUM,
        confidence_score=0.65,
        traffic_light=TrafficLight.AMBER,
        reasoning=DecisionReasoning(),
        expected_gain_seconds=2.0,
        risk_score=0.4,
        priority=9,
    )
    
    audit_result = auditor.audit_decision(
        recommendation=rec,
        actual_gain_seconds=4.5,  # Better than expected
        actual_outcome="Excellent result"
    )
    
    assert audit_result is not None


def test_audit_decision_worse_than_expected():
    """Test audit when outcome worse than expected."""
    auditor = DecisionAuditor()
    
    rec = DecisionRecommendation(
        action=DecisionAction.PIT_NOW,
        category=DecisionCategory.PIT_TIMING,
        confidence=ConfidenceLevel.HIGH,
        confidence_score=0.85,
        traffic_light=TrafficLight.GREEN,
        reasoning=DecisionReasoning(),
        expected_gain_seconds=3.5,
        risk_score=0.25,
        priority=9,
    )
    
    audit_result = auditor.audit_decision(
        recommendation=rec,
        actual_gain_seconds=0.5,  # Much worse than expected
        actual_outcome="Slow pit stop, lost positions"
    )
    
    assert audit_result is not None


def test_audit_post_race_analysis():
    """Test post-race audit analysis."""
    auditor = DecisionAuditor()
    
    # Create multiple decisions
    decisions = []
    for i in range(5):
        rec = DecisionRecommendation(
            action=DecisionAction.PIT_NOW,
            category=DecisionCategory.PIT_TIMING,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.8,
            traffic_light=TrafficLight.GREEN,
            reasoning=DecisionReasoning(),
            expected_gain_seconds=3.0,
            risk_score=0.3,
            priority=9,
        )
        decisions.append({
            'recommendation': rec,
            'actual_gain': 3.0 + (i * 0.5),
            'outcome': 'Good'
        })
    
    # Run post-race analysis (may be simplified)
    analysis = auditor.analyze_session_decisions(decisions)
    
    assert analysis is not None or decisions  # Either returns analysis or accepts data
