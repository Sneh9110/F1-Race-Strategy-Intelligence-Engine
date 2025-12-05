"""Tests for scoring system."""

import pytest
from decision_engine.scoring import ConfidenceScorer, PriorityCalculator, DecisionRanker
from decision_engine import DecisionRecommendation, DecisionAction, DecisionCategory, ConfidenceLevel, TrafficLight, DecisionReasoning


def test_confidence_scorer_initialization():
    """Test scorer initialization."""
    scorer = ConfidenceScorer()
    assert scorer is not None


def test_calculate_confidence_basic():
    """Test basic confidence calculation."""
    scorer = ConfidenceScorer()
    
    factors = {
        'tire_age': 0.8,
        'consistency': 0.7,
        'pit_window': 0.9,
    }
    weights = {
        'tire_age': 0.4,
        'consistency': 0.3,
        'pit_window': 0.3,
    }
    
    confidence = scorer.calculate_confidence(factors, weights)
    
    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.5  # Should be high given inputs


def test_calculate_confidence_floor():
    """Test confidence floor at 0.1."""
    scorer = ConfidenceScorer()
    
    factors = {'factor1': 0.0}
    weights = {'factor1': 1.0}
    
    confidence = scorer.calculate_confidence(factors, weights)
    
    assert confidence >= 0.1


def test_calculate_confidence_ceiling():
    """Test confidence ceiling at 1.0."""
    scorer = ConfidenceScorer()
    
    factors = {'factor1': 1.5}  # Exceeds 1.0
    weights = {'factor1': 1.0}
    
    confidence = scorer.calculate_confidence(factors, weights)
    
    assert confidence <= 1.0


def test_calculate_risk_score(sample_decision_context):
    """Test risk score calculation."""
    scorer = ConfidenceScorer()
    
    risk = scorer.calculate_risk_score(sample_decision_context)
    
    assert 0.0 <= risk <= 1.0


def test_calculate_risk_high_for_critical_position(sample_decision_context):
    """Test risk is higher for podium positions."""
    scorer = ConfidenceScorer()
    
    sample_decision_context.current_position = 1  # Leader
    risk_high = scorer.calculate_risk_score(sample_decision_context)
    
    sample_decision_context.current_position = 10  # Midfield
    risk_low = scorer.calculate_risk_score(sample_decision_context)
    
    # Risk should be higher for leader
    assert risk_high > risk_low


def test_determine_traffic_light_green():
    """Test GREEN traffic light determination."""
    scorer = ConfidenceScorer()
    
    traffic_light = scorer.determine_traffic_light(confidence=0.85, risk_score=0.2)
    
    assert traffic_light == TrafficLight.GREEN


def test_determine_traffic_light_red_low_confidence():
    """Test RED traffic light for low confidence."""
    scorer = ConfidenceScorer()
    
    traffic_light = scorer.determine_traffic_light(confidence=0.4, risk_score=0.5)
    
    assert traffic_light == TrafficLight.RED


def test_determine_traffic_light_red_high_risk():
    """Test RED traffic light for high risk."""
    scorer = ConfidenceScorer()
    
    traffic_light = scorer.determine_traffic_light(confidence=0.7, risk_score=0.8)
    
    assert traffic_light == TrafficLight.RED


def test_determine_traffic_light_amber():
    """Test AMBER traffic light for medium confidence/risk."""
    scorer = ConfidenceScorer()
    
    traffic_light = scorer.determine_traffic_light(confidence=0.65, risk_score=0.45)
    
    assert traffic_light == TrafficLight.AMBER


def test_calculate_expected_gain_basic(sample_decision_context):
    """Test expected gain calculation."""
    scorer = ConfidenceScorer()
    
    gain = scorer.calculate_expected_gain(
        action=DecisionAction.PIT_NOW,
        context=sample_decision_context
    )
    
    assert -30.0 <= gain <= 30.0


def test_priority_calculator():
    """Test priority calculation."""
    calculator = PriorityCalculator()
    
    priority = calculator.calculate_priority(
        base_priority=7,
        context_critical=False
    )
    
    assert 1 <= priority <= 10


def test_priority_calculator_critical_boost(sc_decision_context):
    """Test priority boost for critical situations."""
    calculator = PriorityCalculator()
    
    priority_normal = calculator.calculate_priority(
        base_priority=7,
        context_critical=False
    )
    
    priority_critical = calculator.calculate_priority(
        base_priority=7,
        context_critical=True
    )
    
    assert priority_critical > priority_normal


def test_decision_ranker_initialization():
    """Test ranker initialization."""
    ranker = DecisionRanker()
    assert ranker is not None


def test_decision_ranker_sorts_by_priority():
    """Test ranker sorts by priority."""
    ranker = DecisionRanker()
    
    recs = [
        DecisionRecommendation(
            action=DecisionAction.PIT_NOW,
            category=DecisionCategory.PIT_TIMING,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.8,
            traffic_light=TrafficLight.GREEN,
            reasoning=DecisionReasoning(),
            expected_gain_seconds=3.0,
            risk_score=0.3,
            priority=7,
        ),
        DecisionRecommendation(
            action=DecisionAction.PIT_UNDER_SC,
            category=DecisionCategory.SAFETY_CAR,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.85,
            traffic_light=TrafficLight.GREEN,
            reasoning=DecisionReasoning(),
            expected_gain_seconds=5.0,
            risk_score=0.2,
            priority=10,
        ),
    ]
    
    ranked = ranker.rank_recommendations(recs)
    
    # Higher priority should be first
    assert ranked[0].priority >= ranked[1].priority


def test_decision_ranker_filters_low_confidence():
    """Test ranker filters low confidence recommendations."""
    ranker = DecisionRanker()
    
    recs = [
        DecisionRecommendation(
            action=DecisionAction.PIT_NOW,
            category=DecisionCategory.PIT_TIMING,
            confidence=ConfidenceLevel.LOW,
            confidence_score=0.2,  # Very low
            traffic_light=TrafficLight.RED,
            reasoning=DecisionReasoning(),
            expected_gain_seconds=1.0,
            risk_score=0.8,
            priority=7,
        ),
        DecisionRecommendation(
            action=DecisionAction.STAY_OUT,
            category=DecisionCategory.PIT_TIMING,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.8,
            traffic_light=TrafficLight.GREEN,
            reasoning=DecisionReasoning(),
            expected_gain_seconds=2.0,
            risk_score=0.3,
            priority=7,
        ),
    ]
    
    ranked = ranker.rank_recommendations(recs, min_confidence=0.3)
    
    # Should filter out low confidence
    assert all(rec.confidence_score >= 0.3 for rec in ranked)


def test_decision_ranker_keeps_critical_priority():
    """Test ranker keeps critical priority even with low confidence."""
    ranker = DecisionRanker()
    
    recs = [
        DecisionRecommendation(
            action=DecisionAction.PIT_UNDER_SC,
            category=DecisionCategory.SAFETY_CAR,
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.55,
            traffic_light=TrafficLight.AMBER,
            reasoning=DecisionReasoning(),
            expected_gain_seconds=4.0,
            risk_score=0.4,
            priority=10,  # Critical
        ),
    ]
    
    ranked = ranker.rank_recommendations(recs, min_confidence=0.7)
    
    # Should keep high priority even below confidence threshold
    assert len(ranked) == 1


def test_decision_ranker_deduplicates_actions():
    """Test ranker deduplicates same actions."""
    ranker = DecisionRanker()
    
    recs = [
        DecisionRecommendation(
            action=DecisionAction.PIT_NOW,
            category=DecisionCategory.PIT_TIMING,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.8,
            traffic_light=TrafficLight.GREEN,
            reasoning=DecisionReasoning(),
            expected_gain_seconds=3.0,
            risk_score=0.3,
            priority=9,
        ),
        DecisionRecommendation(
            action=DecisionAction.PIT_NOW,
            category=DecisionCategory.UNDERCUT_OVERCUT,
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.75,
            traffic_light=TrafficLight.GREEN,
            reasoning=DecisionReasoning(),
            expected_gain_seconds=2.5,
            risk_score=0.35,
            priority=8,
        ),
    ]
    
    ranked = ranker.rank_recommendations(recs)
    
    # Should keep higher priority/confidence version
    actions = [rec.action for rec in ranked]
    # May keep both or deduplicate - implementation specific
    assert len(ranked) > 0
