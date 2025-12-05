"""Scoring system for decision confidence and risk."""

from typing import Dict, List, Any, Optional
from decision_engine.schemas import (
    DecisionContext,
    DecisionAction,
    DecisionRecommendation,
    DecisionCategory,
    TrafficLight,
    ConfidenceLevel,
    SimulationContext,
)


class ConfidenceScorer:
    """Calculate confidence scores for decisions."""
    
    @staticmethod
    def calculate_confidence(
        factors: Dict[str, float], 
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate weighted confidence score.
        
        Args:
            factors: Factor name -> value (0-1)
            weights: Factor name -> weight
            
        Returns:
            Confidence score (0-1)
        """
        if not factors or not weights:
            return 0.5
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for factor_name, factor_value in factors.items():
            weight = weights.get(factor_name, 0.5)
            weighted_sum += factor_value * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        confidence = weighted_sum / total_weight
        
        # Apply floor and ceiling
        confidence = max(0.1, min(1.0, confidence))
        
        return confidence
    
    @staticmethod
    def calculate_risk_score(
        context: DecisionContext, 
        action: DecisionAction
    ) -> float:
        """
        Calculate risk score for decision.
        
        Args:
            context: Decision context
            action: Proposed action
            
        Returns:
            Risk score (0-1, higher = more risk)
        """
        risk_factors = {}
        
        # Tire age risk
        if context.tire_compound == 'SOFT':
            optimal_stint = 20
        elif context.tire_compound == 'MEDIUM':
            optimal_stint = 25
        else:
            optimal_stint = 30
        
        risk_factors['tire_age'] = min(1.0, context.tire_age / optimal_stint)
        
        # Position risk (podium = higher risk to change)
        risk_factors['position'] = 1.0 / context.current_position if context.current_position <= 10 else 0.1
        
        # Gap risk
        if context.gap_to_ahead is not None and context.gap_to_ahead > 0:
            risk_factors['gap'] = 1.0 / max(0.5, context.gap_to_ahead)
        else:
            risk_factors['gap'] = 0.3
        
        # SC risk (SC active = unpredictable)
        risk_factors['safety_car'] = 0.7 if context.safety_car_active else 0.2
        
        # Weather risk
        if context.weather_temp and context.track_temp:
            temp_delta = abs(context.track_temp - context.weather_temp)
            risk_factors['weather'] = min(1.0, temp_delta / 20.0)
        else:
            risk_factors['weather'] = 0.3
        
        # Calculate weighted risk
        weights = {
            'tire_age': 0.25,
            'position': 0.25,
            'gap': 0.20,
            'safety_car': 0.15,
            'weather': 0.15,
        }
        
        total_weight = sum(weights.values())
        risk_score = sum(
            risk_factors.get(k, 0.5) * v 
            for k, v in weights.items()
        ) / total_weight
        
        # Action-specific adjustments
        if action == DecisionAction.PIT_NOW:
            risk_score += 0.1
        elif action == DecisionAction.AGGRESSIVE_PACE:
            risk_score += 0.15
        elif action in [DecisionAction.SWITCH_TO_INTERS, DecisionAction.SWITCH_TO_WETS]:
            risk_score += 0.2  # Weather decisions risky
        
        return min(1.0, max(0.0, risk_score))
    
    @staticmethod
    def determine_traffic_light(
        confidence: float, 
        risk_score: float,
        category: Optional[DecisionCategory] = None
    ) -> TrafficLight:
        """
        Determine traffic light color.
        
        Args:
            confidence: Confidence score (0-1)
            risk_score: Risk score (0-1)
            category: Decision category
            
        Returns:
            TrafficLight enum
        """
        # GREEN: High confidence, low risk
        if confidence >= 0.8 and risk_score <= 0.3:
            return TrafficLight.GREEN
        
        # RED: Low confidence or high risk
        if confidence < 0.5 or risk_score > 0.7:
            # Safety car decisions never RED (too urgent)
            if category and category == DecisionCategory.SAFETY_CAR:
                return TrafficLight.AMBER
            return TrafficLight.RED
        
        # AMBER: Medium confidence/risk
        return TrafficLight.AMBER
    
    @staticmethod
    def calculate_expected_gain(
        context: DecisionContext,
        action: DecisionAction,
        simulation_context: Optional[SimulationContext] = None
    ) -> float:
        """
        Calculate expected time gain for action.
        
        Args:
            context: Decision context
            action: Proposed action
            simulation_context: Simulation results
            
        Returns:
            Expected gain in seconds (-30 to +30)
        """
        # Position value mapping (F1 points roughly)
        position_values = {
            1: 25.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0,
            6: 8.0, 7: 6.0, 8: 4.0, 9: 2.0, 10: 1.0,
        }
        
        # Default gains by action type
        action_gains = {
            DecisionAction.PIT_NOW: 3.5,
            DecisionAction.STAY_OUT: 1.5,
            DecisionAction.UNDERCUT_NOW: 2.5,
            DecisionAction.OVERCUT_NOW: 3.0,
            DecisionAction.PIT_UNDER_SC: 8.0,
            DecisionAction.SWITCH_TO_INTERS: 10.0,
            DecisionAction.SWITCH_TO_WETS: 10.0,
            DecisionAction.AGGRESSIVE_PACE: 2.0,
            DecisionAction.CONSERVATIVE_PACE: 1.0,
        }
        
        expected_gain = action_gains.get(action, 0.0)
        
        # Adjust based on simulation context
        if simulation_context:
            # Use position delta from simulation
            current_pos = context.current_position
            expected_pos = simulation_context.expected_position
            position_delta = current_pos - expected_pos
            
            if position_delta > 0:  # Gaining positions
                gain_value = position_values.get(int(expected_pos), 0.0)
                expected_gain += gain_value / 3.0
        
        # Clamp to realistic range
        return max(-30.0, min(30.0, expected_gain))


class PriorityCalculator:
    """Calculate priority for decisions."""
    
    @staticmethod
    def calculate_priority(
        category: DecisionCategory, 
        context: DecisionContext
    ) -> int:
        """
        Calculate decision priority.
        
        Args:
            category: Decision category
            context: Decision context
            
        Returns:
            Priority (1-10, higher = more important)
        """
        # Base priorities
        base_priorities = {
            DecisionCategory.SAFETY_CAR: 10,
            DecisionCategory.RAIN: 10,
            DecisionCategory.PIT_TIMING: 9,
            DecisionCategory.UNDERCUT_OVERCUT: 8,
            DecisionCategory.STRATEGY_CONVERSION: 7,
            DecisionCategory.OFFSET_STRATEGY: 6,
            DecisionCategory.PACE_ADJUSTMENT: 5,
        }
        
        priority = base_priorities.get(category, 5)
        
        # Adjustments
        if context.safety_car_active:
            priority = min(10, priority + 1)  # Boost during SC
        
        # Tire cliff = critical
        if len(context.recent_lap_times) >= 2:
            last_delta = context.recent_lap_times[-1] - context.recent_lap_times[-2]
            if last_delta > 1.0:  # Sudden pace drop
                priority = min(10, priority + 1)
        
        # Clamp to 1-10
        return max(1, min(10, priority))


class DecisionRanker:
    """Rank and filter recommendations."""
    
    @staticmethod
    def rank_recommendations(
        recommendations: List[DecisionRecommendation]
    ) -> List[DecisionRecommendation]:
        """
        Rank recommendations by priority, confidence, and expected gain.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Sorted and filtered recommendations
        """
        if not recommendations:
            return []
        
        # Filter out very low confidence (unless critical)
        filtered = [
            rec for rec in recommendations
            if rec.confidence_score >= 0.3 or rec.priority >= 9
        ]
        
        # Deduplicate similar actions
        seen_actions = set()
        deduped = []
        
        for rec in filtered:
            if rec.action not in seen_actions:
                deduped.append(rec)
                seen_actions.add(rec.action)
            else:
                # Keep higher confidence version
                existing = next(r for r in deduped if r.action == rec.action)
                if rec.confidence_score > existing.confidence_score:
                    deduped.remove(existing)
                    deduped.append(rec)
        
        # Sort by: priority (desc), confidence (desc), expected gain (desc)
        sorted_recs = sorted(
            deduped,
            key=lambda r: (r.priority, r.confidence_score, r.expected_gain_seconds),
            reverse=True
        )
        
        return sorted_recs


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate weighted average."""
    if not values or not weights or len(values) != len(weights):
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return sum(values) / len(values)
    
    return sum(v * w for v, w in zip(values, weights)) / total_weight


def confidence_to_level(confidence: float) -> ConfidenceLevel:
    """Map confidence score to level."""
    if confidence >= 0.85:
        return ConfidenceLevel.VERY_HIGH
    elif confidence >= 0.7:
        return ConfidenceLevel.HIGH
    elif confidence >= 0.5:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW
