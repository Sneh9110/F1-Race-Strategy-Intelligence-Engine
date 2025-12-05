"""Pace monitoring decision module."""

from typing import Optional, Dict, Any, List
from decision_engine.base import BaseDecisionModule
from decision_engine.schemas import (
    DecisionInput,
    DecisionRecommendation,
    DecisionContext,
    DecisionAction,
    DecisionCategory,
    ConfidenceLevel,
    AlternativeOption,
    DecisionReasoning,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PaceMonitoringDecision(BaseDecisionModule):
    """Decision module for pace monitoring (detect pace drops, recommend adjustments)."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize pace monitoring module."""
        super().__init__(config_path)
        
        module_config = self.config.get('modules', {}).get('pace_monitoring', {}).get('config', {})
        self.pace_drop_threshold = module_config.get('pace_drop_threshold', 0.5)
        self.degradation_excess_threshold = module_config.get('degradation_excess_threshold', 0.2)
        self.position_loss_threshold = module_config.get('position_loss_threshold', 2)
        self.gap_increase_threshold = module_config.get('gap_increase_threshold', 0.3)
        self.sample_size_min = module_config.get('sample_size_min', 5)
    
    @property
    def name(self) -> str:
        return "pace_monitoring"
    
    @property
    def version(self) -> str:
        return "v1.0.0"
    
    @property
    def category(self) -> str:
        return DecisionCategory.PACE_ADJUSTMENT.value
    
    @property
    def priority(self) -> int:
        return 5  # Medium-low priority (advisory)
    
    def is_applicable(self, context: DecisionContext) -> bool:
        """Check if pace monitoring is applicable."""
        # Need sufficient lap time data
        return len(context.recent_lap_times) >= self.sample_size_min
    
    def get_confidence(self, context: DecisionContext) -> float:
        """Calculate confidence for pace monitoring."""
        factors = {}
        
        # More data = higher confidence
        sample_size = len(context.recent_lap_times)
        factors['sample_size'] = min(1.0, sample_size / 10.0)
        
        # Lap time consistency
        if sample_size >= 5:
            avg = sum(context.recent_lap_times) / sample_size
            variance = sum((lt - avg) ** 2 for lt in context.recent_lap_times) / sample_size
            factors['consistency'] = max(0.0, 1.0 - variance / 5.0)
        else:
            factors['consistency'] = 0.5
        
        factors['default'] = 0.6
        
        weights = {
            'sample_size': 0.3,
            'consistency': 0.3,
            'default': 0.4,
        }
        
        return self._calculate_confidence(factors, weights)
    
    def evaluate(self, decision_input: DecisionInput) -> Optional[DecisionRecommendation]:
        """Evaluate pace monitoring decision."""
        context = decision_input.context
        
        if len(context.recent_lap_times) < self.sample_size_min:
            return None
        
        factors = []
        rules = []
        model_contributions = {}
        
        # Calculate pace drop
        mid_point = len(context.recent_lap_times) // 2
        early_avg = sum(context.recent_lap_times[:mid_point]) / mid_point
        recent_avg = sum(context.recent_lap_times[mid_point:]) / (len(context.recent_lap_times) - mid_point)
        pace_drop = recent_avg - early_avg
        
        # Rule 1: Pace drop detected
        if pace_drop > self.pace_drop_threshold:
            factors.append(f"Pace drop: {pace_drop:.2f}s (recent avg vs earlier)")
            rules.append("pace_drop_detected")
            
            # Check if degradation-related
            expected_deg = decision_input.feature_data.get('expected_degradation', 0.05)
            actual_deg = decision_input.feature_data.get('actual_degradation', 0.05)
            
            if actual_deg > expected_deg * (1 + self.degradation_excess_threshold):
                factors.append(f"Degradation {(actual_deg/expected_deg - 1)*100:.0f}% above expected")
                rules.append("degradation_exceeds_expected")
                action = DecisionAction.CONSERVATIVE_PACE
            else:
                # Not degradation - maybe traffic or driver error
                traffic_impact = decision_input.feature_data.get('traffic_penalty', 0.0)
                if traffic_impact > 0.3:
                    factors.append(f"Traffic penalty: {traffic_impact:.2f}s/lap")
                    rules.append("traffic_impact")
                    action = DecisionAction.NO_ACTION  # Wait for clear air
                else:
                    factors.append("Unexplained pace drop - investigate")
                    rules.append("unexplained_pace_drop")
                    action = DecisionAction.CONSERVATIVE_PACE
            
            return self._build_pace_recommendation(
                context, action, pace_drop, factors, rules, model_contributions
            )
        
        # Rule 2: Pace improvement detected
        elif pace_drop < -0.3:  # Faster than before
            factors.append(f"Pace improvement: {abs(pace_drop):.2f}s faster")
            rules.append("pace_improvement_detected")
            
            # Can push harder
            action = DecisionAction.AGGRESSIVE_PACE
            
            return self._build_pace_recommendation(
                context, action, pace_drop, factors, rules, model_contributions
            )
        
        # Rule 3: Losing positions
        positions_lost = decision_input.feature_data.get('positions_lost_last_5_laps', 0)
        if positions_lost >= self.position_loss_threshold:
            factors.append(f"Lost {positions_lost} positions in last 5 laps")
            rules.append("losing_positions")
            action = DecisionAction.AGGRESSIVE_PACE
            
            return self._build_pace_recommendation(
                context, action, pace_drop, factors, rules, model_contributions
            )
        
        return None
    
    def _build_pace_recommendation(
        self,
        context: DecisionContext,
        action: DecisionAction,
        pace_delta: float,
        factors: List[str],
        rules: List[str],
        model_contributions: Dict[str, float]
    ) -> DecisionRecommendation:
        """Build pace adjustment recommendation."""
        confidence_score = self.get_confidence(context)
        risk_score = 0.3 if action == DecisionAction.CONSERVATIVE_PACE else 0.5
        
        if action == DecisionAction.CONSERVATIVE_PACE:
            reasoning = DecisionReasoning(
                primary_factors=factors[:5],
                rule_triggers=rules,
                model_contributions=model_contributions,
                risk_assessment="Low risk: Save tires, avoid cliff",
                opportunity_assessment="Opportunity: Extend stint, manage degradation",
            )
            alternatives = [
                AlternativeOption(
                    action=DecisionAction.NO_ACTION,
                    expected_outcome="Maintain current pace, risk tire cliff",
                    confidence=0.5,
                    pros=["Keep position", "No change needed"],
                    cons=["Risk tire failure", "Pace drop continues"],
                ),
                AlternativeOption(
                    action=DecisionAction.PIT_NOW,
                    expected_outcome="Pit for fresh tires",
                    confidence=0.7,
                    pros=["Fresh tires", "Restore pace"],
                    cons=["Pit time loss", "Track position loss"],
                ),
            ]
        else:  # AGGRESSIVE_PACE
            reasoning = DecisionReasoning(
                primary_factors=factors[:5],
                rule_triggers=rules,
                model_contributions=model_contributions,
                risk_assessment="Medium risk: Higher tire wear, potential errors",
                opportunity_assessment="Opportunity: Gain positions, close gaps",
            )
            alternatives = [
                AlternativeOption(
                    action=DecisionAction.NO_ACTION,
                    expected_outcome="Maintain current pace",
                    confidence=0.6,
                    pros=["Manage tires", "Consistent pace"],
                    cons=["Miss overtake opportunities", "Lose positions"],
                ),
            ]
        
        return DecisionRecommendation(
            action=action,
            category=DecisionCategory.PACE_ADJUSTMENT,
            confidence=self._score_to_level(confidence_score),
            confidence_score=confidence_score,
            traffic_light=self._determine_traffic_light(confidence_score, risk_score),
            reasoning=reasoning,
            alternatives=alternatives,
            expected_gain_seconds=abs(pace_delta) * 5,  # Rough estimate
            risk_score=risk_score,
            priority=self.priority,
            metadata={'module': self.name, 'version': self.version, 'pace_delta': pace_delta},
        )
    
    @staticmethod
    def _score_to_level(score: float) -> ConfidenceLevel:
        """Convert score to confidence level."""
        if score >= 0.85:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.7:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
