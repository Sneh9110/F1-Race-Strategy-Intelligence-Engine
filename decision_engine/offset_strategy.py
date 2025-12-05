"""Offset strategy decision module."""

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


class OffsetStrategyDecision(BaseDecisionModule):
    """Decision module for offset strategy (different tire compounds vs rivals)."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize offset strategy module."""
        super().__init__(config_path)
        
        module_config = self.config.get('modules', {}).get('offset_strategy', {}).get('config', {})
        self.tire_age_delta_threshold = module_config.get('tire_age_delta_threshold', 5)
        self.pace_delta_threshold = module_config.get('pace_delta_threshold', 0.3)
        self.undercut_window_threshold = module_config.get('undercut_window_threshold', 3.0)
        self.sustained_advantage_laps = module_config.get('sustained_advantage_laps', 5)
    
    @property
    def name(self) -> str:
        return "offset_strategy"
    
    @property
    def version(self) -> str:
        return "v1.0.0"
    
    @property
    def category(self) -> str:
        return DecisionCategory.OFFSET_STRATEGY.value
    
    @property
    def priority(self) -> int:
        return 6  # Medium priority
    
    def is_applicable(self, context: DecisionContext) -> bool:
        """Check if offset strategy is applicable."""
        # Need rival data
        return True
    
    def get_confidence(self, context: DecisionContext) -> float:
        """Calculate confidence for offset strategy."""
        factors = {}
        
        # Tire age = more data = higher confidence
        factors['tire_age'] = min(1.0, context.tire_age / 20.0)
        
        # Gap stability (recent lap times consistent)
        if len(context.recent_lap_times) >= 5:
            variance = max(0.1, sum((lt - sum(context.recent_lap_times)/len(context.recent_lap_times))**2 
                                    for lt in context.recent_lap_times) / len(context.recent_lap_times))
            factors['gap_stability'] = max(0.0, 1.0 - variance / 5.0)
        else:
            factors['gap_stability'] = 0.5
        
        factors['default'] = 0.6
        
        weights = {
            'tire_age': 0.3,
            'gap_stability': 0.3,
            'default': 0.4,
        }
        
        return self._calculate_confidence(factors, weights)
    
    def evaluate(self, decision_input: DecisionInput) -> Optional[DecisionRecommendation]:
        """Evaluate offset strategy decision."""
        context = decision_input.context
        
        if not decision_input.rival_contexts:
            return None
        
        factors = []
        rules = []
        model_contributions = {}
        
        # Find rivals on different compounds
        offset_opportunities = []
        
        for rival in decision_input.rival_contexts:
            if rival.rival_tire_compound != context.tire_compound:
                tire_age_delta = context.tire_age - rival.rival_tire_age
                
                # Rule 1: We have fresher tires (age delta > threshold)
                if abs(tire_age_delta) > self.tire_age_delta_threshold:
                    factors.append(
                        f"Tire age delta vs P{rival.rival_position}: {tire_age_delta} laps "
                        f"(us: {context.tire_compound}, rival: {rival.rival_tire_compound})"
                    )
                    rules.append("tire_age_delta_exceeds_threshold")
                    
                    # Rule 2: Within undercut/overtake range
                    if rival.gap_to_rival < self.undercut_window_threshold:
                        factors.append(f"Gap to P{rival.rival_position}: {rival.gap_to_rival:.1f}s (undercut range)")
                        rules.append("undercut_window_open")
                        
                        offset_opportunities.append({
                            'rival': rival,
                            'tire_age_delta': tire_age_delta,
                            'gap': rival.gap_to_rival,
                        })
        
        if not offset_opportunities:
            return None
        
        # Rule 3: Pace advantage with offset
        pace_delta = decision_input.feature_data.get('pace_delta', 0.0)
        if pace_delta > self.pace_delta_threshold:
            factors.append(f"Pace advantage: {pace_delta:.2f}s/lap faster")
            rules.append("pace_advantage_detected")
        
        # Build recommendation
        confidence_score = self.get_confidence(context)
        risk_score = 0.4  # Medium risk
        
        reasoning = DecisionReasoning(
            primary_factors=factors[:5],
            rule_triggers=rules,
            model_contributions=model_contributions,
            risk_assessment="Medium risk: Offset requires pace advantage to work",
            opportunity_assessment=f"Opportunity: Exploit tire offset against {len(offset_opportunities)} rival(s)",
        )
        
        alternatives = [
            AlternativeOption(
                action=DecisionAction.NO_ACTION,
                expected_outcome="Match rival tire strategy",
                confidence=0.6,
                pros=["Lower risk", "Easier race management"],
                cons=["No strategic advantage", "Follow rival pace"],
            ),
        ]
        
        return DecisionRecommendation(
            action=DecisionAction.OFFSET_STRATEGY,
            category=DecisionCategory.OFFSET_STRATEGY,
            confidence=self._score_to_level(confidence_score),
            confidence_score=confidence_score,
            traffic_light=self._determine_traffic_light(confidence_score, risk_score),
            reasoning=reasoning,
            alternatives=alternatives,
            expected_gain_seconds=pace_delta * 10,  # Rough estimate
            risk_score=risk_score,
            priority=self.priority,
            metadata={
                'module': self.name,
                'version': self.version,
                'offset_opportunities': len(offset_opportunities),
            },
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
