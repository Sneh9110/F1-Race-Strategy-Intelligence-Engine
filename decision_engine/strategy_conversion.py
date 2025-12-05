"""Strategy conversion decision module."""

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


class StrategyConversionDecision(BaseDecisionModule):
    """Decision module for strategy conversion (1-stop → 2-stop → 3-stop)."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize strategy conversion module."""
        super().__init__(config_path)
        
        module_config = self.config.get('modules', {}).get('strategy_conversion', {}).get('config', {})
        self.degradation_deviation_threshold = module_config.get('degradation_deviation_threshold', 0.2)
        self.pace_gain_threshold = module_config.get('pace_gain_threshold', 0.3)
        self.position_delta_threshold = module_config.get('position_delta_threshold', 2)
        self.time_delta_threshold = module_config.get('time_delta_threshold', 5.0)
    
    @property
    def name(self) -> str:
        return "strategy_conversion"
    
    @property
    def version(self) -> str:
        return "v1.0.0"
    
    @property
    def category(self) -> str:
        return DecisionCategory.STRATEGY_CONVERSION.value
    
    @property
    def priority(self) -> int:
        return 7  # Medium-high priority
    
    def is_applicable(self, context: DecisionContext) -> bool:
        """Check if strategy conversion is applicable."""
        # Only applicable mid-race (laps 10-70% of race)
        if context.lap_number < 10 or context.lap_number > context.total_laps * 0.7:
            return False
        return True
    
    def get_confidence(self, context: DecisionContext) -> float:
        """Calculate confidence for strategy conversion."""
        factors = {}
        
        # More pit stops = more data = higher confidence
        factors['pit_data'] = min(1.0, context.pit_stops_completed / 2.0)
        
        # Mid-race = higher confidence (more time to execute)
        race_progress = context.lap_number / context.total_laps
        if 0.3 <= race_progress <= 0.7:
            factors['race_progress'] = 1.0
        else:
            factors['race_progress'] = 0.6
        
        factors['default'] = 0.7
        
        weights = {
            'pit_data': 0.3,
            'race_progress': 0.3,
            'default': 0.4,
        }
        
        return self._calculate_confidence(factors, weights)
    
    def evaluate(self, decision_input: DecisionInput) -> Optional[DecisionRecommendation]:
        """Evaluate strategy conversion decision."""
        context = decision_input.context
        simulation_context = decision_input.simulation_context
        
        if not simulation_context or not simulation_context.strategy_rankings:
            logger.debug("No simulation context, skipping strategy conversion")
            return None
        
        factors = []
        rules = []
        model_contributions = {}
        
        # Determine current strategy (from pit stops)
        current_stops = context.pit_stops_completed
        
        # Rule 1: Degradation higher than expected
        expected_deg = decision_input.feature_data.get('expected_degradation', 0.05)
        actual_deg = decision_input.feature_data.get('actual_degradation', 0.05)
        
        if actual_deg > expected_deg * (1 + self.degradation_deviation_threshold):
            factors.append(f"Degradation {(actual_deg/expected_deg - 1)*100:.0f}% above expected")
            rules.append("degradation_exceeds_expected")
        
        # Rule 2: SC deployed (can pit cheaply)
        if context.safety_car_active:
            factors.append("Safety car deployed - cheaper pit stops")
            rules.append("sc_deployed")
        
        # Rule 3: Rivals on different strategy gaining pace
        rival_pace_advantage = False
        if decision_input.rival_contexts:
            for rival in decision_input.rival_contexts:
                if rival.rival_pit_stops > current_stops:
                    # Rival on more aggressive strategy
                    factors.append(f"P{rival.rival_position} on {rival.rival_pit_stops+1}-stop strategy")
                    rules.append("rivals_on_different_strategy")
                    rival_pace_advantage = True
                    break
        
        # Compare strategies from simulation
        current_strategy_stops = current_stops + 1  # Total stops planned
        
        # Find best alternative strategy
        best_alternative = None
        best_delta = 0.0
        
        for ranking in simulation_context.strategy_rankings:
            strategy_stops = len(ranking.get('pit_laps', []))
            if strategy_stops != current_strategy_stops:
                position_delta = simulation_context.expected_position - ranking.get('expected_position', 20)
                if position_delta > best_delta:
                    best_delta = position_delta
                    best_alternative = strategy_stops
        
        # Decide action
        action = None
        
        if best_alternative and best_delta >= self.position_delta_threshold:
            if best_alternative == 1:
                action = DecisionAction.SWITCH_TO_ONE_STOP
            elif best_alternative == 2:
                action = DecisionAction.SWITCH_TO_TWO_STOP
            elif best_alternative == 3:
                action = DecisionAction.SWITCH_TO_THREE_STOP
            
            factors.append(f"{best_alternative}-stop expected to gain {best_delta:.1f} positions")
            rules.append("alternative_strategy_better")
        
        if action is None:
            return None
        
        # Build recommendation
        confidence_score = self.get_confidence(context)
        risk_score = 0.5  # Strategy changes are medium-high risk
        
        reasoning = DecisionReasoning(
            primary_factors=factors[:5],
            rule_triggers=rules,
            model_contributions=model_contributions,
            risk_assessment="Medium risk: Strategy change requires execution",
            opportunity_assessment=f"Opportunity: Gain {best_delta:.1f} positions with {best_alternative}-stop",
        )
        
        alternatives = [
            AlternativeOption(
                action=DecisionAction.NO_ACTION,
                expected_outcome=f"Continue current {current_strategy_stops}-stop strategy",
                confidence=0.7,
                pros=["No execution risk", "Strategy already planned"],
                cons=["Potential position loss", "Suboptimal strategy"],
            ),
        ]
        
        return DecisionRecommendation(
            action=action,
            category=DecisionCategory.STRATEGY_CONVERSION,
            confidence=self._score_to_level(confidence_score),
            confidence_score=confidence_score,
            traffic_light=self._determine_traffic_light(confidence_score, risk_score),
            reasoning=reasoning,
            alternatives=alternatives,
            expected_gain_seconds=best_delta * 5.0,  # Rough conversion
            risk_score=risk_score,
            priority=self.priority,
            metadata={'module': self.name, 'version': self.version},
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
