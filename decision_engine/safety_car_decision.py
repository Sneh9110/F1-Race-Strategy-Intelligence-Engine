"""Safety car decision module."""

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


class SafetyCarDecision(BaseDecisionModule):
    """Decision module for safety car decisions (pit under SC or stay out)."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize safety car module."""
        super().__init__(config_path)
        
        module_config = self.config.get('modules', {}).get('safety_car_decision', {}).get('config', {})
        self.tire_age_threshold = module_config.get('tire_age_threshold', 15)
        self.recent_pit_threshold = module_config.get('recent_pit_threshold', 5)
        self.tire_life_threshold = module_config.get('tire_life_threshold', 20)
        self.rival_pit_percentage_threshold = module_config.get('rival_pit_percentage_threshold', 0.5)
    
    @property
    def name(self) -> str:
        return "safety_car_decision"
    
    @property
    def version(self) -> str:
        return "v1.0.0"
    
    @property
    def category(self) -> str:
        return DecisionCategory.SAFETY_CAR.value
    
    @property
    def priority(self) -> int:
        return 10  # Highest priority (time-critical)
    
    def is_applicable(self, context: DecisionContext) -> bool:
        """Check if safety car decision is applicable."""
        # Only applicable when SC active or imminent
        sc_probability = 0.0
        if hasattr(self, '_current_input') and self._current_input:
            sc_probability = self._current_input.feature_data.get('sc_probability', 0.0)
        
        return context.safety_car_active or sc_probability > 0.5
    
    def get_confidence(self, context: DecisionContext) -> float:
        """Calculate confidence for safety car decision."""
        factors = {}
        
        # SC active = higher confidence
        factors['sc_active'] = 1.0 if context.safety_car_active else 0.6
        
        # Tire age = more data = higher confidence
        factors['tire_age'] = min(1.0, context.tire_age / 15.0)
        
        factors['default'] = 0.8  # SC decisions generally high confidence
        
        weights = {
            'sc_active': 0.5,
            'tire_age': 0.2,
            'default': 0.3,
        }
        
        return self._calculate_confidence(factors, weights)
    
    def evaluate(self, decision_input: DecisionInput) -> Optional[DecisionRecommendation]:
        """Evaluate safety car decision."""
        # Store for is_applicable check
        self._current_input = decision_input
        
        context = decision_input.context
        
        if not context.safety_car_active:
            return None
        
        factors = []
        rules = []
        model_contributions = {}
        
        # Rule 1: Just pitted (STAY OUT trigger)
        if context.tire_age < self.recent_pit_threshold:
            factors.append(f"Just pitted {context.tire_age} laps ago")
            rules.append("just_pitted")
            return self._build_stay_out_recommendation(context, factors, rules, model_contributions)
        
        # Rule 2: Tire age high (PIT trigger)
        if context.tire_age > self.tire_age_threshold:
            factors.append(f"High tire age: {context.tire_age} laps")
            rules.append("tire_age_high")
        
        # Rule 3: Podium position (STAY OUT trigger - track position critical)
        if context.current_position <= 3:
            factors.append(f"Podium position: P{context.current_position}")
            rules.append("podium_position")
            # Still might pit if tire age very high
            if context.tire_age < 20:
                return self._build_stay_out_recommendation(context, factors, rules, model_contributions)
        
        # Rule 4: Pit loss reduced under SC (~50%)
        pit_loss_reduction = decision_input.feature_data.get('sc_pit_loss_reduction', 0.5)
        factors.append(f"Pit loss reduced by {pit_loss_reduction*100:.0f}% under SC")
        rules.append("pit_loss_reduced")
        
        # Rule 5: Rival strategy analysis
        rivals_pitting_count = 0
        rivals_staying_count = 0
        
        if decision_input.rival_contexts:
            # Simulated rival decisions (would need real-time data in production)
            for rival in decision_input.rival_contexts:
                if rival.rival_tire_age > self.tire_age_threshold:
                    rivals_pitting_count += 1
                else:
                    rivals_staying_count += 1
            
            total_rivals = len(decision_input.rival_contexts)
            rivals_pitting_pct = rivals_pitting_count / total_rivals if total_rivals > 0 else 0
            
            if rivals_pitting_pct > self.rival_pit_percentage_threshold:
                factors.append(f"{rivals_pitting_pct*100:.0f}% of rivals expected to pit")
                rules.append("rivals_pitting")
                # Stay out to gain track position
                return self._build_stay_out_recommendation(context, factors, rules, model_contributions)
            else:
                factors.append(f"Only {rivals_pitting_pct*100:.0f}% of rivals expected to pit")
                rules.append("rivals_staying_out")
        
        # Default: Pit under SC if tire age moderate-high
        if context.tire_age > 10:
            return self._build_pit_under_sc_recommendation(context, factors, rules, model_contributions)
        else:
            return self._build_stay_out_recommendation(context, factors, rules, model_contributions)
    
    def _build_pit_under_sc_recommendation(
        self,
        context: DecisionContext,
        factors: List[str],
        rules: List[str],
        model_contributions: Dict[str, float]
    ) -> DecisionRecommendation:
        """Build PIT_UNDER_SC recommendation."""
        confidence_score = self.get_confidence(context)
        risk_score = 0.3  # Lower risk under SC
        
        reasoning = DecisionReasoning(
            primary_factors=factors[:5],
            rule_triggers=rules,
            model_contributions=model_contributions,
            risk_assessment="Low risk: Reduced pit time loss under SC",
            opportunity_assessment="Opportunity: Cheap pit stop, fresh tires",
        )
        
        alternatives = [
            AlternativeOption(
                action=DecisionAction.STAY_OUT_SC,
                expected_outcome="Stay out, maintain track position",
                confidence=0.6,
                pros=["Keep track position", "No time loss"],
                cons=["Old tires", "Rivals gain with fresh tires"],
            ),
        ]
        
        return DecisionRecommendation(
            action=DecisionAction.PIT_UNDER_SC,
            category=DecisionCategory.SAFETY_CAR,
            confidence=self._score_to_level(confidence_score),
            confidence_score=confidence_score,
            traffic_light=self._determine_traffic_light(confidence_score, risk_score),
            reasoning=reasoning,
            alternatives=alternatives,
            expected_gain_seconds=8.0,  # SC pit is ~half normal pit loss
            risk_score=risk_score,
            priority=self.priority,
            metadata={'module': self.name, 'version': self.version},
        )
    
    def _build_stay_out_recommendation(
        self,
        context: DecisionContext,
        factors: List[str],
        rules: List[str],
        model_contributions: Dict[str, float]
    ) -> DecisionRecommendation:
        """Build STAY_OUT_SC recommendation."""
        confidence_score = self.get_confidence(context)
        risk_score = 0.4  # Medium risk
        
        reasoning = DecisionReasoning(
            primary_factors=factors[:5],
            rule_triggers=rules,
            model_contributions=model_contributions,
            risk_assessment="Medium risk: Old tires vs rivals with fresh",
            opportunity_assessment="Opportunity: Gain positions, rivals pit",
        )
        
        alternatives = [
            AlternativeOption(
                action=DecisionAction.PIT_UNDER_SC,
                expected_outcome="Pit under SC, lose track position",
                confidence=0.5,
                pros=["Fresh tires", "Reduced pit loss"],
                cons=["Lose track position", "Traffic on exit"],
            ),
        ]
        
        return DecisionRecommendation(
            action=DecisionAction.STAY_OUT_SC,
            category=DecisionCategory.SAFETY_CAR,
            confidence=self._score_to_level(confidence_score),
            confidence_score=confidence_score,
            traffic_light=self._determine_traffic_light(confidence_score, risk_score),
            reasoning=reasoning,
            alternatives=alternatives,
            expected_gain_seconds=2.0,
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
