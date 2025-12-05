"""Undercut/overcut decision module."""

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


class UndercutOvercutDecision(BaseDecisionModule):
    """Decision module for undercut/overcut opportunities."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize undercut/overcut module."""
        super().__init__(config_path)
        
        module_config = self.config.get('modules', {}).get('undercut_overcut', {}).get('config', {})
        self.undercut_gap_threshold = module_config.get('undercut_gap_threshold', 3.0)
        self.overcut_gap_threshold = module_config.get('overcut_gap_threshold', 5.0)
        self.tire_age_delta_threshold = module_config.get('tire_age_delta_threshold', 5)
        self.tire_life_threshold = module_config.get('tire_life_threshold', 10)
        self.undercut_timing_offset = module_config.get('undercut_timing_offset', 2)
        self.overcut_timing_offset = module_config.get('overcut_timing_offset', 4)
    
    @property
    def name(self) -> str:
        return "undercut_overcut"
    
    @property
    def version(self) -> str:
        return "v1.0.0"
    
    @property
    def category(self) -> str:
        return DecisionCategory.UNDERCUT_OVERCUT.value
    
    @property
    def priority(self) -> int:
        return 8  # High priority (tactical)
    
    def is_applicable(self, context: DecisionContext) -> bool:
        """Check if undercut/overcut decision is applicable."""
        # Need rival data
        return True
    
    def get_confidence(self, context: DecisionContext) -> float:
        """Calculate confidence for undercut/overcut."""
        factors = {}
        
        # Gap stability (consistent gaps = higher confidence)
        if context.gap_to_ahead is not None:
            # Assume stable gap (would need historical data)
            factors['gap_stability'] = 0.8
        else:
            factors['gap_stability'] = 0.3
        
        # Tire age (more laps = more data = higher confidence)
        factors['tire_age'] = min(1.0, context.tire_age / 15.0)
        
        factors['default'] = 0.7
        
        weights = {
            'gap_stability': 0.4,
            'tire_age': 0.3,
            'default': 0.3,
        }
        
        return self._calculate_confidence(factors, weights)
    
    def evaluate(self, decision_input: DecisionInput) -> Optional[DecisionRecommendation]:
        """Evaluate undercut/overcut decision."""
        context = decision_input.context
        
        if not decision_input.rival_contexts:
            return None
        
        factors = []
        rules = []
        model_contributions = {}
        
        # Look for undercut opportunities (pit before rival)
        for rival in decision_input.rival_contexts:
            # Undercut: rival ahead, within gap threshold
            if (rival.rival_position < context.current_position and 
                rival.gap_to_rival <= self.undercut_gap_threshold):
                
                tire_age_delta = rival.rival_tire_age - context.tire_age
                
                # Rule 1: Rival has older tires
                if tire_age_delta > self.tire_age_delta_threshold:
                    factors.append(
                        f"Undercut opportunity vs P{rival.rival_position}: "
                        f"gap={rival.gap_to_rival:.1f}s, rival tires {tire_age_delta} laps older"
                    )
                    rules.append("undercut_window_open")
                    
                    # Check pit loss vs gap
                    pit_loss = decision_input.feature_data.get('pit_loss', 22.0)
                    if pit_loss < rival.gap_to_rival:
                        factors.append(f"Pit loss ({pit_loss:.1f}s) < gap ({rival.gap_to_rival:.1f}s)")
                        rules.append("pit_loss_within_gap")
                        
                        return self._build_undercut_recommendation(
                            context, rival, factors, rules, model_contributions
                        )
            
            # Overcut: rival just pitted, we can extend
            if (rival.rival_tire_age < 5 and 
                context.tire_age > self.tire_life_threshold):
                
                # Rule 2: Rival just pitted, we have tire life
                remaining_life = decision_input.feature_data.get('remaining_tire_life', 15)
                if remaining_life > self.tire_life_threshold:
                    factors.append(
                        f"Overcut opportunity vs P{rival.rival_position}: "
                        f"rival just pitted, we have {remaining_life} laps remaining"
                    )
                    rules.append("overcut_window_open")
                    
                    if rival.gap_to_rival <= self.overcut_gap_threshold:
                        factors.append(f"Gap manageable: {rival.gap_to_rival:.1f}s")
                        rules.append("gap_within_overcut_range")
                        
                        return self._build_overcut_recommendation(
                            context, rival, factors, rules, model_contributions
                        )
        
        return None
    
    def _build_undercut_recommendation(
        self,
        context: DecisionContext,
        rival: Any,
        factors: List[str],
        rules: List[str],
        model_contributions: Dict[str, float]
    ) -> DecisionRecommendation:
        """Build UNDERCUT_NOW recommendation."""
        confidence_score = self.get_confidence(context)
        risk_score = 0.4  # Medium risk
        
        reasoning = DecisionReasoning(
            primary_factors=factors[:5],
            rule_triggers=rules,
            model_contributions=model_contributions,
            risk_assessment="Medium risk: Requires track position gain from fresh tires",
            opportunity_assessment=f"Opportunity: Overtake P{rival.rival_position} via undercut",
        )
        
        alternatives = [
            AlternativeOption(
                action=DecisionAction.STAY_OUT,
                expected_outcome="Wait 1-2 laps, safer undercut timing",
                confidence=0.7,
                pros=["More data on rival", "Safer timing"],
                cons=["Risk rival pits first", "Miss window"],
            ),
            AlternativeOption(
                action=DecisionAction.NO_ACTION,
                expected_outcome="No undercut, match rival strategy",
                confidence=0.5,
                pros=["Lower risk", "Follow rival"],
                cons=["No position gain", "Reactive strategy"],
            ),
        ]
        
        return DecisionRecommendation(
            action=DecisionAction.UNDERCUT_NOW,
            category=DecisionCategory.UNDERCUT_OVERCUT,
            confidence=self._score_to_level(confidence_score),
            confidence_score=confidence_score,
            traffic_light=self._determine_traffic_light(confidence_score, risk_score),
            reasoning=reasoning,
            alternatives=alternatives,
            expected_gain_seconds=2.5,  # Expected undercut gain
            risk_score=risk_score,
            priority=self.priority,
            metadata={
                'module': self.name,
                'version': self.version,
                'rival_position': rival.rival_position,
                'gap': rival.gap_to_rival,
            },
        )
    
    def _build_overcut_recommendation(
        self,
        context: DecisionContext,
        rival: Any,
        factors: List[str],
        rules: List[str],
        model_contributions: Dict[str, float]
    ) -> DecisionRecommendation:
        """Build OVERCUT_NOW recommendation."""
        confidence_score = self.get_confidence(context)
        risk_score = 0.5  # Medium-high risk
        
        reasoning = DecisionReasoning(
            primary_factors=factors[:5],
            rule_triggers=rules,
            model_contributions=model_contributions,
            risk_assessment="Medium-high risk: Requires extending stint without pace loss",
            opportunity_assessment=f"Opportunity: Overcut P{rival.rival_position} by extending",
        )
        
        alternatives = [
            AlternativeOption(
                action=DecisionAction.PIT_NOW,
                expected_outcome="Pit to match rival, no overcut",
                confidence=0.6,
                pros=["Match rival strategy", "Fresh tires"],
                cons=["No strategic advantage", "Reactive"],
            ),
            AlternativeOption(
                action=DecisionAction.UNDERCUT_NOW,
                expected_outcome="Pit early for undercut (risky)",
                confidence=0.5,
                pros=["Aggressive strategy", "Gain track position"],
                cons=["High risk", "May not work"],
            ),
        ]
        
        return DecisionRecommendation(
            action=DecisionAction.OVERCUT_NOW,
            category=DecisionCategory.UNDERCUT_OVERCUT,
            confidence=self._score_to_level(confidence_score),
            confidence_score=confidence_score,
            traffic_light=self._determine_traffic_light(confidence_score, risk_score),
            reasoning=reasoning,
            alternatives=alternatives,
            expected_gain_seconds=3.0,  # Expected overcut gain
            risk_score=risk_score,
            priority=self.priority,
            metadata={
                'module': self.name,
                'version': self.version,
                'rival_position': rival.rival_position,
                'gap': rival.gap_to_rival,
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
