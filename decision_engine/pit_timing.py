"""Pit timing decision module."""

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


class PitTimingDecision(BaseDecisionModule):
    """Decision module for pit timing (when to pit NOW vs wait)."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize pit timing module."""
        super().__init__(config_path)
        
        # Load thresholds from config
        module_config = self.config.get('modules', {}).get('pit_timing', {}).get('config', {})
        self.tire_age_threshold = module_config.get('tire_age_threshold', 0.9)
        self.degradation_threshold = module_config.get('degradation_threshold', 0.1)
        self.pace_drop_threshold = module_config.get('pace_drop_threshold', 0.5)
        self.pit_window_min_lap = module_config.get('pit_window_min_lap', 6)
        self.pit_window_max_offset = module_config.get('pit_window_max_offset', 5)
        self.undercut_gap_threshold = module_config.get('undercut_gap_threshold', 3.0)
        self.sc_probability_threshold = module_config.get('sc_probability_threshold', 0.6)
    
    @property
    def name(self) -> str:
        return "pit_timing"
    
    @property
    def version(self) -> str:
        return "v1.0.0"
    
    @property
    def category(self) -> str:
        return DecisionCategory.PIT_TIMING.value
    
    @property
    def priority(self) -> int:
        return 9  # High priority
    
    def is_applicable(self, context: DecisionContext) -> bool:
        """Check if pit timing decision is applicable."""
        # Not applicable if just pitted
        if context.tire_age < 5:
            return False
        
        # Not applicable if outside reasonable pit window
        if context.lap_number < self.pit_window_min_lap:
            return False
        
        return True
    
    def get_confidence(self, context: DecisionContext) -> float:
        """Calculate confidence for pit timing decision."""
        factors = {}
        
        # Tire age factor (higher age = higher confidence)
        factors['tire_age'] = min(1.0, context.tire_age / 30.0)
        
        # Lap time consistency (more consistent = higher confidence)
        if len(context.recent_lap_times) >= 5:
            avg = sum(context.recent_lap_times) / len(context.recent_lap_times)
            variance = sum((lt - avg) ** 2 for lt in context.recent_lap_times) / len(context.recent_lap_times)
            factors['consistency'] = max(0.0, 1.0 - variance / 10.0)
        else:
            factors['consistency'] = 0.5
        
        # Pit window factor (inside window = higher confidence)
        max_pit_lap = context.total_laps - self.pit_window_max_offset
        if self.pit_window_min_lap <= context.lap_number <= max_pit_lap:
            factors['pit_window'] = 1.0
        else:
            factors['pit_window'] = 0.5
        
        weights = {
            'tire_age': 0.4,
            'consistency': 0.3,
            'pit_window': 0.3,
        }
        
        return self._calculate_confidence(factors, weights)
    
    def evaluate(self, decision_input: DecisionInput) -> Optional[DecisionRecommendation]:
        """Evaluate pit timing decision."""
        context = decision_input.context
        
        # Track factors and rules
        factors = []
        rules = []
        model_contributions = {}
        
        # Get optimal stint length for compound (from config or defaults)
        optimal_stint_map = {
            'SOFT': 20,
            'MEDIUM': 25,
            'HARD': 30,
        }
        optimal_stint = optimal_stint_map.get(context.tire_compound, 25)
        
        # Rule 1: Tire age exceeds optimal
        tire_age_ratio = context.tire_age / optimal_stint
        if tire_age_ratio > self.tire_age_threshold:
            factors.append(f"Tire age: {context.tire_age} laps (optimal: {optimal_stint})")
            rules.append("tire_age_exceeds_optimal")
        
        # Rule 2: Pace drop detected
        pace_drop = 0.0
        if len(context.recent_lap_times) >= 5:
            early_avg = sum(context.recent_lap_times[:3]) / 3
            recent_avg = sum(context.recent_lap_times[-3:]) / 3
            pace_drop = recent_avg - early_avg
            
            if pace_drop > self.pace_drop_threshold:
                factors.append(f"Pace drop: {pace_drop:.2f}s")
                rules.append("pace_drop_detected")
        
        # Rule 3: Tire cliff (sudden 1s+ drop)
        tire_cliff = False
        if len(context.recent_lap_times) >= 2:
            last_delta = context.recent_lap_times[-1] - context.recent_lap_times[-2]
            if last_delta > 1.0:
                factors.append(f"Tire cliff: {last_delta:.2f}s drop")
                rules.append("tire_cliff_detected")
                tire_cliff = True
        
        # Rule 4: Undercut opportunity
        undercut_opportunity = False
        if decision_input.rival_contexts:
            for rival in decision_input.rival_contexts:
                if (rival.rival_position < context.current_position and 
                    rival.gap_to_rival < self.undercut_gap_threshold):
                    factors.append(f"Undercut opportunity vs P{rival.rival_position} (gap: {rival.gap_to_rival:.1f}s)")
                    rules.append("undercut_opportunity")
                    undercut_opportunity = True
                    break
        
        # Rule 5: SC imminent (STAY OUT trigger)
        sc_imminent = False
        if decision_input.feature_data.get('sc_probability', 0.0) > self.sc_probability_threshold:
            factors.append(f"SC probability: {decision_input.feature_data['sc_probability']:.2f}")
            rules.append("sc_imminent")
            sc_imminent = True
        
        # Rule 6: Just pitted (STAY OUT trigger)
        if context.tire_age < 5:
            factors.append(f"Just pitted: {context.tire_age} laps ago")
            rules.append("just_pitted")
            return self._build_stay_out_recommendation(context, factors, rules, model_contributions)
        
        # Decide action
        action = None
        
        # PIT NOW triggers
        if (tire_age_ratio > self.tire_age_threshold or 
            pace_drop > self.pace_drop_threshold or
            tire_cliff or
            undercut_opportunity):
            
            # But not if SC imminent
            if not sc_imminent:
                action = DecisionAction.PIT_NOW
        
        # STAY OUT triggers
        elif sc_imminent or context.tire_age < optimal_stint * 0.7:
            action = DecisionAction.STAY_OUT
        
        if action is None:
            return None
        
        # Build recommendation
        if action == DecisionAction.PIT_NOW:
            return self._build_pit_now_recommendation(context, factors, rules, model_contributions)
        else:
            return self._build_stay_out_recommendation(context, factors, rules, model_contributions)
    
    def _build_pit_now_recommendation(
        self,
        context: DecisionContext,
        factors: List[str],
        rules: List[str],
        model_contributions: Dict[str, float]
    ) -> DecisionRecommendation:
        """Build PIT_NOW recommendation."""
        confidence_score = self.get_confidence(context)
        
        # Calculate risk (pitting is medium risk)
        risk_score = 0.4
        
        # Build reasoning
        reasoning = DecisionReasoning(
            primary_factors=factors[:5],
            rule_triggers=rules,
            model_contributions=model_contributions,
            risk_assessment="Medium risk: Pit stop time loss, potential traffic",
            opportunity_assessment="Opportunity: Fresh tires, avoid tire cliff",
        )
        
        # Build alternatives
        alternatives = [
            AlternativeOption(
                action=DecisionAction.STAY_OUT,
                expected_outcome="Extend stint by 2-3 laps, risk tire cliff",
                confidence=max(0.3, confidence_score - 0.2),
                pros=["Avoid pit time loss", "Possible overcut opportunity"],
                cons=["Risk tire degradation", "Lose pace"],
            ),
        ]
        
        return DecisionRecommendation(
            action=DecisionAction.PIT_NOW,
            category=DecisionCategory.PIT_TIMING,
            confidence=self._score_to_level(confidence_score),
            confidence_score=confidence_score,
            traffic_light=self._determine_traffic_light(confidence_score, risk_score),
            reasoning=reasoning,
            alternatives=alternatives,
            expected_gain_seconds=3.5,
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
        """Build STAY_OUT recommendation."""
        confidence_score = self.get_confidence(context)
        
        # Calculate risk (staying out is lower risk)
        risk_score = 0.3
        
        # Build reasoning
        reasoning = DecisionReasoning(
            primary_factors=factors[:5],
            rule_triggers=rules,
            model_contributions=model_contributions,
            risk_assessment="Low risk: Tire life sufficient, track position maintained",
            opportunity_assessment="Opportunity: Possible overcut, wait for SC",
        )
        
        # Build alternatives
        alternatives = [
            AlternativeOption(
                action=DecisionAction.PIT_NOW,
                expected_outcome="Pit now, suboptimal timing",
                confidence=max(0.3, confidence_score - 0.3),
                pros=["Fresh tires", "Avoid later congestion"],
                cons=["Suboptimal timing", "Lose track position"],
            ),
        ]
        
        return DecisionRecommendation(
            action=DecisionAction.STAY_OUT,
            category=DecisionCategory.PIT_TIMING,
            confidence=self._score_to_level(confidence_score),
            confidence_score=confidence_score,
            traffic_light=self._determine_traffic_light(confidence_score, risk_score),
            reasoning=reasoning,
            alternatives=alternatives,
            expected_gain_seconds=1.5,
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
