"""Rain strategy decision module."""

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


class RainStrategyDecision(BaseDecisionModule):
    """Decision module for rain strategy (switch to intermediates/wets)."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize rain strategy module."""
        super().__init__(config_path)
        
        module_config = self.config.get('modules', {}).get('rain_strategy', {}).get('config', {})
        self.track_temp_drop_threshold = module_config.get('track_temp_drop_threshold', 5.0)
        self.lap_time_increase_threshold = module_config.get('lap_time_increase_threshold', 2.0)
        self.heavy_rain_threshold = module_config.get('heavy_rain_threshold', 5.0)
        self.weather_data_max_age = module_config.get('weather_data_max_age', 60)
        self.crossover_point_confidence = module_config.get('crossover_point_confidence', 0.7)
    
    @property
    def name(self) -> str:
        return "rain_strategy"
    
    @property
    def version(self) -> str:
        return "v1.0.0"
    
    @property
    def category(self) -> str:
        return DecisionCategory.RAIN.value
    
    @property
    def priority(self) -> int:
        return 10  # Highest priority (weather critical)
    
    def is_applicable(self, context: DecisionContext) -> bool:
        """Check if rain strategy is applicable."""
        # Check for rain indicators in feature data
        if hasattr(self, '_current_input') and self._current_input:
            rain_intensity = self._current_input.feature_data.get('rain_intensity', 0.0)
            track_wetness = self._current_input.feature_data.get('track_wetness', 0.0)
            return rain_intensity > 0.1 or track_wetness > 0.2
        return False
    
    def get_confidence(self, context: DecisionContext) -> float:
        """Calculate confidence for rain strategy."""
        factors = {}
        
        # Weather data recency
        weather_age = 30  # Assume recent (would get from feature_data)
        factors['weather_recency'] = max(0.0, 1.0 - weather_age / self.weather_data_max_age)
        
        # Lap time delta clarity
        if len(context.recent_lap_times) >= 3:
            last_delta = context.recent_lap_times[-1] - context.recent_lap_times[-3]
            if abs(last_delta) > self.lap_time_increase_threshold:
                factors['lap_time_clarity'] = 1.0
            else:
                factors['lap_time_clarity'] = 0.6
        else:
            factors['lap_time_clarity'] = 0.5
        
        factors['default'] = 0.7
        
        weights = {
            'weather_recency': 0.4,
            'lap_time_clarity': 0.3,
            'default': 0.3,
        }
        
        return self._calculate_confidence(factors, weights)
    
    def evaluate(self, decision_input: DecisionInput) -> Optional[DecisionRecommendation]:
        """Evaluate rain strategy decision."""
        self._current_input = decision_input
        context = decision_input.context
        
        factors = []
        rules = []
        model_contributions = {}
        
        # Get weather data from features
        rain_intensity = decision_input.feature_data.get('rain_intensity', 0.0)
        track_wetness = decision_input.feature_data.get('track_wetness', 0.0)
        
        if rain_intensity < 0.1 and track_wetness < 0.2:
            return None
        
        # Current tire type
        current_tire = context.tire_compound
        
        # Rule 1: Rain detected
        if rain_intensity > 0.2:
            factors.append(f"Rain intensity: {rain_intensity:.2f}")
            rules.append("rain_detected")
        
        # Rule 2: Track temp dropping
        if context.track_temp and context.weather_temp:
            temp_delta = context.track_temp - context.weather_temp
            if temp_delta > self.track_temp_drop_threshold:
                factors.append(f"Track temp dropping: {temp_delta:.1f}°C delta")
                rules.append("track_temp_dropping")
        
        # Rule 3: Lap times increasing
        if len(context.recent_lap_times) >= 3:
            early_avg = sum(context.recent_lap_times[:2]) / 2
            recent_avg = sum(context.recent_lap_times[-2:]) / 2
            lap_time_increase = recent_avg - early_avg
            
            if lap_time_increase > self.lap_time_increase_threshold:
                factors.append(f"Lap times increasing: {lap_time_increase:.2f}s")
                rules.append("lap_times_increasing")
                
                # Decide: Inters or Wets?
                if lap_time_increase > self.heavy_rain_threshold:
                    factors.append("Heavy rain: Wets recommended")
                    rules.append("heavy_rain")
                    action = DecisionAction.SWITCH_TO_WETS
                elif current_tire in ['SOFT', 'MEDIUM', 'HARD']:
                    action = DecisionAction.SWITCH_TO_INTERS
                else:
                    return None  # Already on inters/wets
                
                # Rule 4: Rivals switching
                rivals_on_rain_tires = 0
                if decision_input.rival_contexts:
                    for rival in decision_input.rival_contexts:
                        if rival.rival_tire_compound in ['INTERMEDIATE', 'WET']:
                            rivals_on_rain_tires += 1
                    
                    if rivals_on_rain_tires > 0:
                        factors.append(f"{rivals_on_rain_tires} rival(s) switched to rain tires")
                        rules.append("rivals_switching")
                
                return self._build_rain_recommendation(
                    context, action, factors, rules, model_contributions
                )
        
        return None
    
    def _build_rain_recommendation(
        self,
        context: DecisionContext,
        action: DecisionAction,
        factors: List[str],
        rules: List[str],
        model_contributions: Dict[str, float]
    ) -> DecisionRecommendation:
        """Build rain tire recommendation."""
        confidence_score = self.get_confidence(context)
        risk_score = 0.5  # Medium-high risk (weather changes unpredictable)
        
        tire_type = "intermediates" if action == DecisionAction.SWITCH_TO_INTERS else "wets"
        
        reasoning = DecisionReasoning(
            primary_factors=factors[:5],
            rule_triggers=rules,
            model_contributions=model_contributions,
            risk_assessment=f"Medium risk: Weather unpredictable, {tire_type} may be early/late",
            opportunity_assessment=f"Opportunity: Gain positions with correct tire choice",
        )
        
        # Build alternatives
        if action == DecisionAction.SWITCH_TO_INTERS:
            alternatives = [
                AlternativeOption(
                    action=DecisionAction.STAY_OUT,
                    expected_outcome="Stay on slicks, risk in wet",
                    confidence=0.3,
                    pros=["No pit time loss", "Better if rain stops"],
                    cons=["Dangerous in wet", "Lose massive time"],
                ),
                AlternativeOption(
                    action=DecisionAction.SWITCH_TO_WETS,
                    expected_outcome="Switch to wets, overkill if light rain",
                    confidence=0.5,
                    pros=["Safe in heavy rain"],
                    cons=["Slow if rain is light", "Wasted pit stop"],
                ),
            ]
        else:  # WETS
            alternatives = [
                AlternativeOption(
                    action=DecisionAction.SWITCH_TO_INTERS,
                    expected_outcome="Switch to inters, risk if heavy rain",
                    confidence=0.6,
                    pros=["Faster if moderate rain"],
                    cons=["Insufficient if heavy rain"],
                ),
            ]
        
        return DecisionRecommendation(
            action=action,
            category=DecisionCategory.RAIN,
            confidence=self._score_to_level(confidence_score),
            confidence_score=confidence_score,
            traffic_light=self._determine_traffic_light(confidence_score, risk_score),
            reasoning=reasoning,
            alternatives=alternatives,
            expected_gain_seconds=10.0,  # Huge gain if correct
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
