"""Base class for decision modules."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import time
import yaml
from pathlib import Path

try:
    from circuitbreaker import circuit
except ImportError:
    # Fallback if circuitbreaker not installed
    def circuit(failure_threshold=5, recovery_timeout=60):
        def decorator(func):
            return func
        return decorator

from decision_engine.schemas import (
    DecisionInput,
    DecisionRecommendation,
    DecisionContext,
    DecisionReasoning,
    TrafficLight,
    ConfidenceLevel,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BaseDecisionModule(ABC):
    """Abstract base class for decision modules."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize decision module."""
        self.config = self._load_config(config_path) if config_path else {}
        self.evaluation_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.latencies = []
        
        logger.info(f"Initialized {self.name} v{self.version}")
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Module name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Module version."""
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        """Decision category."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Module priority (1-10, higher = more important)."""
        pass
    
    @property
    def enabled(self) -> bool:
        """Whether module is enabled."""
        return self.config.get('enabled', True)
    
    @abstractmethod
    def evaluate(self, decision_input: DecisionInput) -> Optional[DecisionRecommendation]:
        """
        Evaluate decision input and return recommendation.
        
        Args:
            decision_input: Complete decision context
            
        Returns:
            DecisionRecommendation if applicable, None otherwise
        """
        pass
    
    @abstractmethod
    def get_confidence(self, context: DecisionContext) -> float:
        """
        Calculate confidence score for current context.
        
        Args:
            context: Decision context
            
        Returns:
            Confidence score (0-1)
        """
        pass
    
    @abstractmethod
    def is_applicable(self, context: DecisionContext) -> bool:
        """
        Check if module should run for current context.
        
        Args:
            context: Decision context
            
        Returns:
            True if module is applicable
        """
        pass
    
    def validate_input(self, decision_input: DecisionInput) -> bool:
        """
        Validate decision input.
        
        Args:
            decision_input: Input to validate
            
        Returns:
            True if valid
        """
        try:
            # Pydantic validation already done, just check required fields
            if not decision_input.context:
                return False
            return True
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    @circuit(failure_threshold=5, recovery_timeout=60)
    def evaluate_safe(self, decision_input: DecisionInput) -> Optional[DecisionRecommendation]:
        """
        Evaluate with circuit breaker protection.
        
        Args:
            decision_input: Complete decision context
            
        Returns:
            DecisionRecommendation or None
        """
        start_time = time.time()
        self.evaluation_count += 1
        
        try:
            if not self.validate_input(decision_input):
                logger.warning(f"{self.name}: Invalid input")
                return None
            
            if not self.is_applicable(decision_input.context):
                logger.debug(f"{self.name}: Not applicable")
                return None
            
            recommendation = self.evaluate(decision_input)
            
            latency_ms = (time.time() - start_time) * 1000
            self.latencies.append(latency_ms)
            
            if recommendation:
                self.success_count += 1
                logger.info(
                    f"{self.name}: {recommendation.action.value} "
                    f"(confidence={recommendation.confidence_score:.2f}, "
                    f"latency={latency_ms:.1f}ms)"
                )
            else:
                logger.debug(f"{self.name}: No recommendation")
            
            return recommendation
            
        except Exception as e:
            self.failure_count += 1
            logger.error(f"{self.name}: Evaluation failed: {e}", exc_info=True)
            return self.get_fallback_recommendation()
    
    def get_fallback_recommendation(self) -> Optional[DecisionRecommendation]:
        """
        Fallback recommendation when module fails.
        
        Returns:
            Basic recommendation or None
        """
        logger.warning(f"{self.name}: Using fallback")
        return None
    
    def _calculate_confidence(
        self, 
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
            return 0.5  # Default confidence
        
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
    
    def _determine_traffic_light(
        self, 
        confidence: float, 
        risk_score: float
    ) -> TrafficLight:
        """
        Determine traffic light based on confidence and risk.
        
        Args:
            confidence: Confidence score (0-1)
            risk_score: Risk score (0-1)
            
        Returns:
            TrafficLight enum
        """
        # GREEN: High confidence, low risk
        if confidence >= 0.8 and risk_score <= 0.3:
            return TrafficLight.GREEN
        
        # RED: Low confidence or high risk
        if confidence < 0.5 or risk_score > 0.7:
            # Safety car decisions never RED (too urgent)
            if self.category == "safety_car":
                return TrafficLight.AMBER
            return TrafficLight.RED
        
        # AMBER: Medium confidence/risk
        return TrafficLight.AMBER
    
    def _build_reasoning(
        self,
        factors: List[str],
        rules: List[str],
        models: Dict[str, float]
    ) -> DecisionReasoning:
        """
        Build decision reasoning.
        
        Args:
            factors: Primary factors
            rules: Rule triggers
            models: Model contributions
            
        Returns:
            DecisionReasoning
        """
        return DecisionReasoning(
            primary_factors=factors[:5],  # Top 5
            rule_triggers=rules,
            model_contributions=models,
            risk_assessment="",
            opportunity_assessment="",
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load module configuration from YAML.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Config dict
        """
        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Config file not found: {config_path}")
                return {}
            
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded config from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get module performance statistics.
        
        Returns:
            Stats dict
        """
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            p50_idx = len(sorted_latencies) // 2
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            
            p50 = sorted_latencies[p50_idx]
            p95 = sorted_latencies[p95_idx]
            p99 = sorted_latencies[p99_idx]
        else:
            p50 = p95 = p99 = 0.0
        
        success_rate = (
            self.success_count / self.evaluation_count 
            if self.evaluation_count > 0 
            else 0.0
        )
        
        return {
            'name': self.name,
            'version': self.version,
            'evaluation_count': self.evaluation_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': success_rate,
            'latency_p50_ms': p50,
            'latency_p95_ms': p95,
            'latency_p99_ms': p99,
        }
