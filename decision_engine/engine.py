"""Decision engine orchestrator."""

import time
from typing import List, Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from decision_engine.base import BaseDecisionModule
from decision_engine.schemas import DecisionInput, DecisionOutput, DecisionRecommendation
from decision_engine.registry import DecisionModuleRegistry
from decision_engine.scoring import DecisionRanker
from decision_engine.explainer import DecisionLogger

from app.utils.logger import get_logger

logger = get_logger(__name__)


class DecisionEngine:
    """Central decision engine orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None, enable_cache: bool = True):
        """
        Initialize decision engine.
        
        Args:
            config_path: Path to config YAML
            enable_cache: Whether to enable Redis caching
        """
        self.config_path = config_path
        self.enable_cache = enable_cache
        
        # Initialize registry and load modules
        self.registry = DecisionModuleRegistry()
        self.modules: List[BaseDecisionModule] = []
        self._load_modules()
        
        # Initialize utilities
        self.ranker = DecisionRanker()
        self.decision_logger = DecisionLogger()
        
        # Performance tracking
        self.decision_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Cache (simplified - would use Redis in production)
        self._cache: Dict[str, DecisionOutput] = {}
        
        logger.info(f"Decision Engine initialized with {len(self.modules)} modules")
    
    def _load_modules(self):
        """Load decision modules from registry."""
        from decision_engine.pit_timing import PitTimingDecision
        from decision_engine.strategy_conversion import StrategyConversionDecision
        from decision_engine.offset_strategy import OffsetStrategyDecision
        from decision_engine.safety_car_decision import SafetyCarDecision
        from decision_engine.rain_strategy import RainStrategyDecision
        from decision_engine.pace_monitoring import PaceMonitoringDecision
        from decision_engine.undercut_overcut import UndercutOvercutDecision
        
        # Instantiate modules
        module_classes = [
            PitTimingDecision,
            StrategyConversionDecision,
            OffsetStrategyDecision,
            SafetyCarDecision,
            RainStrategyDecision,
            PaceMonitoringDecision,
            UndercutOvercutDecision,
        ]
        
        for module_class in module_classes:
            try:
                module = module_class(config_path=self.config_path)
                if module.enabled:
                    self.modules.append(module)
                    logger.info(f"Loaded module: {module.name} (priority={module.priority})")
                else:
                    logger.info(f"Skipped disabled module: {module.name}")
            except Exception as e:
                logger.error(f"Failed to load module {module_class.__name__}: {e}")
        
        # Sort by priority (highest first)
        self.modules.sort(key=lambda m: m.priority, reverse=True)
    
    def make_decision(self, decision_input: DecisionInput) -> DecisionOutput:
        """
        Make decision for given input.
        
        Args:
            decision_input: Complete decision context
            
        Returns:
            DecisionOutput with recommendations
        """
        start_time = time.time()
        self.decision_count += 1
        
        context = decision_input.context
        
        # Check cache
        cache_key = f"{context.session_id}:{context.lap_number}:{context.driver_number}"
        if self.enable_cache and cache_key in self._cache:
            self.cache_hits += 1
            cached_output = self._cache[cache_key]
            logger.debug(f"Cache hit for {cache_key}")
            return cached_output
        
        self.cache_misses += 1
        
        # Filter applicable modules
        applicable_modules = [
            m for m in self.modules 
            if m.is_applicable(context)
        ]
        
        logger.info(
            f"Decision for {context.driver_number} lap {context.lap_number}: "
            f"{len(applicable_modules)}/{len(self.modules)} modules applicable"
        )
        
        # Execute modules
        recommendations: List[DecisionRecommendation] = []
        
        for module in applicable_modules:
            try:
                rec = module.evaluate_safe(decision_input)
                if rec:
                    recommendations.append(rec)
            except Exception as e:
                logger.error(f"Module {module.name} failed: {e}", exc_info=True)
        
        # Resolve conflicts and rank
        if recommendations:
            recommendations = self._resolve_conflicts(recommendations)
            recommendations = self.ranker.rank_recommendations(recommendations)
            recommendations = recommendations[:3]  # Top 3
        
        # Build output
        computation_time_ms = (time.time() - start_time) * 1000
        
        output = DecisionOutput(
            recommendations=recommendations,
            session_id=context.session_id,
            lap_number=context.lap_number,
            computation_time_ms=computation_time_ms,
            metadata={
                'driver_number': context.driver_number,
                'applicable_modules': len(applicable_modules),
                'total_recommendations': len(recommendations),
            },
        )
        
        # Cache result
        if self.enable_cache:
            self._cache[cache_key] = output
        
        # Log decision
        self.decision_logger.log_decision(output, context)
        
        logger.info(
            f"Decision completed in {computation_time_ms:.1f}ms: "
            f"{len(recommendations)} recommendation(s)"
        )
        
        return output
    
    async def make_decision_async(self, decision_input: DecisionInput) -> DecisionOutput:
        """
        Make decision asynchronously with parallel module execution.
        
        Args:
            decision_input: Complete decision context
            
        Returns:
            DecisionOutput with recommendations
        """
        start_time = time.time()
        context = decision_input.context
        
        # Filter applicable modules
        applicable_modules = [
            m for m in self.modules 
            if m.is_applicable(context)
        ]
        
        # Execute modules in parallel
        tasks = [
            asyncio.to_thread(m.evaluate_safe, decision_input)
            for m in applicable_modules
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect recommendations
        recommendations: List[DecisionRecommendation] = []
        for result in results:
            if isinstance(result, DecisionRecommendation):
                recommendations.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Async module failed: {result}")
        
        # Rank and build output
        if recommendations:
            recommendations = self._resolve_conflicts(recommendations)
            recommendations = self.ranker.rank_recommendations(recommendations)
            recommendations = recommendations[:3]
        
        computation_time_ms = (time.time() - start_time) * 1000
        
        output = DecisionOutput(
            recommendations=recommendations,
            session_id=context.session_id,
            lap_number=context.lap_number,
            computation_time_ms=computation_time_ms,
            metadata={
                'driver_number': context.driver_number,
                'async': True,
            },
        )
        
        return output
    
    def make_decisions_batch(self, inputs: List[DecisionInput]) -> List[DecisionOutput]:
        """
        Make decisions for multiple drivers in parallel.
        
        Args:
            inputs: List of decision inputs
            
        Returns:
            List of decision outputs
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(executor.map(self.make_decision, inputs))
        
        return outputs
    
    def _resolve_conflicts(self, recommendations: List[DecisionRecommendation]) -> List[DecisionRecommendation]:
        """
        Resolve conflicts between recommendations.
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Filtered recommendations with conflicts resolved
        """
        # Group by category
        by_category: Dict[str, List[DecisionRecommendation]] = {}
        for rec in recommendations:
            category = rec.category.value
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(rec)
        
        # For each category, keep highest priority * confidence
        resolved = []
        for category, recs in by_category.items():
            if len(recs) == 1:
                resolved.append(recs[0])
            else:
                # Weighted voting: priority × confidence_score
                recs_with_scores = [
                    (rec, rec.priority * rec.confidence_score)
                    for rec in recs
                ]
                recs_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                best_rec = recs_with_scores[0][0]
                resolved.append(best_rec)
                
                logger.debug(
                    f"Conflict in {category}: chose {best_rec.action.value} "
                    f"(score={recs_with_scores[0][1]:.2f})"
                )
        
        return resolved
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine performance statistics.
        
        Returns:
            Stats dict
        """
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0.0
        )
        
        module_stats = [m.get_stats() for m in self.modules]
        
        return {
            'decision_count': self.decision_count,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'module_count': len(self.modules),
            'module_stats': module_stats,
        }
