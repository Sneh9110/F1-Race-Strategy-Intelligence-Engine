"""Decision Engine for F1 Race Strategy Intelligence."""

from decision_engine.schemas import (
    DecisionAction,
    ConfidenceLevel,
    TrafficLight,
    DecisionCategory,
    DecisionContext,
    SimulationContext,
    RivalContext,
    DecisionInput,
    AlternativeOption,
    DecisionReasoning,
    DecisionRecommendation,
    DecisionOutput,
)

from decision_engine.base import BaseDecisionModule

from decision_engine.pit_timing import PitTimingDecision
from decision_engine.strategy_conversion import StrategyConversionDecision
from decision_engine.offset_strategy import OffsetStrategyDecision
from decision_engine.safety_car_decision import SafetyCarDecision
from decision_engine.rain_strategy import RainStrategyDecision
from decision_engine.pace_monitoring import PaceMonitoringDecision
from decision_engine.undercut_overcut import UndercutOvercutDecision

from decision_engine.engine import DecisionEngine
from decision_engine.scoring import (
    ConfidenceScorer,
    PriorityCalculator,
    DecisionRanker,
)
from decision_engine.explainer import (
    DecisionExplainer,
    DecisionLogger,
    DecisionAuditor,
)
from decision_engine.registry import DecisionModuleRegistry, register_decision_module

__all__ = [
    # Schemas
    "DecisionAction",
    "ConfidenceLevel",
    "TrafficLight",
    "DecisionCategory",
    "DecisionContext",
    "SimulationContext",
    "RivalContext",
    "DecisionInput",
    "AlternativeOption",
    "DecisionReasoning",
    "DecisionRecommendation",
    "DecisionOutput",
    # Base
    "BaseDecisionModule",
    # Decision Modules
    "PitTimingDecision",
    "StrategyConversionDecision",
    "OffsetStrategyDecision",
    "SafetyCarDecision",
    "RainStrategyDecision",
    "PaceMonitoringDecision",
    "UndercutOvercutDecision",
    # Engine
    "DecisionEngine",
    # Scoring
    "ConfidenceScorer",
    "PriorityCalculator",
    "DecisionRanker",
    # Explainability
    "DecisionExplainer",
    "DecisionLogger",
    "DecisionAuditor",
    # Registry
    "DecisionModuleRegistry",
    "register_decision_module",
]

__version__ = "1.0.0"
