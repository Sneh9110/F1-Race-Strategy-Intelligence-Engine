"""
Race Simulation Engine.

Production-grade Monte Carlo race simulator integrating all ML models.
"""

from .core import RaceSimulator
from .strategy_tree import StrategyTreeExplorer, StrategyNode
from .monte_carlo import MonteCarloSimulator, MonteCarloConfig, MonteCarloResult
from .what_if import WhatIfEngine, WhatIfScenario, ScenarioType, ScenarioComparison
from .race_state import RaceState
from .schemas import (
    SimulationInput,
    SimulationOutput,
    RaceConfig,
    DriverState,
    StrategyOption,
    DriverSimulationResult,
    StrategyRanking,
    LapResult,
    StintResult,
    PitStopInfo,
    TireCompound,
    PaceTarget,
    TrafficState,
)

__all__ = [
    # Core simulator
    "RaceSimulator",
    "RaceState",
    
    # Advanced components
    "StrategyTreeExplorer",
    "StrategyNode",
    "MonteCarloSimulator",
    "MonteCarloConfig",
    "MonteCarloResult",
    "WhatIfEngine",
    "WhatIfScenario",
    "ScenarioType",
    "ScenarioComparison",
    
    # Schemas - Input
    "SimulationInput",
    "RaceConfig",
    "DriverState",
    "StrategyOption",
    
    # Schemas - Output
    "SimulationOutput",
    "DriverSimulationResult",
    "StrategyRanking",
    "LapResult",
    "StintResult",
    "PitStopInfo",
    
    # Enums
    "TireCompound",
    "PaceTarget",
    "TrafficState",
]

__version__ = "1.0.0"
