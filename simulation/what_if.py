"""
What-if scenario analysis engine.

Allows injection of hypothetical scenarios to evaluate strategic decisions.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any, Callable
import logging
from dataclasses import dataclass
from enum import Enum

from .schemas import (
    SimulationInput,
    SimulationOutput,
    StrategyOption,
    TireCompound,
)
from .core import RaceSimulator
from .race_state import RaceState

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    """What-if scenario types."""
    EARLY_SAFETY_CAR = "early_safety_car"
    LATE_SAFETY_CAR = "late_safety_car"
    DOUBLE_SAFETY_CAR = "double_safety_car"
    UNDERCUT_ATTEMPT = "undercut_attempt"
    OVERCUT_ATTEMPT = "overcut_attempt"
    TIRE_OFFSET = "tire_offset"
    AGGRESSIVE_PACING = "aggressive_pacing"
    CONSERVATIVE_PACING = "conservative_pacing"
    RAIN_TRANSITION = "rain_transition"
    TIRE_FAILURE = "tire_failure"


@dataclass
class WhatIfScenario:
    """What-if scenario definition."""
    scenario_type: ScenarioType
    description: str
    parameters: Dict[str, Any]
    injection_lap: Optional[int] = None
    
    def apply(self, sim_input: SimulationInput, race_state: RaceState) -> None:
        """Apply scenario modifications."""
        pass


@dataclass
class ScenarioComparison:
    """Comparison between baseline and scenario."""
    scenario: WhatIfScenario
    baseline_result: SimulationOutput
    scenario_result: SimulationOutput
    position_delta: int
    time_delta: float
    win_probability_delta: float


class WhatIfEngine:
    """What-if scenario analysis engine."""
    
    def __init__(self, simulator: RaceSimulator):
        """
        Initialize what-if engine.
        
        Args:
            simulator: Base race simulator
        """
        self.simulator = simulator
        self.scenario_templates = self._initialize_scenario_templates()
        
        logger.info("WhatIfEngine initialized with 10 scenario templates")
    
    def _initialize_scenario_templates(self) -> Dict[ScenarioType, WhatIfScenario]:
        """Initialize predefined scenario templates."""
        templates = {
            ScenarioType.EARLY_SAFETY_CAR: WhatIfScenario(
                scenario_type=ScenarioType.EARLY_SAFETY_CAR,
                description="Safety car deployed in first third of race",
                parameters={"deployment_lap_range": (10, 20), "duration": 3},
            ),
            
            ScenarioType.LATE_SAFETY_CAR: WhatIfScenario(
                scenario_type=ScenarioType.LATE_SAFETY_CAR,
                description="Safety car deployed in final third of race",
                parameters={"deployment_lap_range": (50, 60), "duration": 3},
            ),
            
            ScenarioType.DOUBLE_SAFETY_CAR: WhatIfScenario(
                scenario_type=ScenarioType.DOUBLE_SAFETY_CAR,
                description="Two safety car periods",
                parameters={"first_lap": 15, "second_lap": 45, "duration": 3},
            ),
            
            ScenarioType.UNDERCUT_ATTEMPT: WhatIfScenario(
                scenario_type=ScenarioType.UNDERCUT_ATTEMPT,
                description="Pit 2-3 laps earlier than planned",
                parameters={"lap_offset": -2},
            ),
            
            ScenarioType.OVERCUT_ATTEMPT: WhatIfScenario(
                scenario_type=ScenarioType.OVERCUT_ATTEMPT,
                description="Pit 2-3 laps later than planned",
                parameters={"lap_offset": 3},
            ),
            
            ScenarioType.TIRE_OFFSET: WhatIfScenario(
                scenario_type=ScenarioType.TIRE_OFFSET,
                description="Start on different compound than rivals",
                parameters={"offset_compound": TireCompound.HARD},
            ),
            
            ScenarioType.AGGRESSIVE_PACING: WhatIfScenario(
                scenario_type=ScenarioType.AGGRESSIVE_PACING,
                description="Push 0.2s faster per lap",
                parameters={"pace_delta": -0.2},
            ),
            
            ScenarioType.CONSERVATIVE_PACING: WhatIfScenario(
                scenario_type=ScenarioType.CONSERVATIVE_PACING,
                description="Conserve tires, 0.3s slower per lap",
                parameters={"pace_delta": 0.3},
            ),
            
            ScenarioType.RAIN_TRANSITION: WhatIfScenario(
                scenario_type=ScenarioType.RAIN_TRANSITION,
                description="Rain starts mid-race",
                parameters={"rain_start_lap": 35, "intensity": "moderate"},
            ),
            
            ScenarioType.TIRE_FAILURE: WhatIfScenario(
                scenario_type=ScenarioType.TIRE_FAILURE,
                description="Unexpected tire failure requiring extra stop",
                parameters={"failure_lap": 40},
            ),
        }
        
        return templates
    
    def analyze_scenario(
        self,
        base_input: SimulationInput,
        scenario: WhatIfScenario,
        target_driver: int = 1,
    ) -> ScenarioComparison:
        """
        Analyze specific what-if scenario.
        
        Args:
            base_input: Baseline simulation input
            scenario: Scenario to analyze
            target_driver: Driver to focus analysis on
        
        Returns:
            Comparison between baseline and scenario
        """
        logger.info(f"Analyzing scenario: {scenario.scenario_type.value}")
        
        # Run baseline
        baseline_output = self.simulator.simulate_race(base_input)
        
        # Apply scenario modifications
        scenario_input = self._apply_scenario(base_input, scenario)
        
        # Run scenario
        scenario_output = self.simulator.simulate_race(scenario_input)
        
        # Extract target driver results
        baseline_result = next(
            (r for r in baseline_output.results if r.driver_number == target_driver),
            baseline_output.results[0],
        )
        
        scenario_result = next(
            (r for r in scenario_output.results if r.driver_number == target_driver),
            scenario_output.results[0],
        )
        
        # Calculate deltas
        position_delta = scenario_result.final_position - baseline_result.final_position
        time_delta = scenario_result.total_race_time - baseline_result.total_race_time
        win_prob_delta = scenario_result.win_probability - baseline_result.win_probability
        
        comparison = ScenarioComparison(
            scenario=scenario,
            baseline_result=baseline_output,
            scenario_result=scenario_output,
            position_delta=position_delta,
            time_delta=time_delta,
            win_probability_delta=win_prob_delta,
        )
        
        logger.info(
            f"Scenario analysis complete: Δposition={position_delta:+d}, "
            f"Δtime={time_delta:+.2f}s, Δwin_prob={win_prob_delta:+.3f}"
        )
        
        return comparison
    
    def _apply_scenario(
        self,
        base_input: SimulationInput,
        scenario: WhatIfScenario,
    ) -> SimulationInput:
        """Apply scenario modifications to input."""
        # Clone input
        scenario_input = base_input.copy(deep=True)
        
        # Apply scenario-specific modifications
        if scenario.scenario_type == ScenarioType.EARLY_SAFETY_CAR:
            # Modify race config to inject SC
            lap_range = scenario.parameters["deployment_lap_range"]
            scenario_input.what_if_params = {
                "inject_safety_car": True,
                "sc_lap": lap_range[0],
                "sc_duration": scenario.parameters["duration"],
            }
        
        elif scenario.scenario_type == ScenarioType.LATE_SAFETY_CAR:
            lap_range = scenario.parameters["deployment_lap_range"]
            scenario_input.what_if_params = {
                "inject_safety_car": True,
                "sc_lap": lap_range[1],
                "sc_duration": scenario.parameters["duration"],
            }
        
        elif scenario.scenario_type == ScenarioType.DOUBLE_SAFETY_CAR:
            scenario_input.what_if_params = {
                "inject_safety_car": True,
                "sc_laps": [
                    scenario.parameters["first_lap"],
                    scenario.parameters["second_lap"],
                ],
                "sc_duration": scenario.parameters["duration"],
            }
        
        elif scenario.scenario_type == ScenarioType.UNDERCUT_ATTEMPT:
            offset = scenario.parameters["lap_offset"]
            scenario_input.strategy_to_evaluate.pit_laps = [
                max(6, lap + offset)
                for lap in scenario_input.strategy_to_evaluate.pit_laps
            ]
        
        elif scenario.scenario_type == ScenarioType.OVERCUT_ATTEMPT:
            offset = scenario.parameters["lap_offset"]
            scenario_input.strategy_to_evaluate.pit_laps = [
                lap + offset
                for lap in scenario_input.strategy_to_evaluate.pit_laps
            ]
        
        elif scenario.scenario_type == ScenarioType.TIRE_OFFSET:
            new_compound = scenario.parameters["offset_compound"]
            scenario_input.strategy_to_evaluate.tire_sequence[0] = new_compound
        
        elif scenario.scenario_type == ScenarioType.AGGRESSIVE_PACING:
            pace_delta = scenario.parameters["pace_delta"]
            scenario_input.what_if_params = {
                "pace_adjustment": pace_delta,
            }
        
        elif scenario.scenario_type == ScenarioType.CONSERVATIVE_PACING:
            pace_delta = scenario.parameters["pace_delta"]
            scenario_input.what_if_params = {
                "pace_adjustment": pace_delta,
            }
        
        elif scenario.scenario_type == ScenarioType.RAIN_TRANSITION:
            scenario_input.what_if_params = {
                "rain_start_lap": scenario.parameters["rain_start_lap"],
                "rain_intensity": scenario.parameters["intensity"],
            }
        
        elif scenario.scenario_type == ScenarioType.TIRE_FAILURE:
            failure_lap = scenario.parameters["failure_lap"]
            # Add extra unplanned pit stop
            scenario_input.strategy_to_evaluate.pit_laps.append(failure_lap)
            scenario_input.strategy_to_evaluate.pit_laps.sort()
            scenario_input.strategy_to_evaluate.tire_sequence.append(
                scenario_input.strategy_to_evaluate.tire_sequence[-1]
            )
        
        return scenario_input
    
    def compare_strategies_under_scenarios(
        self,
        base_input: SimulationInput,
        strategies: List[StrategyOption],
        scenarios: List[WhatIfScenario],
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies across multiple scenarios.
        
        Args:
            base_input: Base simulation input
            strategies: List of strategies to compare
            scenarios: List of scenarios to test
        
        Returns:
            Matrix of results for each strategy-scenario combination
        """
        logger.info(f"Comparing {len(strategies)} strategies across {len(scenarios)} scenarios")
        
        results_matrix = []
        
        for strategy_idx, strategy in enumerate(strategies):
            strategy_results = []
            
            for scenario in scenarios:
                # Create input with this strategy
                test_input = base_input.copy(deep=True)
                test_input.strategy_to_evaluate = strategy
                
                # Analyze scenario
                comparison = self.analyze_scenario(test_input, scenario)
                
                strategy_results.append({
                    "scenario": scenario.scenario_type.value,
                    "position": comparison.scenario_result.results[0].final_position,
                    "time": comparison.scenario_result.results[0].total_race_time,
                    "position_delta": comparison.position_delta,
                    "time_delta": comparison.time_delta,
                })
            
            results_matrix.append({
                "strategy_index": strategy_idx,
                "strategy": strategy,
                "scenario_results": strategy_results,
            })
        
        # Find most robust strategy (best average performance across scenarios)
        avg_positions = [
            sum(r["position"] for r in strat["scenario_results"]) / len(scenarios)
            for strat in results_matrix
        ]
        
        best_strategy_idx = avg_positions.index(min(avg_positions))
        
        return {
            "strategies": strategies,
            "scenarios": [s.scenario_type.value for s in scenarios],
            "results_matrix": results_matrix,
            "most_robust_strategy_index": best_strategy_idx,
            "average_positions": avg_positions,
        }
    
    def generate_scenario_recommendations(
        self,
        base_input: SimulationInput,
        target_driver: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on scenario analysis.
        
        Args:
            base_input: Base simulation input
            target_driver: Driver to analyze
        
        Returns:
            List of recommendations with risk/opportunity assessments
        """
        recommendations = []
        
        # Test key scenarios
        key_scenarios = [
            self.scenario_templates[ScenarioType.EARLY_SAFETY_CAR],
            self.scenario_templates[ScenarioType.LATE_SAFETY_CAR],
            self.scenario_templates[ScenarioType.UNDERCUT_ATTEMPT],
        ]
        
        for scenario in key_scenarios:
            comparison = self.analyze_scenario(base_input, scenario, target_driver)
            
            # Generate recommendation
            if comparison.position_delta < 0:
                recommendation = {
                    "scenario": scenario.scenario_type.value,
                    "impact": "positive",
                    "position_gain": abs(comparison.position_delta),
                    "recommendation": f"Scenario improves position by {abs(comparison.position_delta)} places",
                    "confidence": "high" if abs(comparison.time_delta) > 3.0 else "moderate",
                }
            elif comparison.position_delta > 0:
                recommendation = {
                    "scenario": scenario.scenario_type.value,
                    "impact": "negative",
                    "position_loss": comparison.position_delta,
                    "recommendation": f"Scenario worsens position by {comparison.position_delta} places",
                    "confidence": "high" if abs(comparison.time_delta) > 3.0 else "moderate",
                }
            else:
                recommendation = {
                    "scenario": scenario.scenario_type.value,
                    "impact": "neutral",
                    "recommendation": "Scenario has minimal impact on final position",
                    "confidence": "moderate",
                }
            
            recommendations.append(recommendation)
        
        return recommendations
