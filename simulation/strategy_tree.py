"""
Strategy tree exploration for optimal race strategy discovery.

Uses tree search with pruning to evaluate multiple pit stop strategies.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time

from .schemas import (
    SimulationInput,
    SimulationOutput,
    StrategyOption,
    StrategyRanking,
    TireCompound,
    PaceTarget,
)
from .core import RaceSimulator

logger = logging.getLogger(__name__)


@dataclass
class StrategyNode:
    """Node in strategy tree."""
    pit_laps: List[int]
    tire_sequence: List[TireCompound]
    estimated_time: Optional[float] = None
    simulated: bool = False
    parent: Optional[StrategyNode] = None
    children: List[StrategyNode] = None
    depth: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def to_strategy_option(self, target_pace: PaceTarget = PaceTarget.BALANCED) -> StrategyOption:
        """Convert node to strategy option."""
        return StrategyOption(
            pit_laps=self.pit_laps,
            tire_sequence=self.tire_sequence,
            target_pace=target_pace,
        )


class StrategyTreeExplorer:
    """Explores strategy space using tree search."""
    
    def __init__(
        self,
        simulator: RaceSimulator,
        max_workers: int = 4,
        max_strategies: int = 50,
        pruning_threshold: float = 5.0,
    ):
        """
        Initialize strategy explorer.
        
        Args:
            simulator: Race simulator instance
            max_workers: Max parallel workers
            max_strategies: Max strategies to evaluate
            pruning_threshold: Prune branches worse than best + threshold (seconds)
        """
        self.simulator = simulator
        self.max_workers = max_workers
        self.max_strategies = max_strategies
        self.pruning_threshold = pruning_threshold
        
        logger.info(f"StrategyTreeExplorer initialized: {max_workers} workers, {max_strategies} max strategies")
    
    def explore_strategies(
        self,
        base_input: SimulationInput,
        available_compounds: List[TireCompound] = None,
        max_pit_stops: int = 3,
    ) -> List[StrategyRanking]:
        """
        Explore strategy tree and rank strategies.
        
        Args:
            base_input: Base simulation input (without strategy)
            available_compounds: Available tire compounds
            max_pit_stops: Maximum pit stops to consider
        
        Returns:
            Ranked list of strategies with performance metrics
        """
        start_time = time.time()
        
        if available_compounds is None:
            available_compounds = [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]
        
        logger.info(
            f"Exploring strategies: {len(available_compounds)} compounds, "
            f"max {max_pit_stops} stops, {self.max_strategies} max strategies"
        )
        
        # Generate initial strategy nodes
        strategy_nodes = self._generate_strategy_nodes(
            total_laps=base_input.race_config.total_laps,
            available_compounds=available_compounds,
            max_pit_stops=max_pit_stops,
        )
        
        logger.info(f"Generated {len(strategy_nodes)} initial strategies")
        
        # Limit to max strategies
        if len(strategy_nodes) > self.max_strategies:
            strategy_nodes = strategy_nodes[:self.max_strategies]
            logger.info(f"Pruned to {self.max_strategies} strategies")
        
        # Evaluate strategies in parallel
        rankings = self._evaluate_strategies_parallel(base_input, strategy_nodes)
        
        # Sort by expected race time
        rankings.sort(key=lambda r: r.expected_race_time)
        
        elapsed = time.time() - start_time
        logger.info(f"Strategy exploration complete in {elapsed:.2f}s: {len(rankings)} strategies evaluated")
        
        return rankings
    
    def _generate_strategy_nodes(
        self,
        total_laps: int,
        available_compounds: List[TireCompound],
        max_pit_stops: int,
    ) -> List[StrategyNode]:
        """Generate all viable strategy nodes."""
        nodes = []
        
        # Generate strategies for 1 to max_pit_stops
        for num_stops in range(1, max_pit_stops + 1):
            nodes.extend(
                self._generate_n_stop_strategies(
                    total_laps=total_laps,
                    num_stops=num_stops,
                    available_compounds=available_compounds,
                )
            )
        
        return nodes
    
    def _generate_n_stop_strategies(
        self,
        total_laps: int,
        num_stops: int,
        available_compounds: List[TireCompound],
    ) -> List[StrategyNode]:
        """Generate all n-stop strategies."""
        nodes = []
        
        # Calculate stint lengths
        avg_stint = total_laps // (num_stops + 1)
        
        # Generate pit lap combinations
        pit_lap_combinations = self._generate_pit_lap_combinations(
            total_laps=total_laps,
            num_stops=num_stops,
            avg_stint=avg_stint,
        )
        
        # Generate tire sequences
        tire_sequences = self._generate_tire_sequences(
            num_stints=num_stops + 1,
            available_compounds=available_compounds,
        )
        
        # Combine pit laps and tire sequences
        for pit_laps in pit_lap_combinations:
            for tire_seq in tire_sequences:
                node = StrategyNode(
                    pit_laps=pit_laps,
                    tire_sequence=tire_seq,
                    depth=num_stops,
                )
                nodes.append(node)
        
        return nodes
    
    def _generate_pit_lap_combinations(
        self,
        total_laps: int,
        num_stops: int,
        avg_stint: int,
    ) -> List[List[int]]:
        """Generate pit lap combinations."""
        combinations = []
        
        # Define reasonable pit windows (avoid first/last 5 laps)
        min_pit_lap = 6
        max_pit_lap = total_laps - 5
        
        if num_stops == 1:
            # 1-stop: pit around 1/3 to 2/3 race distance
            early = total_laps // 3
            late = 2 * total_laps // 3
            for lap in range(early, late + 1, 3):
                if min_pit_lap <= lap <= max_pit_lap:
                    combinations.append([lap])
        
        elif num_stops == 2:
            # 2-stop: divide into roughly equal thirds
            first_window = range(total_laps // 4, total_laps // 2, 3)
            second_window = range(total_laps // 2 + 5, 3 * total_laps // 4, 3)
            
            for lap1 in first_window:
                for lap2 in second_window:
                    if min_pit_lap <= lap1 <= max_pit_lap and min_pit_lap <= lap2 <= max_pit_lap:
                        if lap2 - lap1 >= 10:  # Minimum 10 laps between stops
                            combinations.append([lap1, lap2])
        
        elif num_stops == 3:
            # 3-stop: divide into quarters
            combinations.append([total_laps // 5, 2 * total_laps // 5, 3 * total_laps // 5])
            combinations.append([total_laps // 4, total_laps // 2, 3 * total_laps // 4])
        
        return combinations
    
    def _generate_tire_sequences(
        self,
        num_stints: int,
        available_compounds: List[TireCompound],
    ) -> List[List[TireCompound]]:
        """Generate tire compound sequences."""
        sequences = []
        
        # Common strategies
        if TireCompound.SOFT in available_compounds and TireCompound.MEDIUM in available_compounds:
            if num_stints == 2:
                sequences.extend([
                    [TireCompound.SOFT, TireCompound.MEDIUM],
                    [TireCompound.MEDIUM, TireCompound.SOFT],
                    [TireCompound.MEDIUM, TireCompound.MEDIUM],
                ])
            
            if TireCompound.HARD in available_compounds:
                if num_stints == 2:
                    sequences.extend([
                        [TireCompound.SOFT, TireCompound.HARD],
                        [TireCompound.MEDIUM, TireCompound.HARD],
                    ])
                
                elif num_stints == 3:
                    sequences.extend([
                        [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD],
                        [TireCompound.MEDIUM, TireCompound.MEDIUM, TireCompound.HARD],
                        [TireCompound.SOFT, TireCompound.SOFT, TireCompound.MEDIUM],
                        [TireCompound.MEDIUM, TireCompound.HARD, TireCompound.MEDIUM],
                    ])
                
                elif num_stints == 4:
                    sequences.extend([
                        [TireCompound.SOFT, TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD],
                        [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.MEDIUM, TireCompound.HARD],
                    ])
        
        # Fallback: all same compound
        if not sequences:
            for compound in available_compounds:
                sequences.append([compound] * num_stints)
        
        return sequences
    
    def _evaluate_strategies_parallel(
        self,
        base_input: SimulationInput,
        strategy_nodes: List[StrategyNode],
    ) -> List[StrategyRanking]:
        """Evaluate strategies in parallel."""
        rankings = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all strategies
            future_to_node = {
                executor.submit(self._evaluate_strategy, base_input, node): node
                for node in strategy_nodes
            }
            
            # Collect results
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                try:
                    ranking = future.result()
                    if ranking:
                        rankings.append(ranking)
                except Exception as e:
                    logger.warning(f"Strategy evaluation failed: {e}")
        
        return rankings
    
    def _evaluate_strategy(
        self,
        base_input: SimulationInput,
        node: StrategyNode,
    ) -> Optional[StrategyRanking]:
        """Evaluate single strategy."""
        try:
            # Create strategy option
            strategy = node.to_strategy_option()
            
            # Create simulation input with this strategy
            sim_input = SimulationInput(
                race_config=base_input.race_config,
                drivers=base_input.drivers,
                strategy_to_evaluate=strategy,
                monte_carlo_runs=0,  # Single deterministic run
            )
            
            # Simulate
            output = self.simulator.simulate_race(sim_input)
            
            # Find primary driver result (assume driver_number=1)
            primary_result = next(
                (r for r in output.results if r.driver_number == 1),
                output.results[0] if output.results else None,
            )
            
            if not primary_result:
                return None
            
            # Calculate metrics
            expected_time = primary_result.total_race_time
            win_prob = primary_result.win_probability
            podium_prob = primary_result.podium_probability
            final_pos = primary_result.final_position
            
            # Estimate risk (higher pit stops = higher risk)
            risk_score = len(strategy.pit_laps) * 0.1
            
            ranking = StrategyRanking(
                strategy=strategy,
                expected_race_time=expected_time,
                win_probability=win_prob,
                podium_probability=podium_prob,
                expected_position=float(final_pos),
                risk_score=risk_score,
                undercut_opportunity=self._estimate_undercut_opportunity(strategy),
                overcut_opportunity=self._estimate_overcut_opportunity(strategy),
            )
            
            return ranking
            
        except Exception as e:
            logger.warning(f"Strategy evaluation error: {e}")
            return None
    
    def _estimate_undercut_opportunity(self, strategy: StrategyOption) -> float:
        """Estimate undercut opportunity (0-1)."""
        if not strategy.pit_laps:
            return 0.0
        
        # Earlier first pit = higher undercut opportunity
        first_pit = strategy.pit_laps[0]
        if first_pit < 20:
            return 0.8
        elif first_pit < 30:
            return 0.5
        else:
            return 0.2
    
    def _estimate_overcut_opportunity(self, strategy: StrategyOption) -> float:
        """Estimate overcut opportunity (0-1)."""
        if not strategy.pit_laps:
            return 0.0
        
        # Later first pit = higher overcut opportunity
        first_pit = strategy.pit_laps[0]
        if first_pit > 35:
            return 0.8
        elif first_pit > 25:
            return 0.5
        else:
            return 0.2
    
    def analyze_undercut_window(
        self,
        base_input: SimulationInput,
        target_strategy: StrategyOption,
        lap_range: Tuple[int, int] = (15, 35),
    ) -> Dict[int, float]:
        """
        Analyze undercut effectiveness across lap range.
        
        Args:
            base_input: Base simulation input
            target_strategy: Target strategy to undercut
            lap_range: Range of laps to analyze
        
        Returns:
            Dict mapping lap number to time gained
        """
        results = {}
        
        for lap in range(lap_range[0], lap_range[1] + 1):
            # Create undercut strategy (pit 1 lap earlier)
            undercut_strategy = StrategyOption(
                pit_laps=[lap] + target_strategy.pit_laps[1:],
                tire_sequence=target_strategy.tire_sequence,
                target_pace=PaceTarget.AGGRESSIVE,
            )
            
            # Simulate both strategies
            try:
                target_result = self._evaluate_strategy(base_input, StrategyNode(
                    pit_laps=target_strategy.pit_laps,
                    tire_sequence=target_strategy.tire_sequence,
                ))
                
                undercut_result = self._evaluate_strategy(base_input, StrategyNode(
                    pit_laps=undercut_strategy.pit_laps,
                    tire_sequence=undercut_strategy.tire_sequence,
                ))
                
                if target_result and undercut_result:
                    time_gained = target_result.expected_race_time - undercut_result.expected_race_time
                    results[lap] = time_gained
            
            except Exception as e:
                logger.warning(f"Undercut analysis failed for lap {lap}: {e}")
        
        return results
    
    def analyze_overcut_window(
        self,
        base_input: SimulationInput,
        target_strategy: StrategyOption,
        lap_range: Tuple[int, int] = (20, 40),
    ) -> Dict[int, float]:
        """
        Analyze overcut effectiveness across lap range.
        
        Args:
            base_input: Base simulation input
            target_strategy: Target strategy to overcut
            lap_range: Range of laps to analyze
        
        Returns:
            Dict mapping lap number to time gained
        """
        results = {}
        
        for lap in range(lap_range[0], lap_range[1] + 1):
            # Create overcut strategy (pit 1 lap later)
            overcut_strategy = StrategyOption(
                pit_laps=[lap] + target_strategy.pit_laps[1:],
                tire_sequence=target_strategy.tire_sequence,
                target_pace=PaceTarget.CONSERVATIVE,
            )
            
            # Simulate both strategies
            try:
                target_result = self._evaluate_strategy(base_input, StrategyNode(
                    pit_laps=target_strategy.pit_laps,
                    tire_sequence=target_strategy.tire_sequence,
                ))
                
                overcut_result = self._evaluate_strategy(base_input, StrategyNode(
                    pit_laps=overcut_strategy.pit_laps,
                    tire_sequence=overcut_strategy.tire_sequence,
                ))
                
                if target_result and overcut_result:
                    time_gained = target_result.expected_race_time - overcut_result.expected_race_time
                    results[lap] = time_gained
            
            except Exception as e:
                logger.warning(f"Overcut analysis failed for lap {lap}: {e}")
        
        return results
