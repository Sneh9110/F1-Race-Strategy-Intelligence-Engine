"""
Monte Carlo simulation for probabilistic race outcome analysis.

Uses multiprocessing for parallel execution of MC runs with randomness injection.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any
import logging
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass

from .schemas import (
    SimulationInput,
    SimulationOutput,
    DriverSimulationResult,
)
from .core import RaceSimulator
from .race_state import RaceState

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation configuration."""
    num_runs: int = 1000
    num_workers: int = None
    convergence_threshold: float = 0.01
    convergence_window: int = 100
    
    # Noise injection parameters
    tire_deg_noise_std: float = 0.02
    lap_time_noise_std: float = 0.15
    pit_stop_noise_std: float = 0.5
    sc_probability_boost: float = 0.05
    
    def __post_init__(self):
        if self.num_workers is None:
            self.num_workers = min(cpu_count(), 8)


@dataclass
class MonteCarloResult:
    """Single Monte Carlo run result."""
    run_id: int
    final_position: int
    race_time: float
    num_pit_stops: int
    sc_deployments: int
    overtakes: int


class MonteCarloSimulator:
    """Monte Carlo simulation engine."""
    
    def __init__(
        self,
        simulator: RaceSimulator,
        config: Optional[MonteCarloConfig] = None,
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            simulator: Base race simulator
            config: MC configuration
        """
        self.simulator = simulator
        self.config = config or MonteCarloConfig()
        
        logger.info(
            f"MonteCarloSimulator initialized: {self.config.num_runs} runs, "
            f"{self.config.num_workers} workers"
        )
    
    def run_monte_carlo(
        self,
        base_input: SimulationInput,
        target_driver: int = 1,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        
        Args:
            base_input: Base simulation input
            target_driver: Driver number to analyze
        
        Returns:
            Dictionary with MC statistics
        """
        start_time = time.time()
        
        logger.info(f"Starting Monte Carlo: {self.config.num_runs} runs for driver {target_driver}")
        
        # Run simulations in parallel
        mc_results = self._run_parallel_simulations(base_input, target_driver)
        
        # Calculate statistics
        statistics = self._calculate_statistics(mc_results, target_driver)
        
        # Check convergence
        convergence_info = self._check_convergence(mc_results)
        
        elapsed = time.time() - start_time
        logger.info(f"Monte Carlo complete in {elapsed:.2f}s: {len(mc_results)} runs")
        
        return {
            "num_runs": len(mc_results),
            "target_driver": target_driver,
            "statistics": statistics,
            "convergence": convergence_info,
            "elapsed_time_seconds": elapsed,
        }
    
    def _run_parallel_simulations(
        self,
        base_input: SimulationInput,
        target_driver: int,
    ) -> List[MonteCarloResult]:
        """Run simulations in parallel."""
        results = []
        
        # Create argument tuples for each run
        args_list = [
            (base_input, target_driver, run_id, self.config)
            for run_id in range(self.config.num_runs)
        ]
        
        # Run in parallel
        with Pool(processes=self.config.num_workers) as pool:
            results = pool.starmap(_run_single_mc, args_list)
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        return results
    
    def _calculate_statistics(
        self,
        mc_results: List[MonteCarloResult],
        target_driver: int,
    ) -> Dict[str, Any]:
        """Calculate statistics from MC results."""
        if not mc_results:
            return {}
        
        # Extract metrics
        positions = [r.final_position for r in mc_results]
        race_times = [r.race_time for r in mc_results]
        sc_counts = [r.sc_deployments for r in mc_results]
        
        # Position statistics
        position_stats = {
            "mean": float(np.mean(positions)),
            "median": float(np.median(positions)),
            "std": float(np.std(positions)),
            "p10": float(np.percentile(positions, 10)),
            "p90": float(np.percentile(positions, 90)),
        }
        
        # Position distribution
        position_distribution = {}
        for pos in range(1, 21):
            count = sum(1 for r in mc_results if r.final_position == pos)
            position_distribution[pos] = count / len(mc_results)
        
        # Probability metrics
        win_probability = sum(1 for r in mc_results if r.final_position == 1) / len(mc_results)
        podium_probability = sum(1 for r in mc_results if r.final_position <= 3) / len(mc_results)
        points_probability = sum(1 for r in mc_results if r.final_position <= 10) / len(mc_results)
        
        # Race time statistics
        race_time_stats = {
            "mean": float(np.mean(race_times)),
            "median": float(np.median(race_times)),
            "std": float(np.std(race_times)),
            "min": float(np.min(race_times)),
            "max": float(np.max(race_times)),
        }
        
        # Safety car statistics
        sc_stats = {
            "mean_deployments": float(np.mean(sc_counts)),
            "probability_sc": sum(1 for r in mc_results if r.sc_deployments > 0) / len(mc_results),
        }
        
        return {
            "position": position_stats,
            "position_distribution": position_distribution,
            "win_probability": win_probability,
            "podium_probability": podium_probability,
            "points_probability": points_probability,
            "race_time": race_time_stats,
            "safety_car": sc_stats,
        }
    
    def _check_convergence(
        self,
        mc_results: List[MonteCarloResult],
    ) -> Dict[str, Any]:
        """Check if MC simulation has converged."""
        if len(mc_results) < self.config.convergence_window * 2:
            return {"converged": False, "reason": "insufficient_runs"}
        
        # Calculate rolling win probability
        window = self.config.convergence_window
        positions = [r.final_position for r in mc_results]
        
        rolling_win_probs = []
        for i in range(window, len(positions)):
            recent_positions = positions[i - window:i]
            win_prob = sum(1 for p in recent_positions if p == 1) / window
            rolling_win_probs.append(win_prob)
        
        if len(rolling_win_probs) < 2:
            return {"converged": False, "reason": "insufficient_windows"}
        
        # Check if variance in recent windows is below threshold
        recent_variance = np.var(rolling_win_probs[-10:])
        converged = recent_variance < self.config.convergence_threshold
        
        return {
            "converged": converged,
            "variance": float(recent_variance),
            "threshold": self.config.convergence_threshold,
            "windows_analyzed": len(rolling_win_probs),
        }
    
    def analyze_sensitivity(
        self,
        base_input: SimulationInput,
        parameter: str,
        values: List[float],
    ) -> Dict[str, Any]:
        """
        Analyze sensitivity to parameter variations.
        
        Args:
            base_input: Base simulation input
            parameter: Parameter to vary (e.g., 'tire_deg_noise_std')
            values: List of parameter values to test
        
        Returns:
            Sensitivity analysis results
        """
        results = {}
        
        for value in values:
            # Update config
            original_value = getattr(self.config, parameter)
            setattr(self.config, parameter, value)
            
            # Run MC
            mc_result = self.run_monte_carlo(base_input)
            
            # Store results
            results[value] = {
                "win_probability": mc_result["statistics"]["win_probability"],
                "position_mean": mc_result["statistics"]["position"]["mean"],
            }
            
            # Restore original value
            setattr(self.config, parameter, original_value)
        
        return {
            "parameter": parameter,
            "values": values,
            "results": results,
        }


def _run_single_mc(
    base_input: SimulationInput,
    target_driver: int,
    run_id: int,
    config: MonteCarloConfig,
) -> Optional[MonteCarloResult]:
    """Run single Monte Carlo iteration (for multiprocessing)."""
    try:
        # Create simulator for this process
        simulator = RaceSimulator()
        
        # Inject randomness into input
        noisy_input = _inject_noise(base_input, config, run_id)
        
        # Simulate
        output = simulator.simulate_race(noisy_input)
        
        # Extract target driver result
        target_result = next(
            (r for r in output.results if r.driver_number == target_driver),
            None,
        )
        
        if not target_result:
            return None
        
        # Count SC deployments (from race state events)
        sc_deployments = 0  # Simplified
        
        # Count overtakes
        overtakes = 0  # Simplified
        
        return MonteCarloResult(
            run_id=run_id,
            final_position=target_result.final_position,
            race_time=target_result.total_race_time,
            num_pit_stops=len(target_result.pit_stops),
            sc_deployments=sc_deployments,
            overtakes=overtakes,
        )
    
    except Exception as e:
        logger.warning(f"MC run {run_id} failed: {e}")
        return None


def _inject_noise(
    base_input: SimulationInput,
    config: MonteCarloConfig,
    seed: int,
) -> SimulationInput:
    """Inject noise into simulation input."""
    np.random.seed(seed)
    
    # Clone input
    noisy_input = base_input.copy(deep=True)
    
    # Inject tire degradation noise (modify tire ages slightly)
    for driver in noisy_input.drivers:
        if driver.tire_age > 0:
            noise = np.random.normal(0, config.tire_deg_noise_std)
            driver.tire_age = max(0, int(driver.tire_age * (1 + noise)))
    
    # Inject lap time noise (via fuel load variation)
    for driver in noisy_input.drivers:
        noise = np.random.normal(0, config.lap_time_noise_std)
        driver.fuel_load = max(0, min(110, driver.fuel_load + noise))
    
    # Inject pit stop noise (handled in simulator)
    # Noise applied during pit stop execution
    
    # Boost SC probability
    if np.random.random() < config.sc_probability_boost:
        # Inject a simulated incident to increase SC chance
        pass
    
    return noisy_input


def estimate_required_runs(
    base_input: SimulationInput,
    target_confidence: float = 0.95,
    margin_of_error: float = 0.02,
) -> int:
    """
    Estimate required MC runs for target confidence.
    
    Args:
        base_input: Base simulation input
        target_confidence: Target confidence level (e.g., 0.95)
        margin_of_error: Acceptable margin of error (e.g., 0.02)
    
    Returns:
        Estimated required number of runs
    """
    # Use pilot run to estimate variance
    pilot_runs = 100
    
    simulator = RaceSimulator()
    config = MonteCarloConfig(num_runs=pilot_runs, num_workers=4)
    mc_sim = MonteCarloSimulator(simulator, config)
    
    pilot_result = mc_sim.run_monte_carlo(base_input)
    
    # Extract win probability std
    win_prob_std = pilot_result["statistics"]["position"]["std"] / 20  # Rough estimate
    
    # Calculate required runs using normal approximation
    # n = (z * σ / E)^2, where z is z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(target_confidence, 1.96)
    
    required = int((z * win_prob_std / margin_of_error) ** 2)
    
    # Cap at reasonable maximum
    required = min(required, 10000)
    required = max(required, 100)
    
    logger.info(f"Estimated required MC runs: {required} (confidence={target_confidence}, margin={margin_of_error})")
    
    return required
