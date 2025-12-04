"""Integration tests for complete simulation workflow."""

import pytest

from simulation import (
    RaceSimulator,
    StrategyTreeExplorer,
    MonteCarloSimulator,
    MonteCarloConfig,
    WhatIfEngine,
    ScenarioType,
)
from simulation.schemas import TireCompound


class TestFullSimulationWorkflow:
    """Test complete simulation workflow."""
    
    def test_end_to_end_simulation(self, sample_simulation_input):
        """Test end-to-end simulation workflow."""
        # Initialize
        simulator = RaceSimulator()
        
        # Run simulation
        output = simulator.simulate_race(sample_simulation_input)
        
        # Validate output
        assert output is not None
        assert len(output.results) == 20
        assert output.metadata["computation_time_ms"] < 1000
        
        # Check results are sorted
        for i in range(len(output.results) - 1):
            assert output.results[i].final_position <= output.results[i + 1].final_position


class TestStrategyWorkflow:
    """Test strategy exploration workflow."""
    
    def test_explore_and_select_strategy(self, sample_simulation_input):
        """Test exploring strategies and selecting best."""
        # Initialize
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator, max_strategies=20)
        
        # Explore strategies
        rankings = explorer.explore_strategies(
            sample_simulation_input,
            available_compounds=[TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD],
            max_pit_stops=2,
        )
        
        # Validate rankings
        assert len(rankings) > 0
        
        # Best strategy should have lowest expected time
        best_strategy = rankings[0]
        assert all(best_strategy.expected_race_time <= r.expected_race_time for r in rankings)
        
        # Simulate with best strategy
        sample_simulation_input.strategy_to_evaluate = best_strategy.strategy
        output = simulator.simulate_race(sample_simulation_input)
        
        assert output is not None


class TestMonteCarloWorkflow:
    """Test Monte Carlo workflow."""
    
    def test_monte_carlo_analysis(self, sample_simulation_input):
        """Test MC analysis workflow."""
        # Initialize
        simulator = RaceSimulator()
        config = MonteCarloConfig(num_runs=50, num_workers=2)
        mc_sim = MonteCarloSimulator(simulator, config)
        
        # Run MC
        result = mc_sim.run_monte_carlo(sample_simulation_input, target_driver=1)
        
        # Validate results
        assert result["num_runs"] > 0
        assert "statistics" in result
        assert 0.0 <= result["statistics"]["win_probability"] <= 1.0
        assert 0.0 <= result["statistics"]["podium_probability"] <= 1.0


class TestWhatIfWorkflow:
    """Test what-if analysis workflow."""
    
    def test_scenario_analysis_workflow(self, sample_simulation_input):
        """Test scenario analysis workflow."""
        # Initialize
        simulator = RaceSimulator()
        engine = WhatIfEngine(simulator)
        
        # Analyze scenarios
        scenarios_to_test = [
            ScenarioType.EARLY_SAFETY_CAR,
            ScenarioType.UNDERCUT_ATTEMPT,
        ]
        
        comparisons = []
        for scenario_type in scenarios_to_test:
            scenario = engine.scenario_templates[scenario_type]
            comparison = engine.analyze_scenario(sample_simulation_input, scenario)
            comparisons.append(comparison)
        
        # Validate comparisons
        assert len(comparisons) == 2
        
        for comp in comparisons:
            assert comp.baseline_result is not None
            assert comp.scenario_result is not None
            assert isinstance(comp.position_delta, int)


class TestMultiComponentIntegration:
    """Test integration of multiple components."""
    
    def test_strategy_with_monte_carlo(self, sample_simulation_input):
        """Test strategy exploration with MC validation."""
        # Explore strategies
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator, max_strategies=5)
        
        rankings = explorer.explore_strategies(sample_simulation_input, max_pit_stops=1)
        
        # Validate top strategy with MC
        if rankings:
            best_strategy = rankings[0].strategy
            sample_simulation_input.strategy_to_evaluate = best_strategy
            
            config = MonteCarloConfig(num_runs=30, num_workers=2)
            mc_sim = MonteCarloSimulator(simulator, config)
            mc_result = mc_sim.run_monte_carlo(sample_simulation_input)
            
            assert mc_result["num_runs"] > 0
            assert "statistics" in mc_result
    
    def test_strategy_with_what_if(self, sample_simulation_input):
        """Test strategy exploration with what-if analysis."""
        # Explore strategies
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator, max_strategies=5)
        
        rankings = explorer.explore_strategies(sample_simulation_input, max_pit_stops=1)
        
        # Test top strategies under scenarios
        if len(rankings) >= 2:
            engine = WhatIfEngine(simulator)
            
            strategies = [r.strategy for r in rankings[:2]]
            scenarios = [
                engine.scenario_templates[ScenarioType.EARLY_SAFETY_CAR],
            ]
            
            result = engine.compare_strategies_under_scenarios(
                sample_simulation_input,
                strategies,
                scenarios,
            )
            
            assert "most_robust_strategy_index" in result
            assert result["most_robust_strategy_index"] in [0, 1]


class TestPerformanceRequirements:
    """Test performance requirements are met."""
    
    def test_single_simulation_latency(self, sample_simulation_input):
        """Test single simulation meets <500ms target."""
        simulator = RaceSimulator()
        
        output = simulator.simulate_race(sample_simulation_input)
        
        assert output.metadata["computation_time_ms"] < 500
    
    def test_strategy_exploration_latency(self, sample_simulation_input):
        """Test strategy exploration meets <10s target."""
        import time
        
        simulator = RaceSimulator()
        explorer = StrategyTreeExplorer(simulator, max_strategies=30)
        
        start = time.time()
        rankings = explorer.explore_strategies(sample_simulation_input, max_pit_stops=2)
        elapsed = time.time() - start
        
        assert elapsed < 10.0
        assert len(rankings) > 0


class TestDataIntegrity:
    """Test data integrity throughout workflow."""
    
    def test_position_consistency(self, sample_simulation_input):
        """Test positions remain consistent."""
        simulator = RaceSimulator()
        
        output = simulator.simulate_race(sample_simulation_input)
        
        # Check positions are 1-N
        positions = [r.final_position for r in output.results]
        assert sorted(positions) == list(range(1, len(positions) + 1))
    
    def test_driver_number_preservation(self, sample_simulation_input):
        """Test driver numbers preserved."""
        simulator = RaceSimulator()
        
        input_driver_numbers = {d.driver_number for d in sample_simulation_input.drivers}
        
        output = simulator.simulate_race(sample_simulation_input)
        
        output_driver_numbers = {r.driver_number for r in output.results}
        assert input_driver_numbers == output_driver_numbers


class TestErrorHandling:
    """Test error handling in workflows."""
    
    def test_invalid_strategy_handling(self, sample_simulation_input):
        """Test handling of invalid strategy."""
        # This should be caught by Pydantic validation
        from pydantic import ValidationError
        from simulation.schemas import StrategyOption
        
        with pytest.raises(ValidationError):
            invalid_strategy = StrategyOption(
                pit_laps=[100],  # Beyond race length
                tire_sequence=[TireCompound.SOFT, TireCompound.MEDIUM],
                target_pace="BALANCED",
            )
    
    def test_simulation_robustness(self, sample_simulation_input):
        """Test simulation handles edge cases."""
        simulator = RaceSimulator()
        
        # Mid-race simulation
        sample_simulation_input.race_config.current_lap = 40
        
        output = simulator.simulate_race(sample_simulation_input)
        
        assert output is not None
        assert len(output.results) > 0
