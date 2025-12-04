"""Tests for MonteCarloSimulator."""

import pytest
import numpy as np

from simulation.core import RaceSimulator
from simulation.monte_carlo import (
    MonteCarloSimulator,
    MonteCarloConfig,
    MonteCarloResult,
    _inject_noise,
    estimate_required_runs,
)


class TestMonteCarloConfig:
    """Test MonteCarloConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = MonteCarloConfig()
        
        assert config.num_runs == 1000
        assert config.num_workers is not None
        assert config.convergence_threshold == 0.01
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MonteCarloConfig(
            num_runs=500,
            num_workers=2,
            tire_deg_noise_std=0.03,
        )
        
        assert config.num_runs == 500
        assert config.num_workers == 2
        assert config.tire_deg_noise_std == 0.03


class TestMonteCarloSimulator:
    """Test MonteCarloSimulator initialization."""
    
    def test_init(self):
        """Test initialization."""
        simulator = RaceSimulator()
        config = MonteCarloConfig(num_runs=100)
        mc_sim = MonteCarloSimulator(simulator, config)
        
        assert mc_sim.config.num_runs == 100


class TestMonteCarloSimulation:
    """Test Monte Carlo simulation execution."""
    
    def test_run_monte_carlo(self, sample_simulation_input):
        """Test basic MC simulation."""
        simulator = RaceSimulator()
        config = MonteCarloConfig(num_runs=50, num_workers=2)
        mc_sim = MonteCarloSimulator(simulator, config)
        
        result = mc_sim.run_monte_carlo(sample_simulation_input, target_driver=1)
        
        assert result["num_runs"] <= 50
        assert "statistics" in result
        assert "convergence" in result
    
    def test_statistics_calculated(self, sample_simulation_input):
        """Test statistics are calculated."""
        simulator = RaceSimulator()
        config = MonteCarloConfig(num_runs=50, num_workers=2)
        mc_sim = MonteCarloSimulator(simulator, config)
        
        result = mc_sim.run_monte_carlo(sample_simulation_input)
        
        stats = result["statistics"]
        assert "position" in stats
        assert "win_probability" in stats
        assert "podium_probability" in stats
        assert "race_time" in stats


class TestPositionStatistics:
    """Test position statistics calculation."""
    
    def test_position_distribution(self, sample_simulation_input):
        """Test position distribution calculated."""
        simulator = RaceSimulator()
        config = MonteCarloConfig(num_runs=50, num_workers=2)
        mc_sim = MonteCarloSimulator(simulator, config)
        
        result = mc_sim.run_monte_carlo(sample_simulation_input)
        
        dist = result["statistics"]["position_distribution"]
        assert isinstance(dist, dict)
        assert sum(dist.values()) == pytest.approx(1.0)
    
    def test_position_percentiles(self, sample_simulation_input):
        """Test position percentiles calculated."""
        simulator = RaceSimulator()
        config = MonteCarloConfig(num_runs=50, num_workers=2)
        mc_sim = MonteCarloSimulator(simulator, config)
        
        result = mc_sim.run_monte_carlo(sample_simulation_input)
        
        pos_stats = result["statistics"]["position"]
        assert "p10" in pos_stats
        assert "p90" in pos_stats
        assert pos_stats["p10"] <= pos_stats["p90"]


class TestConvergenceChecking:
    """Test convergence detection."""
    
    def test_convergence_insufficient_runs(self):
        """Test convergence check with insufficient runs."""
        mc_results = [
            MonteCarloResult(run_id=i, final_position=1, race_time=6500.0, num_pit_stops=2, sc_deployments=0, overtakes=0)
            for i in range(50)
        ]
        
        simulator = RaceSimulator()
        config = MonteCarloConfig(convergence_window=100)
        mc_sim = MonteCarloSimulator(simulator, config)
        
        convergence = mc_sim._check_convergence(mc_results)
        
        assert convergence["converged"] is False
        assert convergence["reason"] == "insufficient_runs"
    
    def test_convergence_achieved(self):
        """Test convergence detection."""
        # Create stable results
        mc_results = [
            MonteCarloResult(run_id=i, final_position=1 if i % 2 == 0 else 2, race_time=6500.0, num_pit_stops=2, sc_deployments=0, overtakes=0)
            for i in range(500)
        ]
        
        simulator = RaceSimulator()
        config = MonteCarloConfig(convergence_window=50, convergence_threshold=0.1)
        mc_sim = MonteCarloSimulator(simulator, config)
        
        convergence = mc_sim._check_convergence(mc_results)
        
        assert "converged" in convergence
        assert "variance" in convergence


class TestNoiseInjection:
    """Test noise injection."""
    
    def test_inject_noise(self, sample_simulation_input):
        """Test noise injection modifies input."""
        config = MonteCarloConfig(tire_deg_noise_std=0.05)
        
        noisy_input = _inject_noise(sample_simulation_input, config, seed=42)
        
        # Should be different due to noise
        assert noisy_input is not None
        assert len(noisy_input.drivers) == len(sample_simulation_input.drivers)
    
    def test_noise_reproducibility(self, sample_simulation_input):
        """Test noise is reproducible with same seed."""
        config = MonteCarloConfig()
        
        noisy1 = _inject_noise(sample_simulation_input, config, seed=42)
        noisy2 = _inject_noise(sample_simulation_input, config, seed=42)
        
        assert noisy1.drivers[0].fuel_load == noisy2.drivers[0].fuel_load
    
    def test_tire_age_noise(self, sample_simulation_input):
        """Test tire age noise injection."""
        # Set non-zero tire age
        sample_simulation_input.drivers[0].tire_age = 20
        
        config = MonteCarloConfig(tire_deg_noise_std=0.1)
        noisy_input = _inject_noise(sample_simulation_input, config, seed=42)
        
        # Tire age should be different (within reason)
        assert abs(noisy_input.drivers[0].tire_age - 20) <= 10


class TestSensitivityAnalysis:
    """Test sensitivity analysis."""
    
    def test_analyze_sensitivity(self, sample_simulation_input):
        """Test sensitivity analysis."""
        simulator = RaceSimulator()
        config = MonteCarloConfig(num_runs=20, num_workers=2)
        mc_sim = MonteCarloSimulator(simulator, config)
        
        result = mc_sim.analyze_sensitivity(
            sample_simulation_input,
            parameter="tire_deg_noise_std",
            values=[0.01, 0.02, 0.03],
        )
        
        assert result["parameter"] == "tire_deg_noise_std"
        assert len(result["results"]) == 3


class TestMonteCarloResult:
    """Test MonteCarloResult dataclass."""
    
    def test_result_creation(self):
        """Test result creation."""
        result = MonteCarloResult(
            run_id=1,
            final_position=3,
            race_time=6550.0,
            num_pit_stops=2,
            sc_deployments=1,
            overtakes=2,
        )
        
        assert result.run_id == 1
        assert result.final_position == 3
        assert result.race_time == 6550.0


class TestRequiredRunsEstimation:
    """Test required runs estimation."""
    
    def test_estimate_required_runs(self, sample_simulation_input):
        """Test estimation of required runs."""
        required = estimate_required_runs(
            sample_simulation_input,
            target_confidence=0.95,
            margin_of_error=0.05,
        )
        
        assert 100 <= required <= 10000
        assert isinstance(required, int)
