"""Tests for RaceSimulator core."""

import pytest
from unittest.mock import Mock, patch

from simulation.core import RaceSimulator
from simulation.schemas import SimulationInput, TireCompound


class TestRaceSimulatorInitialization:
    """Test RaceSimulator initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        simulator = RaceSimulator()
        
        assert "degradation" in simulator.predictors
        assert "lap_time" in simulator.predictors
        assert "safety_car" in simulator.predictors
        assert "pit_stop_loss" in simulator.predictors
    
    def test_init_with_cache(self):
        """Test initialization with cache client."""
        mock_cache = Mock()
        simulator = RaceSimulator(cache_client=mock_cache)
        
        assert simulator.cache_client is not None


class TestRaceSimulation:
    """Test race simulation execution."""
    
    def test_simulate_race_basic(self, sample_simulation_input):
        """Test basic race simulation."""
        simulator = RaceSimulator()
        
        output = simulator.simulate_race(sample_simulation_input)
        
        assert output is not None
        assert len(output.results) == 20
        assert output.metadata["track_name"] == "Monaco"
        assert "computation_time_ms" in output.metadata
    
    def test_simulation_performance(self, sample_simulation_input):
        """Test simulation meets performance target (<500ms)."""
        simulator = RaceSimulator()
        
        output = simulator.simulate_race(sample_simulation_input)
        
        assert output.metadata["computation_time_ms"] < 500
    
    def test_results_sorted_by_position(self, sample_simulation_input):
        """Test results sorted by final position."""
        simulator = RaceSimulator()
        
        output = simulator.simulate_race(sample_simulation_input)
        
        for i in range(len(output.results) - 1):
            assert output.results[i].final_position <= output.results[i + 1].final_position


class TestLapSimulation:
    """Test single lap simulation."""
    
    @patch.object(RaceSimulator, '_predict_lap_time')
    def test_simulate_lap_calls_predictors(self, mock_lap_time, sample_simulation_input):
        """Test lap simulation calls predictors."""
        mock_lap_time.return_value = 90.0
        
        simulator = RaceSimulator()
        output = simulator.simulate_race(sample_simulation_input)
        
        assert mock_lap_time.called


class TestPitStopPrediction:
    """Test pit stop prediction."""
    
    def test_pit_loss_prediction(self, sample_simulation_input):
        """Test pit loss is predicted."""
        simulator = RaceSimulator()
        
        output = simulator.simulate_race(sample_simulation_input)
        
        # Check driver who pitted
        driver_result = output.results[0]
        if driver_result.pit_stops:
            assert driver_result.pit_stops[0].loss > 0


class TestSafetyCarPrediction:
    """Test safety car prediction."""
    
    def test_safety_car_probability(self, sample_simulation_input):
        """Test SC probability is calculated."""
        simulator = RaceSimulator()
        
        # Force SC deployment scenario
        sample_simulation_input.what_if_params = {
            "inject_safety_car": True,
            "sc_lap": 15,
            "sc_duration": 3,
        }
        
        output = simulator.simulate_race(sample_simulation_input)
        
        assert output is not None


class TestFallbackMechanisms:
    """Test fallback mechanisms when predictions fail."""
    
    @patch.object(RaceSimulator, '_predict_lap_time')
    def test_lap_time_fallback(self, mock_predict, sample_simulation_input):
        """Test fallback lap time calculation."""
        mock_predict.side_effect = Exception("Model failure")
        
        simulator = RaceSimulator()
        
        # Should not raise, uses fallback
        output = simulator.simulate_race(sample_simulation_input)
        
        assert output is not None
        assert len(output.results) > 0


class TestCaching:
    """Test result caching."""
    
    def test_cache_miss(self, sample_simulation_input):
        """Test cache miss scenario."""
        mock_cache = Mock()
        mock_cache.get.return_value = None
        
        simulator = RaceSimulator(cache_client=mock_cache)
        output = simulator.simulate_race(sample_simulation_input)
        
        assert simulator.performance_stats["cache_misses"] == 1
    
    def test_cache_set(self, sample_simulation_input):
        """Test result is cached."""
        mock_cache = Mock()
        mock_cache.get.return_value = None
        
        simulator = RaceSimulator(cache_client=mock_cache)
        output = simulator.simulate_race(sample_simulation_input)
        
        assert mock_cache.setex.called


class TestTrackConfiguration:
    """Test track configuration loading."""
    
    def test_get_track_config(self):
        """Test track config retrieval."""
        simulator = RaceSimulator()
        
        config = simulator._get_track_config("Monaco")
        
        assert "base_lap_time_seconds" in config
        assert "pit_loss_seconds" in config
    
    def test_track_config_defaults(self):
        """Test default config for unknown track."""
        simulator = RaceSimulator()
        
        config = simulator._get_track_config("UnknownTrack")
        
        assert config["base_lap_time_seconds"] == 90.0


class TestPerformanceTracking:
    """Test performance statistics tracking."""
    
    def test_performance_stats_updated(self, sample_simulation_input):
        """Test performance stats are tracked."""
        simulator = RaceSimulator()
        
        simulator.simulate_race(sample_simulation_input)
        
        assert simulator.performance_stats["total_simulations"] == 1
        assert simulator.performance_stats["total_latency_ms"] > 0
    
    def test_model_call_tracking(self, sample_simulation_input):
        """Test model calls are tracked."""
        simulator = RaceSimulator()
        
        simulator.simulate_race(sample_simulation_input)
        
        assert simulator.performance_stats["model_calls"]["lap_time"] > 0
