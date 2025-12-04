"""Tests for simulation schemas."""

import pytest
from pydantic import ValidationError

from simulation.schemas import (
    DriverState,
    RaceConfig,
    StrategyOption,
    SimulationInput,
    LapResult,
    StintResult,
    PitStopInfo,
    DriverSimulationResult,
    StrategyRanking,
    SimulationOutput,
    TireCompound,
    PaceTarget,
    TrafficState,
)


class TestDriverState:
    """Test DriverState schema."""
    
    def test_valid_driver_state(self):
        """Test valid driver state creation."""
        driver = DriverState(
            driver_number=1,
            current_position=1,
            tire_compound=TireCompound.SOFT,
            tire_age=10,
            fuel_load=95.0,
            gap_to_ahead=0.0,
            gap_to_behind=1.5,
            recent_lap_times=[89.5, 90.0, 89.8],
            num_pit_stops=0,
            current_stint=1,
            cumulative_race_time=1800.0,
        )
        
        assert driver.driver_number == 1
        assert driver.tire_compound == TireCompound.SOFT
        assert driver.tire_age == 10
    
    def test_tire_age_validation(self):
        """Test tire age must be between 0-50."""
        with pytest.raises(ValidationError):
            DriverState(
                driver_number=1,
                current_position=1,
                tire_compound=TireCompound.SOFT,
                tire_age=60,  # Invalid
                fuel_load=95.0,
                gap_to_ahead=0.0,
                gap_to_behind=1.5,
                recent_lap_times=[90.0],
                num_pit_stops=0,
                current_stint=1,
                cumulative_race_time=0.0,
            )
    
    def test_fuel_load_validation(self):
        """Test fuel load must be between 0-110."""
        with pytest.raises(ValidationError):
            DriverState(
                driver_number=1,
                current_position=1,
                tire_compound=TireCompound.SOFT,
                tire_age=10,
                fuel_load=120.0,  # Invalid
                gap_to_ahead=0.0,
                gap_to_behind=1.5,
                recent_lap_times=[90.0],
                num_pit_stops=0,
                current_stint=1,
                cumulative_race_time=0.0,
            )


class TestRaceConfig:
    """Test RaceConfig schema."""
    
    def test_valid_race_config(self):
        """Test valid race config creation."""
        config = RaceConfig(
            track_name="Monaco",
            total_laps=78,
            current_lap=1,
            weather_temp=25.0,
            track_temp=35.0,
            grid_positions=list(range(1, 21)),
            safety_car_active=False,
            vsc_active=False,
        )
        
        assert config.track_name == "Monaco"
        assert config.total_laps == 78
        assert len(config.grid_positions) == 20
    
    def test_current_lap_validation(self):
        """Test current_lap <= total_laps."""
        with pytest.raises(ValidationError):
            RaceConfig(
                track_name="Monaco",
                total_laps=78,
                current_lap=80,  # Invalid
                weather_temp=25.0,
                track_temp=35.0,
                grid_positions=list(range(1, 21)),
                safety_car_active=False,
                vsc_active=False,
            )
    
    def test_consecutive_positions(self):
        """Test grid positions must be consecutive 1-N."""
        with pytest.raises(ValidationError):
            RaceConfig(
                track_name="Monaco",
                total_laps=78,
                current_lap=1,
                weather_temp=25.0,
                track_temp=35.0,
                grid_positions=[1, 2, 4, 5],  # Missing 3
                safety_car_active=False,
                vsc_active=False,
            )


class TestStrategyOption:
    """Test StrategyOption schema."""
    
    def test_valid_strategy(self):
        """Test valid strategy creation."""
        strategy = StrategyOption(
            pit_laps=[25, 50],
            tire_sequence=[TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD],
            target_pace=PaceTarget.BALANCED,
        )
        
        assert len(strategy.pit_laps) == 2
        assert len(strategy.tire_sequence) == 3
    
    def test_pit_laps_ordering(self):
        """Test pit laps must be in ascending order."""
        with pytest.raises(ValidationError):
            StrategyOption(
                pit_laps=[50, 25],  # Wrong order
                tire_sequence=[TireCompound.SOFT, TireCompound.MEDIUM],
                target_pace=PaceTarget.BALANCED,
            )
    
    def test_tire_sequence_length(self):
        """Test tire sequence length must be pit_laps + 1."""
        with pytest.raises(ValidationError):
            StrategyOption(
                pit_laps=[25],
                tire_sequence=[TireCompound.SOFT],  # Should be 2
                target_pace=PaceTarget.BALANCED,
            )


class TestSimulationInput:
    """Test SimulationInput schema."""
    
    def test_valid_simulation_input(self, sample_simulation_input):
        """Test valid simulation input."""
        assert sample_simulation_input.race_config.track_name == "Monaco"
        assert len(sample_simulation_input.drivers) == 20
        assert len(sample_simulation_input.strategy_to_evaluate.pit_laps) == 2
    
    def test_unique_driver_numbers(self, sample_race_config, sample_strategy):
        """Test driver numbers must be unique."""
        drivers = [
            DriverState(
                driver_number=1,
                current_position=1,
                tire_compound=TireCompound.SOFT,
                tire_age=0,
                fuel_load=110.0,
                gap_to_ahead=0.0,
                gap_to_behind=1.0,
                recent_lap_times=[90.0],
                num_pit_stops=0,
                current_stint=1,
                cumulative_race_time=0.0,
            ),
            DriverState(
                driver_number=1,  # Duplicate
                current_position=2,
                tire_compound=TireCompound.SOFT,
                tire_age=0,
                fuel_load=110.0,
                gap_to_ahead=1.0,
                gap_to_behind=1.0,
                recent_lap_times=[90.0],
                num_pit_stops=0,
                current_stint=1,
                cumulative_race_time=0.0,
            ),
        ]
        
        with pytest.raises(ValidationError):
            SimulationInput(
                race_config=sample_race_config,
                drivers=drivers,
                strategy_to_evaluate=sample_strategy,
                monte_carlo_runs=0,
            )


class TestSimulationOutput:
    """Test SimulationOutput schema."""
    
    def test_valid_output(self, sample_race_config):
        """Test valid simulation output."""
        result = DriverSimulationResult(
            driver_number=1,
            final_position=1,
            total_race_time=6500.0,
            laps=[],
            stints=[],
            pit_stops=[],
            win_probability=0.75,
            podium_probability=0.95,
        )
        
        output = SimulationOutput(
            race_config=sample_race_config,
            results=[result],
            metadata={"computation_time_ms": 250.0},
        )
        
        assert output.results[0].driver_number == 1
        assert output.metadata["computation_time_ms"] == 250.0


class TestEnums:
    """Test enum values."""
    
    def test_tire_compounds(self):
        """Test tire compound enum."""
        assert TireCompound.SOFT.value == "SOFT"
        assert TireCompound.MEDIUM.value == "MEDIUM"
        assert TireCompound.HARD.value == "HARD"
    
    def test_pace_targets(self):
        """Test pace target enum."""
        assert PaceTarget.AGGRESSIVE.value == "AGGRESSIVE"
        assert PaceTarget.BALANCED.value == "BALANCED"
        assert PaceTarget.CONSERVATIVE.value == "CONSERVATIVE"
    
    def test_traffic_states(self):
        """Test traffic state enum."""
        assert TrafficState.CLEAN_AIR.value == "CLEAN_AIR"
        assert TrafficState.DIRTY_AIR.value == "DIRTY_AIR"
