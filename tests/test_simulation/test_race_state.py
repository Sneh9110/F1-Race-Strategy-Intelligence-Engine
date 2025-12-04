"""Tests for RaceState class."""

import pytest

from simulation.race_state import RaceState, calculate_position_changes, validate_race_state
from simulation.schemas import RaceConfig, DriverState, TireCompound, TrafficState


class TestRaceStateInitialization:
    """Test RaceState initialization."""
    
    def test_init_from_config(self, sample_race_config, sample_drivers):
        """Test initialization from race config and drivers."""
        state = RaceState(sample_race_config, sample_drivers)
        
        assert state.current_lap == 1
        assert len(state.drivers) == 20
        assert not state.safety_car_active
        assert len(state.event_log) == 0
    
    def test_drivers_dictionary(self, sample_race_config, sample_drivers):
        """Test drivers stored as dictionary by driver_number."""
        state = RaceState(sample_race_config, sample_drivers)
        
        assert 1 in state.drivers
        assert state.drivers[1].driver_number == 1
        assert state.drivers[10].driver_number == 10


class TestDriverStateUpdates:
    """Test driver state update methods."""
    
    def test_update_driver_state(self, sample_race_config, sample_drivers):
        """Test updating driver attributes."""
        state = RaceState(sample_race_config, sample_drivers)
        
        state.update_driver_state(1, tire_age=5, fuel_load=100.0)
        
        assert state.drivers[1].tire_age == 5
        assert state.drivers[1].fuel_load == 100.0
    
    def test_update_nonexistent_driver(self, sample_race_config, sample_drivers):
        """Test updating nonexistent driver raises error."""
        state = RaceState(sample_race_config, sample_drivers)
        
        with pytest.raises(KeyError):
            state.update_driver_state(99, tire_age=5)


class TestLapProgression:
    """Test lap advancement."""
    
    def test_advance_lap(self, sample_race_config, sample_drivers):
        """Test advancing to next lap."""
        state = RaceState(sample_race_config, sample_drivers)
        
        initial_tire_age = state.drivers[1].tire_age
        initial_fuel = state.drivers[1].fuel_load
        
        state.advance_lap(fuel_consumption=1.5)
        
        assert state.current_lap == 2
        assert state.drivers[1].tire_age == initial_tire_age + 1
        assert state.drivers[1].fuel_load == pytest.approx(initial_fuel - 1.5)
    
    def test_fuel_cannot_go_negative(self, sample_race_config, sample_drivers):
        """Test fuel load clamped at zero."""
        state = RaceState(sample_race_config, sample_drivers)
        state.update_driver_state(1, fuel_load=1.0)
        
        state.advance_lap(fuel_consumption=2.0)
        
        assert state.drivers[1].fuel_load == 0.0


class TestPitStops:
    """Test pit stop execution."""
    
    def test_execute_pit_stop(self, sample_race_config, sample_drivers):
        """Test pit stop execution."""
        state = RaceState(sample_race_config, sample_drivers)
        state.update_driver_state(1, tire_age=20, cumulative_race_time=1800.0)
        
        pit_loss = 22.0
        state.execute_pit_stop(1, TireCompound.HARD, pit_loss)
        
        assert state.drivers[1].tire_age == 0
        assert state.drivers[1].tire_compound == TireCompound.HARD
        assert state.drivers[1].cumulative_race_time == pytest.approx(1822.0)
        assert state.drivers[1].num_pit_stops == 1
        assert state.drivers[1].current_stint == 2
    
    def test_pit_stop_logged(self, sample_race_config, sample_drivers):
        """Test pit stop creates event log entry."""
        state = RaceState(sample_race_config, sample_drivers)
        
        state.execute_pit_stop(1, TireCompound.MEDIUM, 22.0)
        
        assert len(state.event_log) == 1
        assert state.event_log[0]["event_type"] == "pit_stop"
        assert state.event_log[0]["driver_number"] == 1


class TestSafetyCarManagement:
    """Test safety car deployment and clearing."""
    
    def test_deploy_safety_car(self, sample_race_config, sample_drivers):
        """Test safety car deployment."""
        state = RaceState(sample_race_config, sample_drivers)
        
        state.deploy_safety_car(lap=10)
        
        assert state.safety_car_active
        assert len(state.event_log) == 1
        assert state.event_log[0]["event_type"] == "safety_car_deployed"
    
    def test_clear_safety_car(self, sample_race_config, sample_drivers):
        """Test safety car clearing."""
        state = RaceState(sample_race_config, sample_drivers)
        state.deploy_safety_car(lap=10)
        
        state.clear_safety_car(lap=13)
        
        assert not state.safety_car_active
        assert len(state.event_log) == 2
        assert state.event_log[1]["event_type"] == "safety_car_cleared"


class TestPositionCalculations:
    """Test position calculation methods."""
    
    def test_get_positions(self, sample_race_config, sample_drivers):
        """Test getting sorted positions."""
        state = RaceState(sample_race_config, sample_drivers)
        
        # Set different cumulative times
        state.update_driver_state(1, cumulative_race_time=1800.0)
        state.update_driver_state(2, cumulative_race_time=1805.0)
        
        positions = state.get_positions()
        
        assert positions[0].driver_number == 1
        assert positions[1].driver_number == 2
    
    def test_recalculate_positions(self, sample_race_config, sample_drivers):
        """Test position recalculation."""
        state = RaceState(sample_race_config, sample_drivers)
        
        # Swap times to change positions
        state.update_driver_state(1, cumulative_race_time=1810.0)
        state.update_driver_state(2, cumulative_race_time=1800.0)
        
        state.recalculate_positions()
        
        assert state.drivers[2].current_position == 1
        assert state.drivers[1].current_position == 2
    
    def test_get_driver_gaps(self, sample_race_config, sample_drivers):
        """Test gap calculation."""
        state = RaceState(sample_race_config, sample_drivers)
        
        state.update_driver_state(1, cumulative_race_time=1800.0)
        state.update_driver_state(2, cumulative_race_time=1803.5)
        
        gaps = state.get_driver_gaps()
        
        assert gaps[1] == 0.0  # Leader
        assert gaps[2] == pytest.approx(3.5)


class TestTrafficState:
    """Test traffic state determination."""
    
    def test_clean_air(self, sample_race_config, sample_drivers):
        """Test driver in clean air."""
        state = RaceState(sample_race_config, sample_drivers)
        
        state.update_driver_state(1, cumulative_race_time=1800.0, current_position=1)
        state.update_driver_state(2, cumulative_race_time=1805.0, current_position=2)
        state.recalculate_positions()
        
        traffic = state.get_traffic_state(1)
        
        assert traffic == TrafficState.CLEAN_AIR
    
    def test_dirty_air(self, sample_race_config, sample_drivers):
        """Test driver in dirty air."""
        state = RaceState(sample_race_config, sample_drivers)
        
        state.update_driver_state(1, cumulative_race_time=1800.0, current_position=1)
        state.update_driver_state(2, cumulative_race_time=1800.5, current_position=2)
        state.recalculate_positions()
        
        traffic = state.get_traffic_state(2)
        
        assert traffic == TrafficState.DIRTY_AIR


class TestStateManipulation:
    """Test state cloning and serialization."""
    
    def test_clone(self, sample_race_config, sample_drivers):
        """Test deep cloning of state."""
        state = RaceState(sample_race_config, sample_drivers)
        state.update_driver_state(1, tire_age=10)
        
        cloned = state.clone()
        
        # Modify original
        state.update_driver_state(1, tire_age=20)
        
        # Clone should be unchanged
        assert cloned.drivers[1].tire_age == 10
        assert state.drivers[1].tire_age == 20
    
    def test_to_dict(self, sample_race_config, sample_drivers):
        """Test serialization to dictionary."""
        state = RaceState(sample_race_config, sample_drivers)
        
        state_dict = state.to_dict()
        
        assert "current_lap" in state_dict
        assert "drivers" in state_dict
        assert "safety_car_active" in state_dict
        assert len(state_dict["drivers"]) == 20


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_calculate_position_changes(self):
        """Test position change detection."""
        old_positions = {1: 1, 2: 2, 3: 3}
        new_positions = {1: 1, 2: 3, 3: 2}
        
        changes = calculate_position_changes(old_positions, new_positions)
        
        assert changes[1] == 0
        assert changes[2] == -1  # Lost position
        assert changes[3] == 1   # Gained position
    
    def test_validate_race_state(self, sample_race_config, sample_drivers):
        """Test race state validation."""
        state = RaceState(sample_race_config, sample_drivers)
        
        # Should not raise
        validate_race_state(state)
    
    def test_validate_duplicate_positions(self, sample_race_config, sample_drivers):
        """Test validation catches duplicate positions."""
        state = RaceState(sample_race_config, sample_drivers)
        
        state.update_driver_state(1, current_position=1)
        state.update_driver_state(2, current_position=1)  # Duplicate
        
        with pytest.raises(ValueError, match="Duplicate positions"):
            validate_race_state(state)
