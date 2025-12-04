"""Test fixtures for simulation tests."""

import pytest
from simulation.schemas import (
    SimulationInput,
    RaceConfig,
    DriverState,
    StrategyOption,
    TireCompound,
    PaceTarget,
)


@pytest.fixture
def sample_race_config():
    """Sample race configuration."""
    return RaceConfig(
        track_name="Monaco",
        total_laps=78,
        current_lap=1,
        weather_temp=25.0,
        track_temp=35.0,
        grid_positions=list(range(1, 21)),
        safety_car_active=False,
        vsc_active=False,
    )


@pytest.fixture
def sample_drivers():
    """Sample driver states."""
    drivers = []
    for i in range(1, 21):
        driver = DriverState(
            driver_number=i,
            current_position=i,
            tire_compound=TireCompound.MEDIUM,
            tire_age=0,
            fuel_load=110.0,
            gap_to_ahead=0.0 if i == 1 else 1.0 * (i - 1),
            gap_to_behind=1.0 if i < 20 else 0.0,
            recent_lap_times=[90.0] * 5,
            num_pit_stops=0,
            current_stint=1,
            cumulative_race_time=0.0,
        )
        drivers.append(driver)
    return drivers


@pytest.fixture
def sample_strategy():
    """Sample pit strategy."""
    return StrategyOption(
        pit_laps=[25, 50],
        tire_sequence=[TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD],
        target_pace=PaceTarget.BALANCED,
    )


@pytest.fixture
def sample_simulation_input(sample_race_config, sample_drivers, sample_strategy):
    """Sample complete simulation input."""
    return SimulationInput(
        race_config=sample_race_config,
        drivers=sample_drivers,
        strategy_to_evaluate=sample_strategy,
        monte_carlo_runs=0,
    )


@pytest.fixture
def monza_race_config():
    """Monza race configuration."""
    return RaceConfig(
        track_name="Monza",
        total_laps=53,
        current_lap=1,
        weather_temp=28.0,
        track_temp=40.0,
        grid_positions=list(range(1, 21)),
        safety_car_active=False,
        vsc_active=False,
    )


@pytest.fixture
def mid_race_state(sample_race_config, sample_drivers):
    """Mid-race state with tire age and fuel consumption."""
    drivers = []
    for driver in sample_drivers:
        driver_copy = driver.copy(deep=True)
        driver_copy.tire_age = 15
        driver_copy.fuel_load = 85.0
        driver_copy.num_pit_stops = 1
        driver_copy.current_stint = 2
        drivers.append(driver_copy)
    
    config = sample_race_config.copy(deep=True)
    config.current_lap = 35
    
    return SimulationInput(
        race_config=config,
        drivers=drivers,
        strategy_to_evaluate=StrategyOption(
            pit_laps=[50],
            tire_sequence=[TireCompound.MEDIUM, TireCompound.HARD],
            target_pace=PaceTarget.BALANCED,
        ),
        monte_carlo_runs=0,
    )
