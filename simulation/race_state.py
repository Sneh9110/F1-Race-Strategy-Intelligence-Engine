"""
Race state management for simulation.

Tracks mutable race state (positions, gaps, events) during simulation execution.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from copy import deepcopy
import logging

from .schemas import DriverState, RaceConfig, TireCompound, TrafficState

logger = logging.getLogger(__name__)


class RaceState:
    """Mutable race state during simulation."""
    
    def __init__(self, race_config: RaceConfig, drivers: List[DriverState]):
        """
        Initialize race state.
        
        Args:
            race_config: Race configuration
            drivers: Initial driver states
        """
        self.race_config = race_config
        self.current_lap = race_config.current_lap
        self.drivers: Dict[int, DriverState] = {d.driver_number: d for d in drivers}
        self.safety_car_active = race_config.safety_car_active
        self.vsc_active = race_config.vsc_active
        self.event_log: List[Dict[str, Any]] = []
        
        logger.info(
            f"Initialized race state: track={race_config.track_name}, "
            f"laps={race_config.total_laps}, drivers={len(drivers)}"
        )
    
    def update_driver_state(self, driver_number: int, **kwargs) -> None:
        """
        Update specific driver attributes.
        
        Args:
            driver_number: Driver to update
            **kwargs: Attributes to update (tire_age, fuel_load, position, etc.)
        """
        if driver_number not in self.drivers:
            raise ValueError(f"Driver {driver_number} not found in race state")
        
        driver = self.drivers[driver_number]
        for key, value in kwargs.items():
            if hasattr(driver, key):
                setattr(driver, key, value)
            else:
                logger.warning(f"Unknown driver attribute: {key}")
    
    def advance_lap(self, fuel_consumption_per_lap: float = 1.5) -> None:
        """
        Advance to next lap and update fuel/tire ages.
        
        Args:
            fuel_consumption_per_lap: Fuel consumed per lap in kg
        """
        self.current_lap += 1
        
        for driver in self.drivers.values():
            # Update fuel load
            driver.fuel_load = max(0.0, driver.fuel_load - fuel_consumption_per_lap)
            
            # Update tire age (increment for all drivers not pitting)
            driver.tire_age += 1
        
        logger.debug(f"Advanced to lap {self.current_lap}")
    
    def execute_pit_stop(
        self,
        driver_number: int,
        new_compound: TireCompound,
        pit_loss: float
    ) -> None:
        """
        Execute pit stop for a driver.
        
        Args:
            driver_number: Driver pitting
            new_compound: New tire compound
            pit_loss: Time loss in seconds
        """
        if driver_number not in self.drivers:
            raise ValueError(f"Driver {driver_number} not found")
        
        driver = self.drivers[driver_number]
        old_compound = driver.tire_compound
        
        # Update driver state
        driver.tire_age = 0
        driver.tire_compound = new_compound
        driver.pit_stops_completed += 1
        driver.stint_number += 1
        driver.cumulative_race_time += pit_loss
        
        # Log event
        self.event_log.append({
            "lap": self.current_lap,
            "event_type": "pit_stop",
            "driver_number": driver_number,
            "old_compound": old_compound,
            "new_compound": new_compound,
            "pit_loss": pit_loss,
        })
        
        logger.info(
            f"Lap {self.current_lap}: Driver {driver_number} pitted "
            f"({old_compound} -> {new_compound}), loss={pit_loss:.2f}s"
        )
    
    def deploy_safety_car(self, lap_number: Optional[int] = None) -> None:
        """
        Deploy safety car.
        
        Args:
            lap_number: Lap of deployment (defaults to current_lap)
        """
        self.safety_car_active = True
        lap = lap_number or self.current_lap
        
        self.event_log.append({
            "lap": lap,
            "event_type": "safety_car_deployed",
        })
        
        logger.info(f"Lap {lap}: Safety car deployed")
    
    def clear_safety_car(self) -> None:
        """Clear safety car."""
        self.safety_car_active = False
        
        self.event_log.append({
            "lap": self.current_lap,
            "event_type": "safety_car_cleared",
        })
        
        logger.info(f"Lap {self.current_lap}: Safety car cleared")
    
    def get_driver_gaps(self) -> Dict[int, float]:
        """
        Calculate gaps between drivers based on cumulative times.
        
        Returns:
            Dict mapping driver_number to gap to leader in seconds
        """
        sorted_drivers = sorted(
            self.drivers.values(),
            key=lambda d: d.cumulative_race_time
        )
        
        if not sorted_drivers:
            return {}
        
        leader_time = sorted_drivers[0].cumulative_race_time
        gaps = {}
        
        for driver in sorted_drivers:
            gaps[driver.driver_number] = driver.cumulative_race_time - leader_time
        
        return gaps
    
    def get_positions(self) -> List[int]:
        """
        Get sorted list of driver numbers by position.
        
        Returns:
            List of driver_numbers in race order (P1 to P20)
        """
        sorted_drivers = sorted(
            self.drivers.values(),
            key=lambda d: (d.cumulative_race_time, d.driver_number)
        )
        
        return [d.driver_number for d in sorted_drivers]
    
    def recalculate_positions(self) -> None:
        """Recalculate driver positions based on cumulative times."""
        position_order = self.get_positions()
        
        for position, driver_number in enumerate(position_order, start=1):
            self.drivers[driver_number].current_position = position
        
        # Update gaps
        gaps = self.get_driver_gaps()
        position_order_list = list(position_order)
        
        for i, driver_number in enumerate(position_order):
            driver = self.drivers[driver_number]
            
            # Gap to ahead
            if i > 0:
                ahead_number = position_order_list[i - 1]
                driver.gap_to_ahead = (
                    driver.cumulative_race_time - 
                    self.drivers[ahead_number].cumulative_race_time
                )
            else:
                driver.gap_to_ahead = 0.0
            
            # Gap to behind
            if i < len(position_order) - 1:
                behind_number = position_order_list[i + 1]
                driver.gap_to_behind = (
                    self.drivers[behind_number].cumulative_race_time - 
                    driver.cumulative_race_time
                )
            else:
                driver.gap_to_behind = None
    
    def get_traffic_state(
        self,
        driver_number: int,
        dirty_air_threshold: float = 1.0
    ) -> TrafficState:
        """
        Determine if driver is in clean or dirty air.
        
        Args:
            driver_number: Driver to check
            dirty_air_threshold: Gap threshold for dirty air (seconds)
        
        Returns:
            TrafficState.CLEAN_AIR or TrafficState.DIRTY_AIR
        """
        if driver_number not in self.drivers:
            return TrafficState.CLEAN_AIR
        
        driver = self.drivers[driver_number]
        
        if driver.gap_to_ahead is None or driver.gap_to_ahead > dirty_air_threshold:
            return TrafficState.CLEAN_AIR
        
        return TrafficState.DIRTY_AIR
    
    def clone(self) -> RaceState:
        """
        Create deep copy for Monte Carlo branching.
        
        Returns:
            New RaceState instance with copied data
        """
        new_state = RaceState.__new__(RaceState)
        new_state.race_config = deepcopy(self.race_config)
        new_state.current_lap = self.current_lap
        new_state.drivers = {k: deepcopy(v) for k, v in self.drivers.items()}
        new_state.safety_car_active = self.safety_car_active
        new_state.vsc_active = self.vsc_active
        new_state.event_log = deepcopy(self.event_log)
        
        return new_state
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize state for logging/debugging.
        
        Returns:
            Dict representation of race state
        """
        return {
            "current_lap": self.current_lap,
            "safety_car_active": self.safety_car_active,
            "vsc_active": self.vsc_active,
            "drivers": {
                num: {
                    "position": d.current_position,
                    "tire_compound": d.tire_compound,
                    "tire_age": d.tire_age,
                    "fuel_load": d.fuel_load,
                    "cumulative_time": d.cumulative_race_time,
                }
                for num, d in self.drivers.items()
            },
            "event_count": len(self.event_log),
        }


def calculate_position_changes(old_state: RaceState, new_state: RaceState) -> List[Dict[str, Any]]:
    """
    Detect overtakes between states.
    
    Args:
        old_state: Previous race state
        new_state: Current race state
    
    Returns:
        List of overtake events
    """
    overtakes = []
    
    for driver_number in old_state.drivers:
        old_pos = old_state.drivers[driver_number].current_position
        new_pos = new_state.drivers[driver_number].current_position
        
        if old_pos != new_pos:
            overtakes.append({
                "driver_number": driver_number,
                "old_position": old_pos,
                "new_position": new_pos,
                "positions_gained": old_pos - new_pos,
            })
    
    return overtakes


def validate_race_state(state: RaceState) -> bool:
    """
    Ensure race state consistency.
    
    Args:
        state: Race state to validate
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If state is inconsistent
    """
    positions = [d.current_position for d in state.drivers.values()]
    
    # Check for duplicate positions
    if len(positions) != len(set(positions)):
        raise ValueError("Duplicate positions detected")
    
    # Check position range
    if not all(1 <= p <= len(positions) for p in positions):
        raise ValueError("Invalid position range")
    
    # Check tire ages
    for driver in state.drivers.values():
        if driver.tire_age < 0 or driver.tire_age > 50:
            raise ValueError(f"Invalid tire age: {driver.tire_age}")
    
    # Check fuel loads
    for driver in state.drivers.values():
        if driver.fuel_load < 0 or driver.fuel_load > 110:
            raise ValueError(f"Invalid fuel load: {driver.fuel_load}")
    
    return True
