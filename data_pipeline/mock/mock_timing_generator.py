"""
Mock Timing Data Generator - Realistic lap time simulation

Generates realistic lap times with tire degradation, fuel loads, track evolution.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import random

from data_pipeline.schemas.timing_schema import TimingPoint, LapData, SessionTiming


class MockTimingGenerator:
    """
    Generate realistic timing data for testing.
    
    Simulates:
    - Base lap time + variance
    - Tire degradation
    - Fuel load effect
    - Track evolution
    - Traffic/incidents
    """
    
    def __init__(self, track_name: str = "Monaco", base_laptime: float = 72.5):
        """Initialize mock generator."""
        self.track_name = track_name
        self.base_laptime = base_laptime  # seconds
        
        self.tire_deg_rate = 0.03  # seconds per lap
        self.fuel_effect = 0.035  # seconds per lap of fuel
        self.track_evolution = -0.02  # seconds per lap
    
    def generate_lap(
        self,
        driver: str,
        lap_number: int,
        tire_age: int,
        fuel_laps_remaining: int,
        **kwargs
    ) -> LapData:
        """Generate single lap data."""
        
        # Calculate lap time components
        tire_deg = tire_age * self.tire_deg_rate
        fuel_weight = fuel_laps_remaining * self.fuel_effect
        track_imp = lap_number * self.track_evolution
        variance = random.gauss(0, 0.15)
        
        lap_time = self.base_laptime + tire_deg + fuel_weight + track_imp + variance
        
        # Generate sector times (roughly equal with variance)
        s1 = lap_time * 0.33 + random.gauss(0, 0.05)
        s2 = lap_time * 0.33 + random.gauss(0, 0.05)
        s3 = lap_time - s1 - s2
        
        return LapData(
            lap_number=lap_number,
            lap_time=round(lap_time, 3),
            sector1_time=round(s1, 3),
            sector2_time=round(s2, 3),
            sector3_time=round(s3, 3),
            position=kwargs.get('position', random.randint(1, 20)),
            gap_to_leader=kwargs.get('gap', lap_number * random.uniform(0.3, 1.5)),
            tire_compound=kwargs.get('compound', 'SOFT'),
            tire_age=tire_age,
            is_pit_lap=kwargs.get('is_pit_lap', False),
            timestamp=datetime.utcnow()
        )
    
    def generate_session(
        self,
        num_drivers: int = 20,
        num_laps: int = 78,
        **kwargs
    ) -> SessionTiming:
        """Generate full session timing data."""
        
        drivers = [f"Driver_{i}" for i in range(1, num_drivers + 1)]
        all_timing = []
        
        for driver in drivers:
            tire_age = 0
            fuel_laps = num_laps
            
            for lap in range(1, num_laps + 1):
                lap_data = self.generate_lap(
                    driver=driver,
                    lap_number=lap,
                    tire_age=tire_age,
                    fuel_laps_remaining=fuel_laps,
                    position=drivers.index(driver) + 1
                )
                
                timing_point = TimingPoint(
                    driver=driver,
                    lap_data=lap_data
                )
                all_timing.append(timing_point)
                
                tire_age += 1
                fuel_laps -= 1
                
                # Simulate pit stops
                if tire_age > random.randint(15, 30):
                    tire_age = 0
        
        return SessionTiming(
            session_id=kwargs.get('session_id', 'MOCK_SESSION'),
            session_name=kwargs.get('session_name', 'Race'),
            track_name=self.track_name,
            timing_points=all_timing,
            timestamp=datetime.utcnow()
        )
