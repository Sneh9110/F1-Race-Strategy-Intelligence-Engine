"""
Mock Telemetry Generator - Physics-based telemetry simulation

Generates high-frequency telemetry with realistic physics.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import random
import math

from data_pipeline.schemas.telemetry_schema import TelemetryPoint, TelemetrySession


class MockTelemetryGenerator:
    """
    Generate realistic telemetry data.
    
    Simulates car telemetry with basic physics modeling.
    """
    
    def __init__(self, track_name: str = "Monaco"):
        """Initialize mock telemetry generator."""
        self.track_name = track_name
        self.max_speed = 340 if track_name != "Monaco" else 260
        self.lap_distance = 5.412 if track_name == "Monaco" else 5.5  # km
    
    def generate_lap_telemetry(
        self,
        lap_number: int,
        sample_rate: int = 10,  # Hz
        lap_time: float = 72.5  # seconds
    ) -> List[TelemetryPoint]:
        """Generate telemetry points for one lap."""
        
        points = []
        num_samples = int(lap_time * sample_rate)
        
        for i in range(num_samples):
            elapsed = i / sample_rate
            distance = (elapsed / lap_time) * self.lap_distance * 1000  # meters
            
            # Simulate speed profile (simplified)
            progress = elapsed / lap_time
            speed_factor = 0.5 + 0.5 * math.sin(progress * math.pi * 8)  # Multiple corners
            speed = self.max_speed * speed_factor + random.gauss(0, 5)
            
            # Throttle/brake based on speed change
            if i > 0:
                speed_delta = speed - points[-1].speed
                throttle = max(0, min(100, 50 + speed_delta * 10))
                brake = max(0, min(100, -speed_delta * 15))
            else:
                throttle = 75
                brake = 0
            
            # Gear based on speed
            gear = min(8, max(1, int(speed / 40) + 1))
            
            # RPM based on gear
            rpm = int(8000 + (speed / self.max_speed) * 7000)
            
            # Temperatures increase over lap
            engine_temp = 90 + progress * 20 + random.gauss(0, 2)
            brake_temp = {
                "FL": 300 + brake * 5 + random.gauss(0, 10),
                "FR": 300 + brake * 5 + random.gauss(0, 10),
                "RL": 250 + brake * 3 + random.gauss(0, 10),
                "RR": 250 + brake * 3 + random.gauss(0, 10)
            }
            tire_temp = {
                "FL": 85 + progress * 10 + random.gauss(0, 3),
                "FR": 85 + progress * 10 + random.gauss(0, 3),
                "RL": 80 + progress * 8 + random.gauss(0, 3),
                "RR": 80 + progress * 8 + random.gauss(0, 3)
            }
            
            point = TelemetryPoint(
                timestamp=datetime.utcnow() + timedelta(seconds=elapsed),
                lap_number=lap_number,
                distance=round(distance, 2),
                speed=round(speed, 1),
                throttle=round(throttle, 1),
                brake=round(brake, 1),
                gear=gear,
                rpm=rpm,
                drs=1 if speed > 200 and random.random() > 0.7 else 0,
                engine_temp=round(engine_temp, 1),
                brake_temp=brake_temp,
                tire_temp=tire_temp
            )
            points.append(point)
        
        return points
    
    def generate_session(
        self,
        num_laps: int = 10,
        sample_rate: int = 10
    ) -> TelemetrySession:
        """Generate full session telemetry."""
        
        all_points = []
        
        for lap in range(1, num_laps + 1):
            lap_points = self.generate_lap_telemetry(lap, sample_rate)
            all_points.extend(lap_points)
        
        return TelemetrySession(
            session_id=f"MOCK_TELEMETRY_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
            driver="Mock Driver",
            track_name=self.track_name,
            telemetry_points=all_points,
            timestamp=datetime.utcnow()
        )
