"""
Mock Safety Car Generator - Safety car event simulation

Generates realistic SC/VSC events based on probabilities.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import random

from data_pipeline.schemas.safety_car_schema import SafetyCarEvent, IncidentLog


class MockSafetyCarGenerator:
    """
    Generate realistic safety car events.
    
    Uses track-specific probabilities for SC/VSC/Red Flags.
    """
    
    def __init__(self, track_name: str = "Monaco"):
        """Initialize mock SC generator."""
        self.track_name = track_name
        
        # Track-specific SC probabilities (per race)
        self.sc_probabilities = {
            "Monaco": 0.7,  # High probability
            "Singapore": 0.6,
            "Baku": 0.65,
            "Jeddah": 0.55,
            "Spa": 0.25,
            "Monza": 0.20
        }
        
        self.sc_prob = self.sc_probabilities.get(track_name, 0.35)
    
    def generate_race_events(
        self,
        race_duration_laps: int = 78,
        **kwargs
    ) -> List[SafetyCarEvent]:
        """Generate safety car events for a race."""
        
        events = []
        
        # Determine number of SC events
        if random.random() < self.sc_prob:
            num_events = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        else:
            return events  # No SC events
        
        for i in range(num_events):
            # Random lap for incident (avoid first/last 5 laps)
            incident_lap = random.randint(6, race_duration_laps - 5)
            
            # Event type
            event_type = random.choices(
                ["SAFETY_CAR", "VIRTUAL_SAFETY_CAR", "RED_FLAG"],
                weights=[0.5, 0.4, 0.1]
            )[0]
            
            # Duration
            if event_type == "RED_FLAG":
                duration_laps = random.randint(3, 10)
            elif event_type == "SAFETY_CAR":
                duration_laps = random.randint(2, 5)
            else:  # VSC
                duration_laps = random.randint(1, 3)
            
            # Reason
            reasons = [
                "Collision between drivers",
                "Debris on track",
                "Barrier damage",
                "Car breakdown",
                "Weather conditions",
                "Medical intervention"
            ]
            reason = random.choice(reasons)
            
            # Incident details
            incident = IncidentLog(
                lap_number=incident_lap,
                sector=random.randint(1, 3),
                description=reason,
                drivers_involved=[f"Driver_{random.randint(1, 20)}"],
                severity=random.choice(["LOW", "MEDIUM", "HIGH"])
            )
            
            event = SafetyCarEvent(
                event_type=event_type,
                start_lap=incident_lap,
                end_lap=incident_lap + duration_laps,
                duration_laps=duration_laps,
                reason=reason,
                incident=incident,
                timestamp=datetime.utcnow() + timedelta(minutes=incident_lap * 1.5)
            )
            
            events.append(event)
        
        # Sort by start lap
        events.sort(key=lambda x: x.start_lap)
        
        return events
    
    def generate_session_events(self, session_type: str = "Race") -> List[SafetyCarEvent]:
        """Generate events for different session types."""
        
        if session_type == "Race":
            return self.generate_race_events()
        elif session_type in ["Q", "Qualifying"]:
            # Qualifying rarely has SC
            if random.random() < 0.05:
                return self.generate_race_events(race_duration_laps=20)
            return []
        else:  # Practice
            return []
