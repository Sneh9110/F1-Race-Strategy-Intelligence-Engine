"""
Mock Weather Data Generator - Realistic weather simulation

Generates track-specific weather with realistic variations.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import random
import math

from data_pipeline.schemas.weather_schema import WeatherData, WeatherForecast, WeatherSession


class MockWeatherGenerator:
    """
    Generate realistic weather data for testing.
    
    Simulates track-specific conditions with realistic variations.
    """
    
    def __init__(self, track_name: str = "Monaco"):
        """Initialize mock weather generator."""
        self.track_name = track_name
        
        # Track-specific base conditions
        self.base_conditions = {
            "Monaco": {"temp": 24, "humidity": 65, "pressure": 1013},
            "Singapore": {"temp": 30, "humidity": 80, "pressure": 1010},
            "Spa": {"temp": 18, "humidity": 70, "pressure": 1015},
            "Bahrain": {"temp": 28, "humidity": 45, "pressure": 1012}
        }
        
        self.base = self.base_conditions.get(track_name, self.base_conditions["Monaco"])
    
    def generate_observation(self, timestamp: Optional[datetime] = None) -> WeatherData:
        """Generate single weather observation."""
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        air_temp = self.base["temp"] + random.gauss(0, 2)
        track_temp = air_temp + random.uniform(8, 15)  # Track hotter than air
        
        return WeatherData(
            timestamp=timestamp,
            air_temperature=round(air_temp, 1),
            track_temperature=round(track_temp, 1),
            humidity=round(self.base["humidity"] + random.gauss(0, 5), 1),
            pressure=round(self.base["pressure"] + random.gauss(0, 3), 1),
            wind_speed=round(random.uniform(0, 15), 1),
            wind_direction=random.randint(0, 359),
            rainfall=0.0 if random.random() > 0.1 else round(random.uniform(0.1, 5), 1),
            weather_condition="Clear" if random.random() > 0.2 else random.choice(["Cloudy", "Overcast"])
        )
    
    def generate_forecasts(
        self,
        hours_ahead: int = 3,
        interval_minutes: int = 30
    ) -> List[WeatherForecast]:
        """Generate weather forecasts."""
        
        forecasts = []
        current_time = datetime.utcnow()
        
        for i in range(hours_ahead * (60 // interval_minutes)):
            forecast_time = current_time + timedelta(minutes=interval_minutes * i)
            
            obs = self.generate_observation(forecast_time)
            
            forecast = WeatherForecast(
                forecast_time=forecast_time,
                air_temperature=obs.air_temperature,
                track_temperature=obs.track_temperature,
                humidity=obs.humidity,
                wind_speed=obs.wind_speed,
                precipitation_probability=round(random.uniform(0, 30), 1),
                weather_condition=obs.weather_condition
            )
            forecasts.append(forecast)
        
        return forecasts
    
    def generate_session(self, duration_minutes: int = 120) -> WeatherSession:
        """Generate full session weather data."""
        
        observations = []
        start_time = datetime.utcnow()
        
        for minute in range(0, duration_minutes, 5):
            obs_time = start_time + timedelta(minutes=minute)
            observations.append(self.generate_observation(obs_time))
        
        forecasts = self.generate_forecasts(hours_ahead=2)
        
        return WeatherSession(
            session_id=f"MOCK_WEATHER_{start_time.strftime('%Y%m%d_%H%M')}",
            track_name=self.track_name,
            observations=observations,
            forecasts=forecasts,
            timestamp=start_time
        )
