"""
Physics-Based Fallback Heuristics

Provides fallback predictions when ML models are unavailable.
Uses domain knowledge and physics-based calculations.
"""

import logging
from typing import Dict, Optional
import yaml
from pathlib import Path

from .base import PredictionInput, PredictionOutput, RaceCondition
from config.settings import Settings

logger = logging.getLogger(__name__)


class FallbackHeuristics:
    """
    Physics-based lap time predictions.
    
    Uses domain knowledge to estimate lap times when ML models fail:
    - Base lap time from track data
    - Tire degradation from compound characteristics
    - Fuel load effect (~0.03s per kg)
    - Traffic penalty (exponential decay with gap)
    - Safety car factor (30% slower)
    - Weather adjustment (temperature effect)
    
    Attributes:
        track_data: Track characteristics and base lap times
        tire_compounds: Tire compound characteristics
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize fallback heuristics.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = config_path or Path("config")
        self.track_data = self._load_track_data()
        self.tire_compounds = self._load_tire_compounds()
    
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Predict lap time using physics-based heuristics.
        
        Args:
            input_data: Prediction input
            
        Returns:
            Prediction output with fallback metadata
        """
        # Get base lap time for track
        base_lap_time = self._get_base_lap_time(input_data.track_name)
        
        # Calculate tire effect
        tire_effect = self._calculate_tire_effect(
            input_data.tire_age,
            input_data.tire_compound
        )
        
        # Calculate fuel effect
        fuel_effect = self._calculate_fuel_effect(input_data.fuel_load)
        
        # Calculate traffic penalty
        traffic_penalty = self._calculate_traffic_penalty(
            input_data.traffic_state,
            input_data.gap_to_ahead
        )
        
        # Calculate weather adjustment
        weather_adjustment = self._calculate_weather_adjustment(
            input_data.weather_temp
        )
        
        # Calculate safety car factor
        safety_car_factor = self._calculate_safety_car_factor(
            input_data.safety_car_active
        )
        
        # Combine components
        predicted_lap_time = (
            (base_lap_time + tire_effect + fuel_effect + traffic_penalty + weather_adjustment)
            * safety_car_factor
        )
        
        pace_components = {
            'base_pace': base_lap_time,
            'tire_effect': tire_effect,
            'fuel_effect': fuel_effect,
            'traffic_penalty': traffic_penalty,
            'weather_adjustment': weather_adjustment,
            'safety_car_factor': safety_car_factor,
        }
        
        # Lower confidence for fallback predictions
        confidence = 0.6
        
        return PredictionOutput(
            predicted_lap_time=predicted_lap_time,
            confidence=confidence,
            pace_components=pace_components,
            metadata={
                'model_type': 'fallback_heuristics',
                'method': 'physics_based',
                'warning': 'Using fallback predictions - ML model unavailable',
            }
        )
    
    def _get_base_lap_time(self, track_name: str) -> float:
        """
        Get base lap time for track.
        
        Args:
            track_name: Name of the track
            
        Returns:
            Base lap time in seconds
        """
        if track_name in self.track_data:
            return self.track_data[track_name].get('base_lap_time', 90.0)
        
        # Default if track not found
        logger.warning(f"Track {track_name} not found in data, using default base lap time")
        return 90.0
    
    def _calculate_tire_effect(self, tire_age: int, tire_compound: str) -> float:
        """
        Calculate lap time penalty from tire degradation.
        
        Args:
            tire_age: Age of tires in laps
            tire_compound: Tire compound (SOFT, MEDIUM, HARD)
            
        Returns:
            Time penalty in seconds
        """
        # Get degradation rate for compound
        compound_key = tire_compound.value if hasattr(tire_compound, 'value') else str(tire_compound)
        degradation_rate = self.tire_compounds.get(compound_key, {}).get('degradation_rate', 0.05)
        
        # Linear degradation model
        tire_effect = tire_age * degradation_rate
        
        return tire_effect
    
    def _calculate_fuel_effect(self, fuel_load: float) -> float:
        """
        Calculate lap time penalty from fuel weight.
        
        Args:
            fuel_load: Fuel load in kg
            
        Returns:
            Time penalty in seconds
        """
        # ~0.03s per kg of fuel
        fuel_effect_rate = 0.03
        return fuel_load * fuel_effect_rate
    
    def _calculate_traffic_penalty(
        self,
        traffic_state: RaceCondition,
        gap_to_ahead: Optional[float]
    ) -> float:
        """
        Calculate lap time penalty from traffic.
        
        Args:
            traffic_state: Race condition (clean/dirty air)
            gap_to_ahead: Gap to car ahead in seconds
            
        Returns:
            Time penalty in seconds
        """
        if traffic_state != RaceCondition.DIRTY_AIR:
            return 0.0
        
        # Exponential decay based on gap
        if gap_to_ahead is None:
            gap_to_ahead = 0.5  # Assume close if not specified
        
        # Maximum penalty at 0s gap, decays exponentially
        max_penalty = 0.8  # seconds
        decay_rate = 2.0  # decay factor
        
        penalty = max_penalty * (2.0 ** (-gap_to_ahead / decay_rate))
        
        return penalty
    
    def _calculate_weather_adjustment(self, weather_temp: Optional[float]) -> float:
        """
        Calculate lap time adjustment from weather.
        
        Args:
            weather_temp: Ambient temperature in Celsius
            
        Returns:
            Time adjustment in seconds (positive = slower)
        """
        if weather_temp is None:
            return 0.0
        
        # Optimal temperature around 25°C
        optimal_temp = 25.0
        temp_effect_rate = 0.02  # 0.02s per degree deviation
        
        adjustment = (weather_temp - optimal_temp) * temp_effect_rate
        
        return adjustment
    
    def _calculate_safety_car_factor(self, safety_car_active: bool) -> float:
        """
        Calculate lap time multiplier for safety car.
        
        Args:
            safety_car_active: Whether safety car is deployed
            
        Returns:
            Multiplier (1.0 = normal, 1.3 = safety car)
        """
        if safety_car_active:
            return 1.3  # 30% slower under safety car
        return 1.0
    
    def _load_track_data(self) -> Dict:
        """
        Load track data from configuration.
        
        Returns:
            Dictionary of track characteristics
        """
        track_file = self.config_path / "tracks.yaml"
        
        if not track_file.exists():
            logger.warning(f"Track data file not found: {track_file}")
            return self._get_default_track_data()
        
        try:
            with open(track_file, 'r') as f:
                data = yaml.safe_load(f)
            return data.get('tracks', {})
        except Exception as e:
            logger.error(f"Failed to load track data: {e}")
            return self._get_default_track_data()
    
    def _load_tire_compounds(self) -> Dict:
        """
        Load tire compound characteristics from configuration.
        
        Returns:
            Dictionary of tire compound properties
        """
        tire_file = self.config_path / "tire_compounds.yaml"
        
        if not tire_file.exists():
            logger.warning(f"Tire data file not found: {tire_file}")
            return self._get_default_tire_data()
        
        try:
            with open(tire_file, 'r') as f:
                data = yaml.safe_load(f)
            return data.get('compounds', {})
        except Exception as e:
            logger.error(f"Failed to load tire data: {e}")
            return self._get_default_tire_data()
    
    def _get_default_track_data(self) -> Dict:
        """Get default track data if config unavailable."""
        return {
            'default': {
                'base_lap_time': 90.0,
                'length_km': 5.0,
            }
        }
    
    def _get_default_tire_data(self) -> Dict:
        """Get default tire compound data if config unavailable."""
        return {
            'SOFT': {
                'degradation_rate': 0.08,
                'grip_level': 1.0,
            },
            'MEDIUM': {
                'degradation_rate': 0.05,
                'grip_level': 0.95,
            },
            'HARD': {
                'degradation_rate': 0.03,
                'grip_level': 0.90,
            },
        }
