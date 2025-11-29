"""
Weather-adjusted feature calculators.

Normalizes lap times for weather conditions and analyzes track evolution.
"""

from typing import List, Dict, Any
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from features.base import BaseFeature, FeatureConfig
from features.registry import register_feature
from data_pipeline.schemas.weather_schema import WeatherData, WeatherForecast
from config.settings import settings


def load_track_config(track_name: str) -> Dict[str, Any]:
    """Load track-specific configuration."""
    config_path = Path(settings.CONFIG_DIR) / "tracks.yaml"
    with open(config_path, 'r') as f:
        tracks = yaml.safe_load(f)
    return tracks.get(track_name, {})


def load_tire_config() -> Dict[str, Any]:
    """Load tire compound configuration."""
    config_path = Path(settings.CONFIG_DIR) / "tire_compounds.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@register_feature(
    name='weather_adjusted_pace',
    version='v1.0.0',
    description='Normalize lap times for weather conditions',
    dependencies=[],
    computation_cost_ms=50,
    tags=['weather', 'pace', 'normalization']
)
class WeatherAdjustedPaceFeature(BaseFeature):
    """
    Normalizes lap times accounting for weather effects.
    
    Correction factors:
    - track_temp_factor = 1.0 + (track_temp - optimal_temp) * 0.002
    - rainfall_factor = 1.0 + rainfall_mm * 0.05
    - wind_factor = 1.0 + (wind_speed / 50.0) * 0.01
    
    Output columns:
    - lap_number: Lap number
    - driver_number: Driver number
    - raw_lap_time: Original lap time
    - track_temp: Track temperature (°C)
    - rainfall: Rainfall amount (mm)
    - wind_speed: Wind speed (km/h)
    - temp_correction_factor: Temperature correction
    - rain_correction_factor: Rain correction
    - wind_correction_factor: Wind correction
    - adjusted_lap_time: Weather-normalized lap time
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='weather_adjusted_pace',
            version='v1.0.0',
            config=config or FeatureConfig(thresholds={
                'optimal_temp': 45.0,
                'temp_factor': 0.002,
                'rain_factor': 0.05,
                'wind_factor': 0.01
            })
        )
    
    def _calculate(
        self,
        lap_data: pd.DataFrame,
        weather_data: List[WeatherData],
        **kwargs
    ) -> pd.DataFrame:
        """Adjust lap times for weather."""
        # Convert weather to DataFrame
        weather_df = pd.DataFrame([w.dict() for w in weather_data])
        
        results = []
        optimal_temp = self.config.thresholds.get('optimal_temp', 45.0)
        temp_factor = self.config.thresholds.get('temp_factor', 0.002)
        rain_factor = self.config.thresholds.get('rain_factor', 0.05)
        wind_factor = self.config.thresholds.get('wind_factor', 0.01)
        
        for _, lap in lap_data.iterrows():
            # Find closest weather reading
            lap_time_weather = weather_df.iloc[
                (weather_df['timestamp'] - lap['timestamp']).abs().argmin()
            ] if not weather_df.empty else None
            
            if lap_time_weather is not None:
                track_temp = lap_time_weather['track_temp_celsius']
                rainfall = lap_time_weather.get('rainfall_mm', 0.0)
                wind_speed = lap_time_weather.get('wind_speed_kph', 0.0)
            else:
                # Default values
                track_temp = optimal_temp
                rainfall = 0.0
                wind_speed = 0.0
            
            # Calculate correction factors
            temp_correction = 1.0 + (track_temp - optimal_temp) * temp_factor
            rain_correction = 1.0 + rainfall * rain_factor
            wind_correction = 1.0 + (wind_speed / 50.0) * wind_factor
            
            # Combined correction
            total_correction = temp_correction * rain_correction * wind_correction
            
            # Adjusted lap time (divide by correction to normalize)
            adjusted_lap_time = lap['lap_time'] / total_correction
            
            results.append({
                'lap_number': lap['lap_number'],
                'driver_number': lap['driver_number'],
                'raw_lap_time': lap['lap_time'],
                'track_temp': float(track_temp),
                'rainfall': float(rainfall),
                'wind_speed': float(wind_speed),
                'temp_correction_factor': float(temp_correction),
                'rain_correction_factor': float(rain_correction),
                'wind_correction_factor': float(wind_correction),
                'total_correction_factor': float(total_correction),
                'adjusted_lap_time': float(adjusted_lap_time)
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='track_evolution',
    version='v1.0.0',
    description='Model track grip improvement over session',
    dependencies=[],
    computation_cost_ms=30,
    tags=['weather', 'track', 'evolution']
)
class TrackEvolutionFeature(BaseFeature):
    """
    Models track grip improvement as rubber is laid down.
    
    Formula:
        evolution_factor = 1.0 - (session_progress * track_evolution_factor)
        adjusted_pace = lap_time / evolution_factor
    
    Output columns:
    - lap_number: Lap number
    - session_progress: Progress through session (0.0 to 1.0)
    - evolution_factor: Track evolution factor
    - raw_lap_time: Original lap time
    - evolution_adjusted_time: Evolution-normalized lap time
    """
    
    def __init__(self, config: FeatureConfig = None, track_name: str = "generic"):
        super().__init__(
            name='track_evolution',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
        self.track_config = load_track_config(track_name)
    
    def _calculate(self, data: pd.DataFrame, total_laps: int = 50, **kwargs) -> pd.DataFrame:
        """Model track evolution."""
        track_evolution_factor = self.track_config.get('track_evolution_factor', 0.01)
        
        results = []
        for _, lap in data.iterrows():
            lap_num = lap['lap_number']
            session_progress = lap_num / total_laps
            
            # Evolution factor (grip improves over time)
            evolution_factor = 1.0 - (session_progress * track_evolution_factor)
            
            # Adjusted lap time
            adjusted_time = lap['lap_time'] / evolution_factor
            
            results.append({
                'lap_number': lap_num,
                'driver_number': lap['driver_number'],
                'session_progress': float(session_progress),
                'evolution_factor': float(evolution_factor),
                'raw_lap_time': lap['lap_time'],
                'evolution_adjusted_time': float(adjusted_time)
            })
        
        return pd.DataFrame(results)
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='weather_trend',
    version='v1.0.0',
    description='Predict weather condition changes',
    dependencies=[],
    computation_cost_ms=40,
    tags=['weather', 'forecast', 'trend']
)
class WeatherTrendFeature(BaseFeature):
    """
    Predicts weather trends from historical data and forecasts.
    
    Output columns:
    - current_temp: Current track temperature
    - temp_trend: Temperature change rate (°C/hour)
    - rain_likelihood: Probability of rain (0-1)
    - condition_change_probability: Probability of condition change
    - forecast_horizon_minutes: Forecast time horizon
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='weather_trend',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
    
    def _calculate(
        self,
        weather_data: List[WeatherData],
        forecast_data: List[WeatherForecast] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Calculate weather trends."""
        weather_df = pd.DataFrame([w.dict() for w in weather_data])
        
        if weather_df.empty:
            return pd.DataFrame()
        
        # Sort by timestamp
        weather_df = weather_df.sort_values('timestamp')
        
        # Temperature trend (linear regression)
        if len(weather_df) >= 2:
            temps = weather_df['track_temp_celsius'].values
            times = np.arange(len(temps))
            temp_trend = np.polyfit(times, temps, 1)[0]  # degrees per reading
        else:
            temp_trend = 0.0
        
        # Rain likelihood from forecast
        if forecast_data:
            rain_probs = [f.rain_probability_percent for f in forecast_data]
            rain_likelihood = np.mean(rain_probs) / 100.0
        else:
            rain_likelihood = 0.0
        
        # Current conditions
        current = weather_df.iloc[-1]
        
        # Condition change probability (simplified)
        condition_change_prob = min(rain_likelihood * 0.5, 0.9)
        
        return pd.DataFrame([{
            'current_temp': float(current['track_temp_celsius']),
            'air_temp': float(current['air_temp_celsius']),
            'humidity': float(current['humidity_percent']),
            'temp_trend': float(temp_trend),
            'rain_likelihood': float(rain_likelihood),
            'condition_change_probability': float(condition_change_prob),
            'forecast_horizon_minutes': 30
        }])
    
    def _get_dependencies(self) -> List[str]:
        return []


@register_feature(
    name='compound_weather_suitability',
    version='v1.0.0',
    description='Match tire compounds to weather conditions',
    dependencies=[],
    computation_cost_ms=30,
    tags=['weather', 'tire', 'suitability']
)
class CompoundWeatherSuitabilityFeature(BaseFeature):
    """
    Determines optimal tire compound for current weather.
    
    Output columns:
    - track_temp: Current track temperature
    - optimal_compound: Best compound for conditions
    - soft_suitability_score: Suitability score for soft (0-1)
    - medium_suitability_score: Suitability score for medium (0-1)
    - hard_suitability_score: Suitability score for hard (0-1)
    - compound_risk_score: Risk of using non-optimal compound
    """
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(
            name='compound_weather_suitability',
            version='v1.0.0',
            config=config or FeatureConfig()
        )
        self.tire_config = load_tire_config()
    
    def _calculate(self, weather_data: WeatherData, **kwargs) -> pd.DataFrame:
        """Calculate compound suitability."""
        track_temp = weather_data.track_temp_celsius
        
        compounds = self.tire_config.get('compounds', {})
        
        # Calculate suitability scores based on temperature range
        scores = {}
        for compound_name, compound_data in compounds.items():
            optimal_range = compound_data.get('optimal_temp_range_celsius', [30, 50])
            temp_min, temp_max = optimal_range
            
            # Distance from optimal range
            if track_temp < temp_min:
                distance = temp_min - track_temp
            elif track_temp > temp_max:
                distance = track_temp - temp_max
            else:
                distance = 0.0
            
            # Suitability score (1.0 = perfect, 0.0 = very unsuitable)
            suitability = max(0.0, 1.0 - (distance / 20.0))
            scores[compound_name] = suitability
        
        # Find optimal compound
        optimal_compound = max(scores, key=scores.get)
        
        # Risk score (how far from optimal)
        max_score = scores[optimal_compound]
        avg_score = np.mean(list(scores.values()))
        risk_score = 1.0 - (avg_score / max_score) if max_score > 0 else 1.0
        
        return pd.DataFrame([{
            'track_temp': float(track_temp),
            'air_temp': float(weather_data.air_temp_celsius),
            'optimal_compound': optimal_compound,
            'soft_suitability_score': float(scores.get('SOFT', 0.0)),
            'medium_suitability_score': float(scores.get('MEDIUM', 0.0)),
            'hard_suitability_score': float(scores.get('HARD', 0.0)),
            'compound_risk_score': float(risk_score)
        }])
    
    def _get_dependencies(self) -> List[str]:
        return []
