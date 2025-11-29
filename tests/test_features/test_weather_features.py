"""
Unit tests for weather feature calculators.
"""

import pytest
import pandas as pd
import numpy as np
from features.weather_features import (
    WeatherAdjustedPaceFeature,
    TrackEvolutionFeature,
    WeatherTrendFeature
)
from features.base import FeatureConfig


class TestWeatherAdjustedPaceFeature:
    """Tests for WeatherAdjustedPaceFeature."""
    
    def test_temperature_correction(self):
        """Test temperature-based pace correction."""
        data = pd.DataFrame({
            'lap_number': [1, 2, 3],
            'lap_time': [78.5, 78.7, 78.9],
            'air_temp': [25, 30, 35],  # Increasing temperature
            'track_temp': [40, 45, 50],
            'rainfall': [0, 0, 0],
            'wind_speed': [10, 10, 10]
        })
        
        config = FeatureConfig(feature_name="weather_adjusted_pace", version="1.0.0")
        feature = WeatherAdjustedPaceFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        assert 'adjusted_pace' in result.data.columns
        assert 'temp_correction' in result.data.columns
        
        # Higher temp should have correction
        temp_corrections = result.data['temp_correction'].values
        assert temp_corrections[2] > temp_corrections[0], \
            "Higher temperature should have larger correction"
    
    def test_rain_correction(self):
        """Test rain-based pace correction."""
        data = pd.DataFrame({
            'lap_number': [1, 2, 3],
            'lap_time': [78.5, 82.0, 85.5],
            'air_temp': [25, 25, 25],
            'track_temp': [40, 40, 40],
            'rainfall': [0, 2, 5],  # Increasing rain
            'wind_speed': [10, 10, 10]
        })
        
        config = FeatureConfig(feature_name="weather_adjusted_pace", version="1.0.0")
        feature = WeatherAdjustedPaceFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'rain_correction' in result.data.columns:
            rain_corrections = result.data['rain_correction'].values
            assert rain_corrections[2] > rain_corrections[0], \
                "More rain should have larger correction"


class TestTrackEvolutionFeature:
    """Tests for TrackEvolutionFeature."""
    
    def test_track_improvement(self):
        """Test track evolution/rubber buildup."""
        data = pd.DataFrame({
            'session_id': ['2024_MONACO_RACE'] * 10,
            'lap_number': range(1, 11),
            'session_elapsed_time': range(0, 600, 60),  # 60s per lap
            'lap_time': [78.5 - 0.05*i for i in range(10)]  # Track improving
        })
        
        config = FeatureConfig(feature_name="track_evolution", version="1.0.0")
        feature = TrackEvolutionFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'track_evolution_factor' in result.data.columns:
            evolution_factors = result.data['track_evolution_factor'].values
            # Evolution factor should decrease over time (track gets faster)
            assert evolution_factors[-1] < evolution_factors[0], \
                "Track evolution factor should decrease"
            # But should be close to 1.0
            assert all(0.95 < f <= 1.0 for f in evolution_factors), \
                "Evolution factors should be between 0.95 and 1.0"


class TestWeatherTrendFeature:
    """Tests for WeatherTrendFeature."""
    
    def test_temperature_trend(self):
        """Test temperature trend detection."""
        data = pd.DataFrame({
            'lap_number': range(1, 21),
            'timestamp': pd.date_range('2024-06-01 14:00', periods=20, freq='1min'),
            'air_temp': [25 + 0.5*i for i in range(20)],  # Increasing temp
            'track_temp': [40 + i for i in range(20)],
            'rainfall': [0] * 20,
            'wind_speed': [10] * 20
        })
        
        config = FeatureConfig(feature_name="weather_trend", version="1.0.0")
        feature = WeatherTrendFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'temp_trend' in result.data.columns and len(result.data) > 0:
            temp_trend = result.data['temp_trend'].iloc[-1]
            assert temp_trend > 0, "Temperature trend should be positive (increasing)"
    
    def test_rain_probability(self):
        """Test rain probability calculation."""
        data = pd.DataFrame({
            'lap_number': range(1, 11),
            'timestamp': pd.date_range('2024-06-01 14:00', periods=10, freq='1min'),
            'air_temp': [25] * 10,
            'track_temp': [40] * 10,
            'rainfall': [0, 0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5],
            'wind_speed': [10] * 10
        })
        
        config = FeatureConfig(feature_name="weather_trend", version="1.0.0")
        feature = WeatherTrendFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'rain_probability' in result.data.columns and len(result.data) > 0:
            rain_prob = result.data['rain_probability'].iloc[-1]
            assert 0 <= rain_prob <= 1, "Rain probability should be between 0 and 1"
            assert rain_prob > 0.5, "Rain probability should be high with increasing rainfall"
