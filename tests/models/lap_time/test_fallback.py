"""
Tests for fallback heuristics.
"""

import pytest

from models.lap_time.fallback import FallbackHeuristics
from models.lap_time.base import RaceCondition


class TestFallbackHeuristics:
    """Test physics-based fallback predictions."""
    
    def test_fallback_initialization(self):
        """Test fallback can be initialized."""
        fallback = FallbackHeuristics()
        assert fallback is not None
    
    def test_clean_air_prediction(self, sample_prediction_input):
        """Test prediction in clean air."""
        fallback = FallbackHeuristics()
        sample_prediction_input.traffic_state = RaceCondition.CLEAN_AIR
        
        output = fallback.predict(sample_prediction_input)
        
        assert output.predicted_lap_time > 60.0
        assert output.predicted_lap_time < 150.0
        assert output.confidence == 0.6  # Lower confidence for fallback
        assert 'fallback_heuristics' in output.metadata['model_type']
    
    def test_dirty_air_penalty(self, sample_prediction_input):
        """Test traffic penalty is applied."""
        fallback = FallbackHeuristics()
        
        # Clean air
        sample_prediction_input.traffic_state = RaceCondition.CLEAN_AIR
        clean_output = fallback.predict(sample_prediction_input)
        
        # Dirty air
        sample_prediction_input.traffic_state = RaceCondition.DIRTY_AIR
        sample_prediction_input.gap_to_ahead = 0.5
        dirty_output = fallback.predict(sample_prediction_input)
        
        # Dirty air should be slower
        assert dirty_output.predicted_lap_time > clean_output.predicted_lap_time
    
    def test_safety_car_effect(self, sample_prediction_input):
        """Test safety car slows lap time."""
        fallback = FallbackHeuristics()
        
        # Normal conditions
        sample_prediction_input.safety_car_active = False
        normal_output = fallback.predict(sample_prediction_input)
        
        # Safety car
        sample_prediction_input.safety_car_active = True
        sc_output = fallback.predict(sample_prediction_input)
        
        # Safety car should be significantly slower
        assert sc_output.predicted_lap_time > normal_output.predicted_lap_time * 1.2
    
    def test_tire_degradation(self, sample_prediction_input):
        """Test tire degradation increases lap time."""
        fallback = FallbackHeuristics()
        
        # Fresh tires
        sample_prediction_input.tire_age = 0
        fresh_output = fallback.predict(sample_prediction_input)
        
        # Worn tires
        sample_prediction_input.tire_age = 30
        worn_output = fallback.predict(sample_prediction_input)
        
        # Worn tires should be slower
        assert worn_output.predicted_lap_time > fresh_output.predicted_lap_time
