"""
Tests for base classes and data structures.
"""

import pytest
import numpy as np
from pydantic import ValidationError

from models.lap_time.base import (
    PredictionInput,
    PredictionOutput,
    RaceCondition,
    BaseLapTimeModel,
    ModelConfig
)


class TestPredictionInput:
    """Test PredictionInput validation."""
    
    def test_valid_input(self, sample_prediction_input):
        """Test that valid input passes validation."""
        assert sample_prediction_input.tire_age == 10
        assert sample_prediction_input.tire_compound == "SOFT"
        assert sample_prediction_input.fuel_load == 50.0
    
    def test_tire_age_bounds(self):
        """Test tire age must be within bounds."""
        with pytest.raises(ValidationError):
            PredictionInput(
                tire_age=-1,  # Invalid: negative
                tire_compound="SOFT",
                fuel_load=50.0,
                traffic_state=RaceCondition.CLEAN_AIR,
                safety_car_active=False,
                weather_temp=25.0,
                track_name="Monaco",
                driver_number=44,
                lap_number=1,
                session_progress=0.1,
                recent_lap_times=[88.5] * 5
            )
    
    def test_fuel_load_bounds(self):
        """Test fuel load must be within bounds."""
        with pytest.raises(ValidationError):
            PredictionInput(
                tire_age=10,
                tire_compound="SOFT",
                fuel_load=150.0,  # Invalid: > 110kg
                traffic_state=RaceCondition.CLEAN_AIR,
                safety_car_active=False,
                weather_temp=25.0,
                track_name="Monaco",
                driver_number=44,
                lap_number=1,
                session_progress=0.1,
                recent_lap_times=[88.5] * 5
            )
    
    def test_session_progress_bounds(self):
        """Test session progress must be 0-1."""
        with pytest.raises(ValidationError):
            PredictionInput(
                tire_age=10,
                tire_compound="SOFT",
                fuel_load=50.0,
                traffic_state=RaceCondition.CLEAN_AIR,
                safety_car_active=False,
                weather_temp=25.0,
                track_name="Monaco",
                driver_number=44,
                lap_number=1,
                session_progress=1.5,  # Invalid: > 1.0
                recent_lap_times=[88.5] * 5
            )


class TestPredictionOutput:
    """Test PredictionOutput validation."""
    
    def test_valid_output(self):
        """Test that valid output passes validation."""
        output = PredictionOutput(
            predicted_lap_time=88.5,
            confidence=0.95,
            pace_components={
                'base_pace': 85.0,
                'tire_effect': 0.5,
                'fuel_effect': 1.5,
                'traffic_penalty': 0.0,
                'weather_adjustment': 0.5,
                'safety_car_factor': 1.0,
            }
        )
        assert output.predicted_lap_time == 88.5
        assert output.confidence == 0.95
    
    def test_lap_time_bounds(self):
        """Test lap time must be realistic."""
        with pytest.raises(ValidationError):
            PredictionOutput(
                predicted_lap_time=30.0,  # Invalid: unrealistic
                confidence=0.95,
                pace_components={}
            )
    
    def test_confidence_bounds(self):
        """Test confidence must be 0-1."""
        with pytest.raises(ValidationError):
            PredictionOutput(
                predicted_lap_time=88.5,
                confidence=1.5,  # Invalid: > 1.0
                pace_components={}
            )


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_config_creation(self, model_config):
        """Test creating model configuration."""
        assert model_config.name == "lap_time_test"
        assert model_config.version == "1.0.0"
        assert model_config.model_type == "test"


class TestRaceCondition:
    """Test RaceCondition enum."""
    
    def test_enum_values(self):
        """Test all race conditions are defined."""
        assert RaceCondition.CLEAN_AIR.value == "clean_air"
        assert RaceCondition.DIRTY_AIR.value == "dirty_air"
        assert RaceCondition.SAFETY_CAR.value == "safety_car"
        assert RaceCondition.VIRTUAL_SAFETY_CAR.value == "virtual_safety_car"
        assert RaceCondition.NORMAL.value == "normal"
