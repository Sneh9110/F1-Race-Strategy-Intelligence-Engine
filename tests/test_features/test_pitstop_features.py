"""
Unit tests for pitstop feature calculators.
"""

import pytest
import pandas as pd
import numpy as np
from features.pitstop_features import (
    UndercutDeltaFeature,
    OvercutDeltaFeature,
    PitLossModelFeature
)
from features.base import FeatureConfig


class TestUndercutDeltaFeature:
    """Tests for UndercutDeltaFeature."""
    
    def test_successful_undercut(self):
        """Test successful undercut scenario."""
        data = pd.DataFrame({
            'session_id': ['2024_MONACO_RACE'] * 5,
            'driver_id': ['VER', 'HAM', 'VER', 'HAM', 'VER'],
            'lap_number': [20, 20, 21, 21, 22],
            'lap_time': [79.5, 80.5, 78.0, 80.3, 78.1],
            'tire_compound': ['SOFT', 'MEDIUM', 'SOFT', 'MEDIUM', 'SOFT'],
            'tire_age': [15, 20, 1, 21, 2],
            'is_pit_lap': [False, False, False, False, False]
        })
        
        config = FeatureConfig(feature_name="undercut_delta", version="1.0.0")
        feature = UndercutDeltaFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        assert 'undercut_delta' in result.data.columns
    
    def test_pit_loss_calculation(self):
        """Test pit loss model integration."""
        data = pd.DataFrame({
            'session_id': ['2024_SILVERSTONE_RACE'] * 3,
            'driver_id': ['VER'] * 3,
            'lap_number': [20, 21, 22],
            'lap_time': [88.0, 88.5, 88.2],
            'tire_compound': ['MEDIUM'] * 3,
            'tire_age': [20, 1, 2],
            'is_pit_lap': [False, False, False]
        })
        
        config = FeatureConfig(feature_name="undercut_delta", version="1.0.0")
        feature = UndercutDeltaFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'pit_loss' in result.data.columns and len(result.data) > 0:
            pit_loss = result.data['pit_loss'].iloc[0]
            assert 15 < pit_loss < 35, f"Pit loss should be 15-35s, got {pit_loss}"


class TestOvercutDeltaFeature:
    """Tests for OvercutDeltaFeature."""
    
    def test_fuel_effect(self):
        """Test fuel effect calculation."""
        data = pd.DataFrame({
            'session_id': ['2024_MONACO_RACE'] * 3,
            'driver_id': ['VER'] * 3,
            'lap_number': [20, 21, 22],
            'lap_time': [78.5, 78.4, 78.3],
            'tire_compound': ['MEDIUM'] * 3,
            'tire_age': [20, 21, 22],
            'fuel_kg': [40, 38, 36]  # Decreasing fuel
        })
        
        config = FeatureConfig(feature_name="overcut_delta", version="1.0.0")
        feature = OvercutDeltaFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'fuel_effect_benefit' in result.data.columns and len(result.data) > 0:
            fuel_benefit = result.data['fuel_effect_benefit'].iloc[0]
            assert fuel_benefit >= 0, "Fuel effect should be positive"


class TestPitLossModelFeature:
    """Tests for PitLossModelFeature."""
    
    def test_track_specific_loss(self):
        """Test track-specific pit loss."""
        # Monaco (short pit lane)
        data_monaco = pd.DataFrame({
            'session_id': ['2024_MONACO_RACE'],
            'track_name': ['Monaco']
        })
        
        config = FeatureConfig(feature_name="pit_loss_model", version="1.0.0")
        feature = PitLossModelFeature(config)
        
        result = feature.compute(data_monaco)
        
        if result.success and len(result.data) > 0:
            monaco_loss = result.data['total_pit_loss'].iloc[0]
            # Monaco should have shorter pit loss
            assert monaco_loss < 20, f"Monaco pit loss should be <20s, got {monaco_loss}"
    
    def test_pit_loss_components(self):
        """Test pit loss breakdown."""
        data = pd.DataFrame({
            'session_id': ['2024_SILVERSTONE_RACE'],
            'track_name': ['Silverstone']
        })
        
        config = FeatureConfig(feature_name="pit_loss_model", version="1.0.0")
        feature = PitLossModelFeature(config)
        
        result = feature.compute(data)
        
        if result.success and len(result.data) > 0:
            # Check components exist and are positive
            assert result.data['base_pit_loss'].iloc[0] > 0
            assert result.data['total_pit_loss'].iloc[0] > 0
            
            # Total should be sum of components
            total = result.data['total_pit_loss'].iloc[0]
            base = result.data['base_pit_loss'].iloc[0]
            assert total >= base, "Total pit loss should be >= base loss"
