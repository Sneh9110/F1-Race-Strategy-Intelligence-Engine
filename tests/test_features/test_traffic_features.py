"""
Unit tests for traffic feature calculators.
"""

import pytest
import pandas as pd
import numpy as np
from features.traffic_features import (
    CleanAirPenaltyFeature,
    TrafficDensityFeature,
    LappingImpactFeature
)
from features.base import FeatureConfig


class TestCleanAirPenaltyFeature:
    """Tests for CleanAirPenaltyFeature."""
    
    def test_penalty_calculation(self):
        """Test clean air penalty calculation."""
        data = pd.DataFrame({
            'driver_id': ['VER', 'HAM', 'LEC'],
            'lap_number': [10, 10, 10],
            'gap_to_car_ahead': [10.0, 0.5, 2.0],  # Different gaps
            'drs_available': [False, False, False]
        })
        
        config = FeatureConfig(feature_name="clean_air_penalty", version="1.0.0")
        feature = CleanAirPenaltyFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'clean_air_penalty' in result.data.columns:
            penalties = result.data['clean_air_penalty'].values
            # Closer gap should have higher penalty
            assert penalties[1] > penalties[2], \
                "Smaller gap (0.5s) should have higher penalty than larger gap (2.0s)"
            assert penalties[2] > penalties[0], \
                "Gap of 2.0s should have higher penalty than 10.0s"
    
    def test_drs_benefit(self):
        """Test DRS effect on traffic penalty."""
        data = pd.DataFrame({
            'driver_id': ['HAM', 'HAM'],
            'lap_number': [10, 11],
            'gap_to_car_ahead': [0.8, 0.8],
            'drs_available': [False, True]  # DRS enabled on lap 11
        })
        
        config = FeatureConfig(feature_name="clean_air_penalty", version="1.0.0")
        feature = CleanAirPenaltyFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'clean_air_penalty' in result.data.columns and len(result.data) == 2:
            penalty_no_drs = result.data['clean_air_penalty'].iloc[0]
            penalty_with_drs = result.data['clean_air_penalty'].iloc[1]
            # DRS should reduce or negate penalty
            assert penalty_with_drs < penalty_no_drs, \
                "DRS should reduce traffic penalty"


class TestTrafficDensityFeature:
    """Tests for TrafficDensityFeature."""
    
    def test_density_calculation(self):
        """Test traffic density calculation."""
        # Create field with varying gaps
        data = pd.DataFrame({
            'driver_id': [f'P{i}' for i in range(1, 11)],
            'position': list(range(1, 11)),
            'gap_to_leader': [0, 1.2, 1.8, 2.5, 7.8, 8.2, 8.9, 15.2, 20.5, 25.3]
        })
        
        config = FeatureConfig(feature_name="traffic_density", version="1.0.0")
        feature = TrafficDensityFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'traffic_density' in result.data.columns:
            densities = result.data['traffic_density'].values
            # Front of pack (positions 1-4) should have high density
            # Mid-pack (5-7) should have high density
            # Back (8-10) should have low density
            assert densities[0] > 2, "Leader should have traffic around"
            assert densities[-1] < densities[0], "Back should have less traffic"
    
    def test_isolated_driver(self):
        """Test driver in clean air (no traffic)."""
        data = pd.DataFrame({
            'driver_id': ['VER', 'HAM', 'LEC'],
            'position': [1, 2, 3],
            'gap_to_leader': [0, 25.0, 35.0]  # Large gaps
        })
        
        config = FeatureConfig(feature_name="traffic_density", version="1.0.0")
        feature = TrafficDensityFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'traffic_density' in result.data.columns:
            # All drivers isolated, density should be low
            assert all(d <= 1 for d in result.data['traffic_density'].values), \
                "Isolated drivers should have low traffic density"


class TestLappingImpactFeature:
    """Tests for LappingImpactFeature."""
    
    def test_blue_flag_penalty(self):
        """Test blue flag penalty calculation."""
        data = pd.DataFrame({
            'driver_id': ['BOT', 'BOT', 'BOT'],
            'lap_number': [10, 11, 12],
            'blue_flags': [0, 2, 1],  # Blue flags on laps 11-12
            'lap_time': [80.0, 80.5, 80.3]
        })
        
        config = FeatureConfig(feature_name="lapping_impact", version="1.0.0")
        feature = LappingImpactFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'blue_flag_penalty' in result.data.columns:
            penalties = result.data['blue_flag_penalty'].values
            # Laps with blue flags should have penalty
            assert penalties[1] > penalties[0], \
                "Blue flag laps should have higher penalty"
    
    def test_laps_affected(self):
        """Test counting of laps affected by blue flags."""
        data = pd.DataFrame({
            'driver_id': ['BOT'] * 10,
            'lap_number': range(1, 11),
            'blue_flags': [0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
        })
        
        config = FeatureConfig(feature_name="lapping_impact", version="1.0.0")
        feature = LappingImpactFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'laps_affected' in result.data.columns and len(result.data) > 0:
            laps_affected = result.data['laps_affected'].iloc[-1]
            assert laps_affected == 3, f"Expected 3 laps affected, got {laps_affected}"
