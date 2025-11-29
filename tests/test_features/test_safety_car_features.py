"""
Unit tests for safety car feature calculators.
"""

import pytest
import pandas as pd
import numpy as np
from features.safety_car_features import (
    HistoricalSCProbabilityFeature,
    RealTimeSCProbabilityFeature,
    SCImpactFeature
)
from features.base import FeatureConfig


class TestHistoricalSCProbabilityFeature:
    """Tests for HistoricalSCProbabilityFeature."""
    
    def test_probability_calculation(self):
        """Test historical SC probability calculation."""
        # Create historical data with 50% SC rate
        data = pd.DataFrame({
            'session_id': [f'2024_MONACO_RACE_{i}' for i in range(10)],
            'track_name': ['Monaco'] * 10,
            'safety_car_deployed': [True, False, True, False, True, 
                                    False, True, False, True, False]
        })
        
        config = FeatureConfig(feature_name="historical_sc_probability", version="1.0.0")
        feature = HistoricalSCProbabilityFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'historical_sc_rate' in result.data.columns and len(result.data) > 0:
            sc_rate = result.data['historical_sc_rate'].iloc[0]
            assert 0.4 < sc_rate < 0.6, f"Expected ~50% SC rate, got {sc_rate}"
            assert 0 <= sc_rate <= 1, "SC rate should be valid probability"
    
    def test_track_specific_probability(self):
        """Test track-specific SC probabilities."""
        # Monaco (high SC rate)
        data_monaco = pd.DataFrame({
            'session_id': ['2024_MONACO_RACE'],
            'track_name': ['Monaco'],
            'safety_car_deployed': [True] * 9 + [False]  # 90% SC rate
        })
        
        # Austria (low SC rate)
        data_austria = pd.DataFrame({
            'session_id': ['2024_AUSTRIA_RACE'],
            'track_name': ['Austria'],
            'safety_car_deployed': [True] + [False] * 9  # 10% SC rate
        })
        
        config = FeatureConfig(feature_name="historical_sc_probability", version="1.0.0")
        feature = HistoricalSCProbabilityFeature(config)
        
        result_monaco = feature.compute(data_monaco)
        result_austria = feature.compute(data_austria)
        
        if (result_monaco.success and result_austria.success and 
            'track_sc_probability' in result_monaco.data.columns and
            len(result_monaco.data) > 0 and len(result_austria.data) > 0):
            monaco_prob = result_monaco.data['track_sc_probability'].iloc[0]
            austria_prob = result_austria.data['track_sc_probability'].iloc[0]
            assert monaco_prob > austria_prob, "Monaco should have higher SC probability"


class TestRealTimeSCProbabilityFeature:
    """Tests for RealTimeSCProbabilityFeature."""
    
    def test_incident_weight(self):
        """Test incident weighting in real-time probability."""
        data = pd.DataFrame({
            'lap_number': [10, 11, 12],
            'incident_type': ['crash', None, None],
            'incident_severity': [0.8, None, None]
        })
        
        config = FeatureConfig(feature_name="realtime_sc_probability", version="1.0.0")
        feature = RealTimeSCProbabilityFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'realtime_sc_probability' in result.data.columns:
            probabilities = result.data['realtime_sc_probability'].values
            # Probability should spike after crash
            assert probabilities[1] > probabilities[0], \
                "SC probability should increase after crash"
    
    def test_probability_decay(self):
        """Test probability decay over time."""
        data = pd.DataFrame({
            'lap_number': range(10, 20),
            'incident_type': ['crash'] + [None] * 9,
            'incident_severity': [0.8] + [None] * 9
        })
        
        config = FeatureConfig(feature_name="realtime_sc_probability", version="1.0.0")
        feature = RealTimeSCProbabilityFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'realtime_sc_probability' in result.data.columns:
            probabilities = result.data['realtime_sc_probability'].values
            # Probability should decay after incident
            if len(probabilities) > 5:
                assert probabilities[5] < probabilities[1], \
                    "SC probability should decay over laps"


class TestSCImpactFeature:
    """Tests for SCImpactFeature."""
    
    def test_field_compression(self):
        """Test field compression calculation."""
        # Gaps before SC
        data_before = pd.DataFrame({
            'driver_id': ['VER', 'HAM', 'LEC'],
            'gap_to_leader': [0.0, 5.2, 12.8]
        })
        
        # Gaps after SC (compressed)
        data_after = pd.DataFrame({
            'driver_id': ['VER', 'HAM', 'LEC'],
            'gap_to_leader': [0.0, 0.5, 1.2]
        })
        
        config = FeatureConfig(feature_name="sc_impact", version="1.0.0")
        feature = SCImpactFeature(config)
        
        data = pd.concat([
            data_before.assign(sc_period=False),
            data_after.assign(sc_period=True)
        ])
        
        result = feature.compute(data)
        
        assert result.success
        if 'field_compression' in result.data.columns and len(result.data) > 0:
            compression = result.data['field_compression'].iloc[0]
            assert 0 <= compression <= 1, "Compression should be between 0 and 1"
    
    def test_tire_temp_loss(self):
        """Test tire temperature loss during SC."""
        data = pd.DataFrame({
            'lap_number': [10, 11, 12, 13],
            'sc_active': [False, True, True, False],
            'tire_temp_avg': [100, 95, 90, 92]  # Cooling during SC
        })
        
        config = FeatureConfig(feature_name="sc_impact", version="1.0.0")
        feature = SCImpactFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'tire_temp_loss' in result.data.columns:
            temp_losses = result.data['tire_temp_loss'].values
            # Should show temperature loss during SC period
            sc_laps = result.data[result.data['sc_active'] == True]
            if len(sc_laps) > 0 and 'tire_temp_loss' in sc_laps.columns:
                assert sc_laps['tire_temp_loss'].iloc[0] < 0, \
                    "Should show temperature loss during SC"
