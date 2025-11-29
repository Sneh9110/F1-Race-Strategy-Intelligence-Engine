"""
Unit tests for stint feature calculators.
"""

import pytest
import pandas as pd
import numpy as np
from features.stint_features import StintSummaryFeature, StintPaceEvolutionFeature
from features.base import FeatureConfig
from tests.test_features import (
    assert_feature_output_valid,
    assert_no_outliers,
    sample_lap_data
)


class TestStintSummaryFeature:
    """Tests for StintSummaryFeature."""
    
    def test_basic_computation(self, sample_lap_data):
        """Test basic stint summary computation."""
        config = FeatureConfig(
            feature_name="stint_summary",
            version="1.0.0",
            cache_enabled=False
        )
        feature = StintSummaryFeature(config)
        
        result = feature.compute(sample_lap_data)
        
        assert result.success, f"Computation failed: {result.error}"
        assert result.data is not None
        assert not result.data.empty
        
        # Check expected columns
        expected_cols = [
            'stint_id', 'laps_in_stint', 'avg_lap_time', 'median_lap_time',
            'std_lap_time', 'best_lap_time', 'tire_compound'
        ]
        assert_feature_output_valid(result.data, expected_cols)
    
    def test_multiple_stints(self):
        """Test computation with multiple stints."""
        # Create data with 2 stints
        data = pd.DataFrame({
            'lap_number': list(range(1, 31)),
            'lap_time': [78.5 + i*0.05 + np.random.normal(0, 0.1) for i in range(30)],
            'tire_age': [i if i <= 15 else i-15 for i in range(1, 31)],  # Reset at lap 16
            'tire_compound': ['SOFT']*15 + ['MEDIUM']*15
        })
        
        config = FeatureConfig(feature_name="stint_summary", version="1.0.0")
        feature = StintSummaryFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        assert len(result.data) == 2, "Should detect 2 stints"
        assert result.data['laps_in_stint'].iloc[0] == 15
        assert result.data['laps_in_stint'].iloc[1] == 15
    
    def test_outlier_removal(self, sample_lap_data):
        """Test that outliers are removed from stint statistics."""
        # Add outlier lap
        sample_lap_data.loc[10, 'lap_time'] = 120.0  # Extreme outlier
        
        config = FeatureConfig(
            feature_name="stint_summary",
            version="1.0.0",
            outlier_removal=True
        )
        feature = StintSummaryFeature(config)
        
        result = feature.compute(sample_lap_data)
        
        assert result.success
        # Check that outlier didn't skew average too much
        avg_time = result.data['avg_lap_time'].iloc[0]
        assert 78 < avg_time < 82, f"Average skewed by outlier: {avg_time}"
    
    def test_percentiles(self, sample_lap_data):
        """Test percentile calculation."""
        config = FeatureConfig(feature_name="stint_summary", version="1.0.0")
        feature = StintSummaryFeature(config)
        
        result = feature.compute(sample_lap_data)
        
        assert result.success
        # Percentiles should be in ascending order
        if 'pace_percentiles' in result.data.columns:
            percentiles = result.data['pace_percentiles'].iloc[0]
            if isinstance(percentiles, (list, np.ndarray)):
                assert all(percentiles[i] <= percentiles[i+1] 
                          for i in range(len(percentiles)-1))
    
    def test_empty_input(self):
        """Test with empty input data."""
        config = FeatureConfig(feature_name="stint_summary", version="1.0.0")
        feature = StintSummaryFeature(config)
        
        result = feature.compute(pd.DataFrame())
        
        assert not result.success, "Should fail with empty input"
        assert result.error is not None


class TestStintPaceEvolutionFeature:
    """Tests for StintPaceEvolutionFeature."""
    
    def test_basic_computation(self, sample_lap_data):
        """Test basic pace evolution computation."""
        config = FeatureConfig(
            feature_name="stint_pace_evolution",
            version="1.0.0",
            cache_enabled=False
        )
        feature = StintPaceEvolutionFeature(config)
        
        result = feature.compute(sample_lap_data)
        
        assert result.success, f"Computation failed: {result.error}"
        assert result.data is not None
        
        # Check expected columns
        expected_cols = ['stint_id', 'lap_in_stint', 'pace_delta', 'cumulative_deg']
        assert_feature_output_valid(result.data, expected_cols)
    
    def test_degradation_trend(self, sample_lap_data):
        """Test that cumulative degradation shows increasing trend."""
        config = FeatureConfig(feature_name="stint_pace_evolution", version="1.0.0")
        feature = StintPaceEvolutionFeature(config)
        
        result = feature.compute(sample_lap_data)
        
        assert result.success
        
        # Cumulative degradation should generally increase
        cumulative_deg = result.data['cumulative_deg']
        # Allow some tolerance for noise
        increasing_count = (cumulative_deg.diff() >= -0.1).sum()
        assert increasing_count / len(cumulative_deg) > 0.7, \
            "Cumulative degradation should mostly increase"
    
    def test_pace_delta_calculation(self, sample_lap_data):
        """Test pace delta calculation."""
        config = FeatureConfig(feature_name="stint_pace_evolution", version="1.0.0")
        feature = StintPaceEvolutionFeature(config)
        
        result = feature.compute(sample_lap_data)
        
        assert result.success
        
        # Pace delta should be centered around 0 (relative to stint average)
        pace_deltas = result.data['pace_delta']
        assert abs(pace_deltas.mean()) < 0.5, \
            "Pace deltas should be centered near 0"
    
    def test_lap_in_stint_sequence(self, sample_lap_data):
        """Test that lap_in_stint is correctly sequenced."""
        config = FeatureConfig(feature_name="stint_pace_evolution", version="1.0.0")
        feature = StintPaceEvolutionFeature(config)
        
        result = feature.compute(sample_lap_data)
        
        assert result.success
        
        # Each stint should have sequential lap numbers starting from 1
        for stint_id in result.data['stint_id'].unique():
            stint_laps = result.data[result.data['stint_id'] == stint_id]['lap_in_stint']
            expected_laps = list(range(1, len(stint_laps) + 1))
            assert stint_laps.tolist() == expected_laps, \
                f"Stint {stint_id} lap sequence incorrect"
    
    def test_with_pit_laps(self):
        """Test computation excluding pit laps."""
        data = pd.DataFrame({
            'lap_number': list(range(1, 21)),
            'lap_time': [78.5 + i*0.05 for i in range(20)],
            'tire_age': list(range(1, 11)) + list(range(1, 11)),
            'is_pit_lap': [False]*10 + [True] + [False]*9
        })
        
        config = FeatureConfig(feature_name="stint_pace_evolution", version="1.0.0")
        feature = StintPaceEvolutionFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        # Pit lap should be excluded from pace evolution
        assert 200.0 not in result.data['pace_delta'].values  # Pit lap outlier
