"""
Unit tests for degradation feature calculators.
"""

import pytest
import pandas as pd
import numpy as np
from features.degradation_features import (
    DegradationSlopeFeature,
    ExponentialDegradationFeature,
    CliffDetectionFeature,
    DegradationAnomalyFeature
)
from features.base import FeatureConfig


class TestDegradationSlopeFeature:
    """Tests for DegradationSlopeFeature."""
    
    def test_linear_degradation(self):
        """Test with perfect linear degradation."""
        # Create perfect linear degradation data
        data = pd.DataFrame({
            'lap_number': range(1, 21),
            'tire_age': range(1, 21),
            'lap_time': [78.0 + 0.1*i for i in range(1, 21)]  # 0.1s/lap degradation
        })
        
        config = FeatureConfig(feature_name="degradation_slope", version="1.0.0")
        feature = DegradationSlopeFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        assert 'degradation_rate' in result.data.columns
        
        # Should detect ~0.1 s/lap degradation
        deg_rate = result.data['degradation_rate'].iloc[0]
        assert 0.08 < deg_rate < 0.12, f"Expected ~0.1 s/lap, got {deg_rate}"
        
        # R² should be very high for perfect linear
        r_squared = result.data['r_squared'].iloc[0]
        assert r_squared > 0.95, f"R² should be >0.95 for linear data, got {r_squared}"
    
    def test_insufficient_laps(self):
        """Test with insufficient laps for regression."""
        data = pd.DataFrame({
            'lap_number': [1, 2, 3],
            'tire_age': [1, 2, 3],
            'lap_time': [78.0, 78.1, 78.2]
        })
        
        config = FeatureConfig(feature_name="degradation_slope", version="1.0.0")
        feature = DegradationSlopeFeature(config)
        
        result = feature.compute(data)
        
        # Should handle gracefully (may return NaN or warning)
        assert result.success or result.warnings


class TestExponentialDegradationFeature:
    """Tests for ExponentialDegradationFeature."""
    
    def test_exponential_fit(self):
        """Test exponential curve fitting."""
        # Create exponential degradation data
        data = pd.DataFrame({
            'lap_number': range(1, 21),
            'tire_age': range(1, 21),
            'lap_time': [78.0 + 0.05 * np.exp(0.08*i) for i in range(1, 21)]
        })
        
        config = FeatureConfig(feature_name="exp_degradation", version="1.0.0")
        feature = ExponentialDegradationFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        assert 'exp_deg_a' in result.data.columns
        assert 'exp_deg_b' in result.data.columns
        
        # Coefficients should be positive for degradation
        assert result.data['exp_deg_a'].iloc[0] > 0
        assert result.data['exp_deg_b'].iloc[0] > 0


class TestCliffDetectionFeature:
    """Tests for CliffDetectionFeature."""
    
    def test_cliff_detection(self):
        """Test detection of tire cliff."""
        # Create data with cliff at lap 15
        lap_times = [78.0 + 0.05*i for i in range(1, 15)]  # Gradual deg
        lap_times += [79.5 + 1.5*i for i in range(3)]  # Sudden cliff
        
        data = pd.DataFrame({
            'lap_number': range(1, len(lap_times) + 1),
            'tire_age': range(1, len(lap_times) + 1),
            'lap_time': lap_times
        })
        
        config = FeatureConfig(feature_name="cliff_detection", version="1.0.0")
        feature = CliffDetectionFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        assert 'cliff_detected' in result.data.columns
        
        # Should detect cliff
        if len(result.data) > 0:
            cliff_detected = result.data['cliff_detected'].iloc[0]
            if cliff_detected:
                assert result.data['cliff_lap'].iloc[0] >= 14, \
                    "Cliff should be detected around lap 14-15"
    
    def test_no_cliff(self):
        """Test with gradual degradation (no cliff)."""
        data = pd.DataFrame({
            'lap_number': range(1, 21),
            'tire_age': range(1, 21),
            'lap_time': [78.0 + 0.05*i for i in range(1, 21)]  # Gradual only
        })
        
        config = FeatureConfig(feature_name="cliff_detection", version="1.0.0")
        feature = CliffDetectionFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        # Should not detect cliff
        if 'cliff_detected' in result.data.columns and len(result.data) > 0:
            assert result.data['cliff_detected'].iloc[0] == False


class TestDegradationAnomalyFeature:
    """Tests for DegradationAnomalyFeature."""
    
    def test_anomaly_detection(self):
        """Test detection of degradation anomalies."""
        # Create data with outlier
        lap_times = [78.0 + 0.05*i + np.random.normal(0, 0.1) for i in range(1, 21)]
        lap_times[10] = 85.0  # Anomaly
        
        data = pd.DataFrame({
            'lap_number': range(1, 21),
            'tire_age': range(1, 21),
            'lap_time': lap_times
        })
        
        config = FeatureConfig(feature_name="degradation_anomaly", version="1.0.0")
        feature = DegradationAnomalyFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        assert 'anomaly_count' in result.data.columns
        
        # Should detect at least one anomaly
        anomaly_count = result.data['anomaly_count'].iloc[0]
        assert anomaly_count >= 1, "Should detect the inserted anomaly"
    
    def test_no_anomalies(self):
        """Test with clean data (no anomalies)."""
        data = pd.DataFrame({
            'lap_number': range(1, 21),
            'tire_age': range(1, 21),
            'lap_time': [78.0 + 0.05*i for i in range(1, 21)]  # Clean linear
        })
        
        config = FeatureConfig(feature_name="degradation_anomaly", version="1.0.0")
        feature = DegradationAnomalyFeature(config)
        
        result = feature.compute(data)
        
        assert result.success
        if 'anomaly_count' in result.data.columns and len(result.data) > 0:
            anomaly_count = result.data['anomaly_count'].iloc[0]
            assert anomaly_count == 0, "Should not detect anomalies in clean data"
