"""
Test suite for feature engineering module.

Provides fixtures, utilities, and base classes for feature testing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Test configuration
TEST_SESSION_ID = "2024_MONACO_RACE"
TEST_DRIVER_ID = "VER"
TEST_RACE_LAPS = 78


@pytest.fixture
def sample_lap_data():
    """Generate sample lap timing data for testing."""
    np.random.seed(42)
    
    laps = []
    for lap_num in range(1, 31):
        # Simulate degradation: base time + degradation + noise
        base_time = 78.5
        degradation = 0.05 * lap_num  # 0.05s per lap degradation
        noise = np.random.normal(0, 0.2)
        
        lap_time = base_time + degradation + noise
        
        laps.append({
            'session_id': TEST_SESSION_ID,
            'driver_id': TEST_DRIVER_ID,
            'lap_number': lap_num,
            'lap_time': lap_time,
            'sector_1_time': lap_time * 0.32,
            'sector_2_time': lap_time * 0.41,
            'sector_3_time': lap_time * 0.27,
            'tire_age': lap_num,
            'tire_compound': 'MEDIUM',
            'track_status': 'GREEN',
            'is_pit_lap': False
        })
    
    return pd.DataFrame(laps)


@pytest.fixture
def sample_stint_data():
    """Generate sample stint data for testing."""
    stints = [
        {
            'session_id': TEST_SESSION_ID,
            'driver_id': TEST_DRIVER_ID,
            'stint_number': 1,
            'start_lap': 1,
            'end_lap': 15,
            'tire_compound': 'SOFT',
            'laps_in_stint': 15,
            'avg_lap_time': 78.8,
            'best_lap_time': 78.2,
            'tire_age_start': 1,
            'tire_age_end': 15
        },
        {
            'session_id': TEST_SESSION_ID,
            'driver_id': TEST_DRIVER_ID,
            'stint_number': 2,
            'start_lap': 16,
            'end_lap': 30,
            'tire_compound': 'MEDIUM',
            'laps_in_stint': 15,
            'avg_lap_time': 79.5,
            'best_lap_time': 78.9,
            'tire_age_start': 1,
            'tire_age_end': 15
        }
    ]
    return pd.DataFrame(stints)


@pytest.fixture
def sample_weather_data():
    """Generate sample weather data for testing."""
    weather = []
    for lap in range(1, 31):
        weather.append({
            'session_id': TEST_SESSION_ID,
            'lap_number': lap,
            'air_temp': 25.0 + np.random.normal(0, 1),
            'track_temp': 42.0 + np.random.normal(0, 2),
            'humidity': 60.0 + np.random.normal(0, 5),
            'rainfall': 0.0,
            'wind_speed': 10.0 + np.random.normal(0, 3)
        })
    return pd.DataFrame(weather)


@pytest.fixture
def sample_telemetry_data():
    """Generate sample telemetry data for testing."""
    telemetry = []
    for lap in range(1, 31):
        telemetry.append({
            'session_id': TEST_SESSION_ID,
            'driver_id': TEST_DRIVER_ID,
            'lap_number': lap,
            'throttle_max': 98.0 + np.random.normal(0, 1),
            'brake_max': 100.0,
            'speed_max': 285.0 + np.random.normal(0, 5),
            'rpm_max': 11500 + np.random.normal(0, 200),
            'drs_activations': np.random.randint(0, 3),
            'fuel_kg': 110 - (lap * 1.4)  # Fuel consumption
        })
    return pd.DataFrame(telemetry)


def assert_feature_output_valid(df: pd.DataFrame, expected_columns: List[str]):
    """
    Assert that feature output DataFrame is valid.
    
    Args:
        df: Feature output DataFrame
        expected_columns: List of expected column names
    """
    assert df is not None, "Feature output is None"
    assert not df.empty, "Feature output is empty"
    assert all(col in df.columns for col in expected_columns), \
        f"Missing columns: {set(expected_columns) - set(df.columns)}"
    assert df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) < 0.05, \
        "More than 5% null values in feature output"


def assert_no_outliers(df: pd.DataFrame, column: str, iqr_multiplier: float = 3.0):
    """
    Assert that a column has no extreme outliers.
    
    Args:
        df: DataFrame to check
        column: Column name to check for outliers
        iqr_multiplier: IQR multiplier for outlier detection
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    outlier_pct = outliers / len(df) * 100
    
    assert outlier_pct < 5, f"More than 5% outliers in {column}: {outlier_pct:.1f}%"


def assert_monotonic_trend(series: pd.Series, increasing: bool = True, tolerance: float = 0.1):
    """
    Assert that a series follows a monotonic trend (with tolerance).
    
    Args:
        series: Series to check
        increasing: Whether to check for increasing or decreasing trend
        tolerance: Tolerance for violations (as fraction)
    """
    if increasing:
        violations = (series.diff() < 0).sum()
    else:
        violations = (series.diff() > 0).sum()
    
    violation_pct = violations / (len(series) - 1)
    assert violation_pct < tolerance, \
        f"Series not monotonic: {violation_pct*100:.1f}% violations"
