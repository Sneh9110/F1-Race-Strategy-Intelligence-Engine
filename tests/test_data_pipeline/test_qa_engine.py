"""
Tests for QA Engine
"""

import pytest
import pandas as pd
from datetime import datetime

from data_pipeline.base.qa_engine import QAEngine


@pytest.fixture
def qa_engine():
    """Create QA engine instance."""
    return QAEngine(config={})


@pytest.fixture
def timing_data():
    """Sample timing data for testing."""
    return pd.DataFrame({
        'lap_number': [1, 2, 3],
        'lap_time': [72.5, 73.1, 72.8],
        'sector1_time': [24.2, 24.5, 24.3],
        'sector2_time': [24.1, 24.3, 24.2],
        'sector3_time': [24.2, 24.3, 24.3],
        'position': [1, 1, 1],
        'gap_to_leader': [0.0, 0.0, 0.0],
        'timestamp': [datetime.utcnow()] * 3
    })


def test_schema_compliance(qa_engine, timing_data):
    """Test schema compliance check."""
    report = qa_engine.run_checks(timing_data, source="timing")
    
    assert report.passed is True
    assert report.valid_records == 3


def test_value_range_validation(qa_engine):
    """Test value range validation."""
    invalid_data = pd.DataFrame({
        'lap_time': [20.0],  # Too fast, invalid
        'speed': [400],  # Too high
        'timestamp': [datetime.utcnow()]
    })
    
    report = qa_engine.run_checks(invalid_data, source="timing")
    
    # Should detect out-of-range values
    assert report.failed_records > 0 or len(report.warnings) > 0


def test_consistency_checks(qa_engine):
    """Test data consistency validation."""
    inconsistent_data = pd.DataFrame({
        'lap_time': [72.5],
        'sector1_time': [20.0],
        'sector2_time': [20.0],
        'sector3_time': [20.0],  # Sum != lap_time
        'timestamp': [datetime.utcnow()]
    })
    
    report = qa_engine.run_checks(inconsistent_data, source="timing")
    
    # Should detect inconsistency
    assert len(report.warnings) > 0


def test_anomaly_detection(qa_engine):
    """Test statistical anomaly detection."""
    # Normal data with one outlier
    data = pd.DataFrame({
        'lap_time': [72.0, 72.5, 73.0, 72.8, 150.0],  # 150.0 is outlier
        'timestamp': [datetime.utcnow()] * 5
    })
    
    report = qa_engine.run_checks(data, source="timing")
    
    assert report.anomalies_detected > 0


def test_completeness_check(qa_engine):
    """Test completeness validation."""
    incomplete_data = pd.DataFrame({
        'lap_time': [72.5, None, 73.0],
        'sector1_time': [24.0, 24.1, None],
        'timestamp': [datetime.utcnow()] * 3
    })
    
    report = qa_engine.run_checks(incomplete_data, source="timing")
    
    # Should detect null values
    assert report.failed_records > 0 or len(report.warnings) > 0
