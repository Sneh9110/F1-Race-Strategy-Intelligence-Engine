"""Tests for fallback heuristics."""

import pytest

from models.tire_degradation.fallback import FallbackHeuristics


def test_fallback_initialization():
    """Test fallback initialization."""
    fallback = FallbackHeuristics()
    assert fallback.compound_data is not None
    assert 'MEDIUM' in fallback.compound_data


def test_fallback_prediction(sample_prediction_input):
    """Test fallback prediction."""
    fallback = FallbackHeuristics()
    output = fallback.predict(sample_prediction_input)
    
    assert output.degradation_rate > 0
    assert len(output.degradation_curve) > 0
    assert output.metadata['model_type'] == 'fallback_heuristic'
    assert output.confidence < 0.8  # Lower confidence for heuristics


def test_get_compound_info():
    """Test compound info retrieval."""
    fallback = FallbackHeuristics()
    info = fallback.get_compound_info('SOFT')
    
    assert 'base_degradation_rate' in info
    assert 'expected_life' in info
