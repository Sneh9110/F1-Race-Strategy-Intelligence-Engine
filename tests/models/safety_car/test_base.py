import pytest
from models.safety_car.base import PredictionInput, PredictionOutput, IncidentLog


def test_prediction_input_valid(sample_prediction_input):
    assert sample_prediction_input.track_name == "Monaco"
    assert sample_prediction_input.current_lap == 15
    assert 0.0 <= sample_prediction_input.race_progress <= 1.0


def test_prediction_input_invalid_lap():
    with pytest.raises(ValueError):
        PredictionInput(
            track_name="Monaco",
            current_lap=0,
            total_laps=78,
            race_progress=0.19,
        )


def test_prediction_output_valid():
    out = PredictionOutput(sc_probability=0.5, confidence=0.7)
    assert out.sc_probability == 0.5
    assert out.confidence == 0.7
