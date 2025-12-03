import pytest
from models.pit_stop_loss.base import PredictionInput, PredictionOutput


def test_prediction_input_valid(sample_prediction_input):
    assert sample_prediction_input.track_name == "Monza"
    assert sample_prediction_input.current_lap == 20


def test_prediction_input_invalid_traffic():
    with pytest.raises(ValueError):
        PredictionInput(
            track_name="Monza",
            current_lap=20,
            cars_in_pit_window=3,
            pit_stop_duration=2.5,
            traffic_density=1.5,
            tire_compound_change=True,
            current_position=5,
        )


def test_prediction_output_valid():
    out = PredictionOutput(
        total_pit_loss=22.0,
        pit_delta=2.0,
        window_sensitivity=0.5,
        congestion_penalty=3.0,
        base_pit_loss=20.0,
    )
    assert out.total_pit_loss == 22.0
