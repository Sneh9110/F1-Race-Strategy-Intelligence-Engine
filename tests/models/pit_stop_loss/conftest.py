import pytest
import pandas as pd
import numpy as np
from models.pit_stop_loss.base import PredictionInput, ModelConfig
from models.pit_stop_loss.xgboost_model import XGBoostPitStopLossModel
from models.pit_stop_loss.lightgbm_model import LightGBMPitStopLossModel
from models.pit_stop_loss.ensemble_model import EnsemblePitStopLossModel


@pytest.fixture
def sample_prediction_input():
    return PredictionInput(
        track_name="Monza",
        current_lap=20,
        cars_in_pit_window=3,
        pit_stop_duration=2.5,
        traffic_density=0.6,
        tire_compound_change=True,
        current_position=5,
        gap_to_ahead=2.3,
        gap_to_behind=1.8,
    )


@pytest.fixture
def sample_training_data():
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "track_name": np.random.choice(["Monaco", "Monza", "Spa"], n),
        "cars_in_pit_window": np.random.randint(0, 6, n),
        "pit_stop_duration": np.random.uniform(2.0, 4.0, n),
        "traffic_density": np.random.rand(n),
        "tire_change": np.random.choice([0, 1], n),
        "gap_to_ahead": np.random.uniform(0.5, 5.0, n),
        "actual_pit_loss": np.random.uniform(15.0, 30.0, n),
    })
    return df


@pytest.fixture
def model_config():
    return ModelConfig(
        model_type="xgboost",
        hyperparameters={"n_estimators": 10, "max_depth": 3},
        version="test-1.0.0",
    )


@pytest.fixture
def xgboost_model(model_config):
    return XGBoostPitStopLossModel(config=model_config)


@pytest.fixture
def lightgbm_model(model_config):
    return LightGBMPitStopLossModel(config=model_config)


@pytest.fixture
def ensemble_model(model_config):
    return EnsemblePitStopLossModel(config=model_config)


@pytest.fixture
def trained_xgboost_model(xgboost_model, sample_training_data):
    X = sample_training_data.drop(columns=["actual_pit_loss"])
    y = sample_training_data["actual_pit_loss"]
    xgboost_model.train(X, y)
    return xgboost_model


@pytest.fixture
def temp_model_dir(tmp_path):
    return str(tmp_path / "models")
