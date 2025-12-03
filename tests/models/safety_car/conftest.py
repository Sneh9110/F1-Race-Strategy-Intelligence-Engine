import pytest
import pandas as pd
import numpy as np
from models.safety_car.base import PredictionInput, IncidentLog, ModelConfig
from models.safety_car.xgboost_model import XGBoostSafetyCarModel
from models.safety_car.lightgbm_model import LightGBMSafetyCarModel
from models.safety_car.ensemble_model import EnsembleSafetyCarModel


@pytest.fixture
def sample_prediction_input():
    incidents = [
        IncidentLog(lap=10, sector="T1", severity="minor"),
        IncidentLog(lap=12, sector="T2", severity="moderate"),
    ]
    return PredictionInput(
        track_name="Monaco",
        current_lap=15,
        total_laps=78,
        race_progress=0.19,
        incident_logs=incidents,
        driver_proximity_data=pd.DataFrame({"gap_to_ahead": [1.2, 2.3]}),
        sector_risks={"T1": 0.6, "T2": 0.4, "T3": 0.3},
    )


@pytest.fixture
def sample_training_data():
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "track_name": np.random.choice(["Monaco", "Monza", "Spa"], n),
        "race_progress": np.random.rand(n),
        "current_lap": np.random.randint(1, 70, n),
        "incident_count": np.random.randint(0, 5, n),
        "sector_T1": np.random.rand(n),
        "sector_T2": np.random.rand(n),
        "sector_T3": np.random.rand(n),
        "sc_deployed": np.random.choice([0, 1], n, p=[0.7, 0.3]),
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
    return XGBoostSafetyCarModel(config=model_config)


@pytest.fixture
def lightgbm_model(model_config):
    return LightGBMSafetyCarModel(config=model_config)


@pytest.fixture
def ensemble_model(model_config):
    return EnsembleSafetyCarModel(config=model_config)


@pytest.fixture
def trained_xgboost_model(xgboost_model, sample_training_data):
    X = sample_training_data.drop(columns=["sc_deployed"])
    y = sample_training_data["sc_deployed"]
    xgboost_model.train(X, y)
    return xgboost_model


@pytest.fixture
def temp_model_dir(tmp_path):
    return str(tmp_path / "models")
