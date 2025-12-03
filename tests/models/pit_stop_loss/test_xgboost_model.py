import pytest


def test_xgboost_model_init(xgboost_model):
    assert xgboost_model.model is None


def test_xgboost_model_train(xgboost_model, sample_training_data):
    X = sample_training_data.drop(columns=["actual_pit_loss"])
    y = sample_training_data["actual_pit_loss"]
    xgboost_model.train(X, y)
    assert xgboost_model.model is not None


def test_xgboost_model_predict(trained_xgboost_model, sample_prediction_input):
    out = trained_xgboost_model.predict(sample_prediction_input)
    assert out.total_pit_loss > 0.0
    assert 0.0 <= out.confidence <= 1.0
