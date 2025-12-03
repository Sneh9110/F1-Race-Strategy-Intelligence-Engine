from typing import Optional, Tuple, Dict, Any
from .base import ModelConfig
import pandas as pd
import time


class ModelTrainer:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(model_type="pit_stop_trainer")

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns=["actual_pit_loss"], errors="ignore")
        y = df.get("actual_pit_loss")
        return X, y

    def train(self, model, df: pd.DataFrame, optimize: bool = False, **kwargs) -> Dict[str, Any]:
        X, y = self.prepare_data(df)
        start = time.time()
        model.train(X, y)
        elapsed = time.time() - start
        metrics = {"training_time_s": elapsed}
        return {"model": model, "metrics": metrics}

    def optimize_hyperparameters(self, df: pd.DataFrame, n_trials: int = 20) -> Dict[str, Any]:
        return {"best_params": {}}
