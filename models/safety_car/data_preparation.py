from typing import Tuple
import pandas as pd


class DataPreparationPipeline:
    def __init__(self, config: dict = None):
        self.config = config or {}

    def prepare_training_data(self, path: str) -> Tuple[pd.DataFrame, pd.Series]:
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        # assume label column is 'sc_deployed'
        y = df.get("sc_deployed")
        X = df.drop(columns=["sc_deployed"], errors="ignore")
        return X, y

    def prepare_inference_data(self, inp) -> pd.DataFrame:
        import pandas as pd
        if isinstance(inp, dict):
            return pd.DataFrame([inp])
        if hasattr(inp, "dict"):
            return pd.DataFrame([inp.dict()])
        raise ValueError("Unsupported input type for inference data preparation")
