from typing import Dict, Any
from sklearn import metrics


class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate(self, y_true, y_pred) -> Dict[str, Any]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        return {"mae": mae, "rmse": rmse, "r2": r2}
