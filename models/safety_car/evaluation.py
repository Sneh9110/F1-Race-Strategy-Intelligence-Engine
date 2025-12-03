from typing import Dict, Any
import numpy as np
from sklearn import metrics


class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate(self, y_true, y_pred_prob) -> Dict[str, Any]:
        try:
            auc = metrics.roc_auc_score(y_true, y_pred_prob)
        except Exception:
            auc = None
        brier = metrics.brier_score_loss(y_true, y_pred_prob) if auc is not None else None
        return {"auc": auc, "brier": brier}

    def calibration_curve(self, y_true, y_prob):
        return metrics.calibration_curve(y_true, y_prob, n_bins=10)
