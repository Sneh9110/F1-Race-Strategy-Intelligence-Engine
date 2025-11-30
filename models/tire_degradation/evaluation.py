"""
Model evaluation framework with comprehensive metrics.

Calculates MAE, RMSE, R², cliff detection accuracy, and usable life error.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.tire_degradation.base import BaseDegradationModel, PredictionInput
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for tire degradation.
    
    Metrics:
    - MAE, RMSE, R² for degradation rate
    - Cliff detection accuracy
    - Usable life prediction error
    - Curve similarity (DTW)
    """
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate(
        self,
        model: BaseDegradationModel,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        additional_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test targets (degradation rates)
            additional_data: Additional data for curve/life evaluation
        
        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model performance")
        
        # Get predictions
        y_pred = self._get_predictions(model, X_test)
        
        # Basic regression metrics
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': self._mean_absolute_percentage_error(y_test, y_pred)
        }
        
        # Additional metrics if data available
        if additional_data is not None:
            try:
                curve_metrics = self._evaluate_curves(model, additional_data)
                metrics.update(curve_metrics)
                
                life_metrics = self._evaluate_usable_life(model, additional_data)
                metrics.update(life_metrics)
                
                cliff_metrics = self._evaluate_cliff_detection(model, additional_data)
                metrics.update(cliff_metrics)
            except Exception as e:
                logger.warning(f"Could not compute additional metrics: {e}")
        
        logger.info(f"Evaluation complete: {metrics}")
        return metrics
    
    def _get_predictions(
        self,
        model: BaseDegradationModel,
        X_test: pd.DataFrame
    ) -> np.ndarray:
        """Get model predictions for test set."""
        predictions = []
        
        for idx in range(len(X_test)):
            try:
                if hasattr(model, 'model'):
                    # Use underlying model directly
                    pred = model.model.predict(X_test.iloc[[idx]])[0]
                else:
                    # Use model's predict method (requires PredictionInput)
                    pred = model.predict(self._row_to_input(X_test.iloc[idx])).degradation_rate
                
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Prediction failed for row {idx}: {e}")
                predictions.append(0.0)
        
        return np.array(predictions)
    
    def _row_to_input(self, row: pd.Series) -> PredictionInput:
        """Convert DataFrame row to PredictionInput."""
        # This is a simplified conversion - adjust based on actual features
        return PredictionInput(
            tire_compound=row.get('tire_compound', 'MEDIUM'),
            tire_age=row.get('tire_age', 10),
            stint_history=[],
            weather_temp=row.get('weather_temp', 25.0),
            driver_aggression=row.get('driver_aggression', 0.5),
            track_name=row.get('track_name', 'default')
        )
    
    def _mean_absolute_percentage_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Calculate MAPE."""
        # Avoid division by zero
        mask = y_true != 0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    def _evaluate_curves(
        self,
        model: BaseDegradationModel,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate degradation curve predictions.
        
        Args:
            model: Model to evaluate
            data: Test data with actual curves
        
        Returns:
            Curve evaluation metrics
        """
        dtw_distances = []
        curve_rmses = []
        
        for idx in range(min(len(data), 100)):  # Sample for efficiency
            try:
                row = data.iloc[idx]
                pred_input = self._row_to_input(row)
                
                # Get predicted curve
                pred_curve = model.predict_curve(pred_input, num_laps=30)
                
                # Get actual curve if available
                if 'actual_curve' in row and row['actual_curve']:
                    actual_curve = row['actual_curve'][:len(pred_curve)]
                    
                    # Pad if needed
                    if len(actual_curve) < len(pred_curve):
                        actual_curve = actual_curve + [actual_curve[-1]] * (len(pred_curve) - len(actual_curve))
                    
                    # Calculate DTW distance
                    dtw_dist = self._dtw_distance(pred_curve, actual_curve[:len(pred_curve)])
                    dtw_distances.append(dtw_dist)
                    
                    # Calculate RMSE
                    curve_rmse = np.sqrt(mean_squared_error(actual_curve[:len(pred_curve)], pred_curve))
                    curve_rmses.append(curve_rmse)
            
            except Exception as e:
                logger.warning(f"Curve evaluation failed for row {idx}: {e}")
                continue
        
        metrics = {}
        if dtw_distances:
            metrics['curve_dtw_mean'] = float(np.mean(dtw_distances))
            metrics['curve_dtw_std'] = float(np.std(dtw_distances))
        
        if curve_rmses:
            metrics['curve_rmse_mean'] = float(np.mean(curve_rmses))
            metrics['curve_rmse_std'] = float(np.std(curve_rmses))
        
        return metrics
    
    def _dtw_distance(self, series1: List[float], series2: List[float]) -> float:
        """
        Calculate Dynamic Time Warping distance.
        
        Args:
            series1: First time series
            series2: Second time series
        
        Returns:
            DTW distance
        """
        n, m = len(series1), len(series2)
        dtw = np.zeros((n + 1, m + 1))
        
        for i in range(1, n + 1):
            dtw[i][0] = float('inf')
        for j in range(1, m + 1):
            dtw[0][j] = float('inf')
        dtw[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(series1[i-1] - series2[j-1])
                dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
        
        return float(dtw[n][m])
    
    def _evaluate_usable_life(
        self,
        model: BaseDegradationModel,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate usable life predictions.
        
        Args:
            model: Model to evaluate
            data: Test data with actual usable life
        
        Returns:
            Usable life metrics
        """
        errors = []
        
        for idx in range(min(len(data), 100)):
            try:
                row = data.iloc[idx]
                pred_input = self._row_to_input(row)
                
                # Get predicted usable life
                pred_life = model.predict_usable_life(pred_input)
                
                # Get actual usable life if available
                if 'actual_usable_life' in row:
                    actual_life = row['actual_usable_life']
                    error = abs(pred_life - actual_life)
                    errors.append(error)
            
            except Exception as e:
                logger.warning(f"Life evaluation failed for row {idx}: {e}")
                continue
        
        metrics = {}
        if errors:
            metrics['usable_life_mae'] = float(np.mean(errors))
            metrics['usable_life_std'] = float(np.std(errors))
            metrics['usable_life_max_error'] = float(np.max(errors))
        
        return metrics
    
    def _evaluate_cliff_detection(
        self,
        model: BaseDegradationModel,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate cliff/dropoff detection accuracy.
        
        Args:
            model: Model to evaluate
            data: Test data with actual cliff laps
        
        Returns:
            Cliff detection metrics
        """
        correct = 0
        total = 0
        errors = []
        
        for idx in range(min(len(data), 100)):
            try:
                row = data.iloc[idx]
                pred_input = self._row_to_input(row)
                
                # Get predicted dropoff lap
                pred_dropoff = model.predict_dropoff_lap(pred_input)
                
                # Get actual dropoff if available
                if 'actual_dropoff_lap' in row:
                    actual_dropoff = row['actual_dropoff_lap']
                    
                    # Both predicted and actual have dropoff
                    if pred_dropoff is not None and actual_dropoff is not None:
                        error = abs(pred_dropoff - actual_dropoff)
                        errors.append(error)
                        
                        # Within 2 laps is considered correct
                        if error <= 2:
                            correct += 1
                        total += 1
                    
                    # Both predicted no dropoff
                    elif pred_dropoff is None and actual_dropoff is None:
                        correct += 1
                        total += 1
                    
                    # One predicted dropoff, other didn't
                    else:
                        total += 1
            
            except Exception as e:
                logger.warning(f"Cliff evaluation failed for row {idx}: {e}")
                continue
        
        metrics = {}
        if total > 0:
            metrics['cliff_detection_accuracy'] = correct / total
        
        if errors:
            metrics['cliff_prediction_mae'] = float(np.mean(errors))
        
        return metrics
    
    def compare_models(
        self,
        models: Dict[str, BaseDegradationModel],
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models: Dict of model_name -> model
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Comparison DataFrame
        """
        results = []
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}")
            metrics = self.evaluate(model, X_test, y_test)
            metrics['model_name'] = name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        
        # Reorder columns
        cols = ['model_name'] + [c for c in df.columns if c != 'model_name']
        df = df[cols]
        
        return df
