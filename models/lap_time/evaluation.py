"""
Model Evaluation Framework

Comprehensive metrics and analysis for lap time prediction models.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base import BaseLapTimeModel, PredictionInput, RaceCondition
from config.settings import Settings

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluation framework for lap time models.
    
    Metrics:
    - MAE, RMSE, R², MAPE (standard regression metrics)
    - Accuracy within thresholds (0.5s, 1.0s)
    - Condition-specific accuracy (clean air, dirty air, safety car)
    - Track-specific performance
    - Compound-specific performance
    
    Attributes:
        None (stateless)
    """
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate_model(
        self,
        model: BaseLapTimeModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_data: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets (actual lap times)
            test_data: Full test dataset with metadata
            output_path: Optional path to save evaluation plots
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        # Generate predictions
        predictions = self._predict_from_features(model, X_test)
        
        # Calculate standard metrics
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions),
            'mape': self._calculate_mape(y_test, predictions),
        }
        
        # Accuracy within thresholds
        metrics['accuracy_0.5s'] = np.mean(np.abs(y_test - predictions) <= 0.5)
        metrics['accuracy_1.0s'] = np.mean(np.abs(y_test - predictions) <= 1.0)
        metrics['accuracy_2.0s'] = np.mean(np.abs(y_test - predictions) <= 2.0)
        
        # Pace component errors (if available in test_data)
        if 'base_pace' in test_data.columns:
            pace_errors = self._evaluate_pace_components(
                model, X_test, test_data
            )
            metrics.update(pace_errors)
        
        # Condition-specific accuracy
        condition_metrics = self._evaluate_by_condition(
            predictions, y_test, test_data
        )
        metrics.update(condition_metrics)
        
        # Track-specific performance
        if 'track_name' in test_data.columns:
            track_metrics = self._evaluate_by_track(
                predictions, y_test, test_data
            )
            metrics.update(track_metrics)
        
        # Compound-specific performance
        if 'tire_compound' in test_data.columns:
            compound_metrics = self._evaluate_by_compound(
                predictions, y_test, test_data
            )
            metrics.update(compound_metrics)
        
        # Generate evaluation plots
        if output_path:
            self.plot_evaluation(
                predictions, y_test, test_data, output_path
            )
        
        logger.info(f"Evaluation complete. MAE: {metrics['mae']:.3f}s, RMSE: {metrics['rmse']:.3f}s, R²: {metrics['r2']:.3f}")
        
        return metrics
    
    def compare_models(
        self,
        models: Dict[str, BaseLapTimeModel],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare multiple models on same test set.
        
        Args:
            models: Dictionary of {model_name: model}
            X_test: Test features
            y_test: Test targets
            test_data: Full test dataset
            
        Returns:
            Comparison DataFrame
        """
        logger.info(f"Comparing {len(models)} models")
        
        results = []
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, X_test, y_test, test_data)
            metrics['model_name'] = model_name
            results.append(metrics)
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.set_index('model_name')
        
        return comparison_df
    
    def plot_evaluation(
        self,
        predictions: np.ndarray,
        actuals: pd.Series,
        test_data: pd.DataFrame,
        output_path: Path
    ) -> None:
        """
        Generate evaluation plots.
        
        Args:
            predictions: Model predictions
            actuals: Actual lap times
            test_data: Full test dataset
            output_path: Directory to save plots
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Prediction vs Actual scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        plt.xlabel('Actual Lap Time (s)')
        plt.ylabel('Predicted Lap Time (s)')
        plt.title('Prediction vs Actual Lap Times')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'prediction_vs_actual.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Residual plot
        residuals = actuals - predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Lap Time (s)')
        plt.ylabel('Residual (s)')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'residuals.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Error distribution
        errors = np.abs(actuals - predictions)
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black')
        plt.xlabel('Absolute Error (s)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.axvline(x=0.5, color='g', linestyle='--', label='0.5s threshold')
        plt.axvline(x=1.0, color='orange', linestyle='--', label='1.0s threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path / 'error_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Error by condition (if available)
        if 'traffic_state' in test_data.columns:
            plt.figure(figsize=(10, 6))
            test_data['error'] = errors
            sns.boxplot(data=test_data, x='traffic_state', y='error')
            plt.ylabel('Absolute Error (s)')
            plt.title('Error by Traffic Condition')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path / 'error_by_condition.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Evaluation plots saved to {output_path}")
    
    def analyze_errors(
        self,
        model: BaseLapTimeModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_data: pd.DataFrame,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Analyze worst prediction errors.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            test_data: Full test dataset
            top_n: Number of worst errors to analyze
            
        Returns:
            DataFrame with worst predictions
        """
        predictions = self._predict_from_features(model, X_test)
        errors = np.abs(y_test - predictions)
        
        # Get indices of worst errors
        worst_indices = np.argsort(errors)[-top_n:][::-1]
        
        # Create analysis DataFrame
        analysis = pd.DataFrame({
            'actual': y_test.iloc[worst_indices].values,
            'predicted': predictions[worst_indices],
            'error': errors.iloc[worst_indices].values,
        })
        
        # Add context columns if available
        context_cols = ['track_name', 'tire_compound', 'tire_age', 'fuel_load', 
                       'traffic_state', 'safety_car_active']
        for col in context_cols:
            if col in test_data.columns:
                analysis[col] = test_data[col].iloc[worst_indices].values
        
        return analysis
    
    def _predict_from_features(
        self,
        model: BaseLapTimeModel,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Generate predictions from feature matrix.
        
        Args:
            model: Trained model
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        # Get model type
        model_type = model.config.model_type
        
        if model_type == "xgboost":
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X, feature_names=model.feature_names)
            return model.model.predict(dmatrix)
        elif model_type == "lightgbm":
            return model.model.predict(X)
        elif model_type == "ensemble":
            # Use ensemble's internal method
            return model._ensemble_predict(X)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _calculate_mape(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def _evaluate_pace_components(
        self,
        model: BaseLapTimeModel,
        X_test: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate pace component predictions.
        
        Args:
            model: Trained model
            X_test: Test features
            test_data: Full test dataset with ground truth components
            
        Returns:
            Dictionary of component-specific metrics
        """
        # This requires ground truth pace components
        # For now, return empty dict
        # TODO: Implement if ground truth components available
        return {}
    
    def _evaluate_by_condition(
        self,
        predictions: np.ndarray,
        actuals: pd.Series,
        test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate performance by race condition.
        
        Args:
            predictions: Model predictions
            actuals: Actual lap times
            test_data: Full test dataset
            
        Returns:
            Dictionary of condition-specific metrics
        """
        metrics = {}
        
        if 'traffic_state' not in test_data.columns:
            return metrics
        
        conditions = test_data['traffic_state'].unique()
        for condition in conditions:
            mask = test_data['traffic_state'] == condition
            if mask.sum() > 0:
                cond_actuals = actuals[mask]
                cond_preds = predictions[mask]
                
                cond_mae = mean_absolute_error(cond_actuals, cond_preds)
                cond_rmse = np.sqrt(mean_squared_error(cond_actuals, cond_preds))
                
                metrics[f'{condition}_mae'] = cond_mae
                metrics[f'{condition}_rmse'] = cond_rmse
        
        return metrics
    
    def _evaluate_by_track(
        self,
        predictions: np.ndarray,
        actuals: pd.Series,
        test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate performance by track.
        
        Args:
            predictions: Model predictions
            actuals: Actual lap times
            test_data: Full test dataset
            
        Returns:
            Dictionary of track-specific metrics
        """
        metrics = {}
        
        if 'track_name' not in test_data.columns:
            return metrics
        
        # Calculate average MAE per track
        track_maes = []
        for track in test_data['track_name'].unique():
            mask = test_data['track_name'] == track
            if mask.sum() > 0:
                track_actuals = actuals[mask]
                track_preds = predictions[mask]
                track_mae = mean_absolute_error(track_actuals, track_preds)
                track_maes.append(track_mae)
        
        if track_maes:
            metrics['track_mae_mean'] = np.mean(track_maes)
            metrics['track_mae_std'] = np.std(track_maes)
        
        return metrics
    
    def _evaluate_by_compound(
        self,
        predictions: np.ndarray,
        actuals: pd.Series,
        test_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Evaluate performance by tire compound.
        
        Args:
            predictions: Model predictions
            actuals: Actual lap times
            test_data: Full test dataset
            
        Returns:
            Dictionary of compound-specific metrics
        """
        metrics = {}
        
        if 'tire_compound' not in test_data.columns:
            return metrics
        
        compounds = test_data['tire_compound'].unique()
        for compound in compounds:
            mask = test_data['tire_compound'] == compound
            if mask.sum() > 0:
                comp_actuals = actuals[mask]
                comp_preds = predictions[mask]
                
                comp_mae = mean_absolute_error(comp_actuals, comp_preds)
                metrics[f'{compound}_mae'] = comp_mae
        
        return metrics
