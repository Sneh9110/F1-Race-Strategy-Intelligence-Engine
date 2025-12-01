"""
Ensemble Lap Time Prediction Model

Combines XGBoost (speed) and LightGBM (accuracy) for optimal performance.
Uses weighted voting with dynamic weight adjustment based on conditions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd
from pydantic import ValidationError

from .base import (
    BaseLapTimeModel,
    ModelConfig,
    PredictionInput,
    PredictionOutput,
    RaceCondition
)
from .xgboost_model import XGBoostLapTimeModel
from .lightgbm_model import LightGBMLapTimeModel
from config.settings import Settings

logger = logging.getLogger(__name__)


class EnsembleLapTimeModel(BaseLapTimeModel):
    """
    Ensemble model combining XGBoost and LightGBM.
    
    Strategy:
    - XGBoost: Fast predictions, good for common scenarios (weight: 0.4)
    - LightGBM: Accurate predictions with uncertainty, good for rare scenarios (weight: 0.6)
    - Dynamic weighting based on confidence scores
    
    Attributes:
        xgboost_model: XGBoost implementation
        lightgbm_model: LightGBM implementation
        weights: Ensemble weights [xgb_weight, lgb_weight]
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize ensemble model.
        
        Args:
            config: Model configuration. If None, uses default settings.
        """
        super().__init__(config)
        self.xgboost_model = XGBoostLapTimeModel(config)
        self.lightgbm_model = LightGBMLapTimeModel(config)
        
        # Default weights (can be optimized during training)
        self.weights = [0.4, 0.6]  # [xgboost, lightgbm]
        self.optimal_weights_found = False
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train both base models and optimize ensemble weights.
        
        Args:
            X_train: Training features
            y_train: Training targets (lap times in seconds)
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training ensemble model on {len(X_train)} samples")
        
        # Train XGBoost
        logger.info("Training XGBoost component...")
        xgb_metrics = self.xgboost_model.train(X_train, y_train, X_val, y_val, **kwargs)
        
        # Train LightGBM
        logger.info("Training LightGBM component...")
        lgb_metrics = self.lightgbm_model.train(X_train, y_train, X_val, y_val, **kwargs)
        
        # Optimize ensemble weights on validation set
        if X_val is not None and y_val is not None:
            logger.info("Optimizing ensemble weights...")
            self._optimize_weights(X_val, y_val)
        
        # Calculate ensemble metrics
        train_pred = self._ensemble_predict(X_train)
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        train_mae = np.mean(np.abs(y_train - train_pred))
        
        metrics = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'xgb_train_rmse': xgb_metrics['train_rmse'],
            'lgb_train_rmse': lgb_metrics['train_rmse'],
            'ensemble_weights': self.weights,
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self._ensemble_predict(X_val)
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
            val_mae = np.mean(np.abs(y_val - val_pred))
            
            metrics.update({
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'xgb_val_rmse': xgb_metrics.get('val_rmse', 0),
                'lgb_val_rmse': lgb_metrics.get('val_rmse', 0),
                'improvement_over_xgb': (xgb_metrics.get('val_rmse', val_rmse) - val_rmse) / xgb_metrics.get('val_rmse', val_rmse),
                'improvement_over_lgb': (lgb_metrics.get('val_rmse', val_rmse) - val_rmse) / lgb_metrics.get('val_rmse', val_rmse),
            })
        
        logger.info(f"Ensemble training complete. Train RMSE: {train_rmse:.3f}s, MAE: {train_mae:.3f}s")
        if 'val_rmse' in metrics:
            logger.info(f"Validation RMSE: {metrics['val_rmse']:.3f}s, MAE: {metrics['val_mae']:.3f}s")
            logger.info(f"Ensemble weights: XGBoost={self.weights[0]:.2f}, LightGBM={self.weights[1]:.2f}")
        
        return metrics
    
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Predict lap time using ensemble of models.
        
        Args:
            input_data: Prediction input with race conditions
            
        Returns:
            Prediction output with lap time and confidence
        """
        # Validate input
        self._validate_input(input_data)
        
        # Get predictions from both models
        xgb_output = self.xgboost_model.predict(input_data)
        lgb_output = self.lightgbm_model.predict(input_data)
        
        # Dynamic weighting based on confidence
        if self.optimal_weights_found:
            # Use optimized weights
            xgb_weight, lgb_weight = self.weights
        else:
            # Adjust weights based on confidence scores
            xgb_conf = xgb_output.confidence
            lgb_conf = lgb_output.confidence
            total_conf = xgb_conf + lgb_conf
            xgb_weight = xgb_conf / total_conf if total_conf > 0 else 0.5
            lgb_weight = lgb_conf / total_conf if total_conf > 0 else 0.5
        
        # Weighted ensemble prediction
        ensemble_lap_time = (
            xgb_weight * xgb_output.predicted_lap_time +
            lgb_weight * lgb_output.predicted_lap_time
        )
        
        # Combine pace components
        pace_components = {}
        for key in xgb_output.pace_components:
            pace_components[key] = (
                xgb_weight * xgb_output.pace_components[key] +
                lgb_weight * lgb_output.pace_components[key]
            )
        
        # Use LightGBM's uncertainty range if available
        uncertainty_range = lgb_output.uncertainty_range
        
        # Ensemble confidence (weighted average)
        ensemble_confidence = xgb_weight * xgb_output.confidence + lgb_weight * lgb_output.confidence
        
        # Boost confidence if models agree closely
        prediction_diff = abs(xgb_output.predicted_lap_time - lgb_output.predicted_lap_time)
        if prediction_diff < 0.5:  # Models agree within 0.5s
            ensemble_confidence = min(1.0, ensemble_confidence * 1.1)
        elif prediction_diff > 2.0:  # Models disagree significantly
            ensemble_confidence *= 0.9
        
        return PredictionOutput(
            predicted_lap_time=float(ensemble_lap_time),
            confidence=ensemble_confidence,
            pace_components=pace_components,
            uncertainty_range=uncertainty_range,
            metadata={
                'model_type': 'ensemble',
                'version': self.config.version,
                'xgb_prediction': xgb_output.predicted_lap_time,
                'lgb_prediction': lgb_output.predicted_lap_time,
                'xgb_weight': xgb_weight,
                'lgb_weight': lgb_weight,
                'prediction_agreement': float(prediction_diff),
            }
        )
    
    def predict_batch(self, inputs: List[PredictionInput]) -> List[PredictionOutput]:
        """
        Predict lap times for multiple inputs using ensemble.
        
        Args:
            inputs: List of prediction inputs
            
        Returns:
            List of prediction outputs
        """
        # Get predictions from both models
        xgb_outputs = self.xgboost_model.predict_batch(inputs)
        lgb_outputs = self.lightgbm_model.predict_batch(inputs)
        
        # Combine predictions
        ensemble_outputs = []
        for xgb_out, lgb_out, input_data in zip(xgb_outputs, lgb_outputs, inputs):
            # Dynamic weighting
            if self.optimal_weights_found:
                xgb_weight, lgb_weight = self.weights
            else:
                xgb_conf = xgb_out.confidence
                lgb_conf = lgb_out.confidence
                total_conf = xgb_conf + lgb_conf
                xgb_weight = xgb_conf / total_conf if total_conf > 0 else 0.5
                lgb_weight = lgb_conf / total_conf if total_conf > 0 else 0.5
            
            # Ensemble prediction
            ensemble_lap_time = (
                xgb_weight * xgb_out.predicted_lap_time +
                lgb_weight * lgb_out.predicted_lap_time
            )
            
            # Combine pace components
            pace_components = {}
            for key in xgb_out.pace_components:
                pace_components[key] = (
                    xgb_weight * xgb_out.pace_components[key] +
                    lgb_weight * lgb_out.pace_components[key]
                )
            
            # Ensemble confidence
            ensemble_confidence = xgb_weight * xgb_out.confidence + lgb_weight * lgb_out.confidence
            prediction_diff = abs(xgb_out.predicted_lap_time - lgb_out.predicted_lap_time)
            if prediction_diff < 0.5:
                ensemble_confidence = min(1.0, ensemble_confidence * 1.1)
            elif prediction_diff > 2.0:
                ensemble_confidence *= 0.9
            
            ensemble_outputs.append(PredictionOutput(
                predicted_lap_time=float(ensemble_lap_time),
                confidence=ensemble_confidence,
                pace_components=pace_components,
                uncertainty_range=lgb_out.uncertainty_range,
                metadata={
                    'model_type': 'ensemble',
                    'version': self.config.version,
                    'xgb_weight': xgb_weight,
                    'lgb_weight': lgb_weight,
                }
            ))
        
        return ensemble_outputs
    
    def save(self, path: Path) -> None:
        """
        Save ensemble model to disk.
        
        Args:
            path: Directory to save model files
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Save base models
        self.xgboost_model.save(path / "xgboost")
        self.lightgbm_model.save(path / "lightgbm")
        
        # Save ensemble metadata
        metadata = {
            'weights': self.weights,
            'optimal_weights_found': self.optimal_weights_found,
            'config': self.config.__dict__,
        }
        with open(path / "ensemble_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Ensemble model saved to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load ensemble model from disk.
        
        Args:
            path: Directory containing model files
        """
        # Load base models
        self.xgboost_model.load(path / "xgboost")
        self.lightgbm_model.load(path / "lightgbm")
        
        # Load ensemble metadata
        metadata_path = path / "ensemble_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.weights = metadata['weights']
            self.optimal_weights_found = metadata['optimal_weights_found']
            
            if 'config' in metadata:
                self.config = ModelConfig(**metadata['config'])
        
        logger.info(f"Ensemble model loaded from {path}")
    
    def get_metadata(self) -> Dict:
        """
        Get ensemble model metadata.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'ensemble',
            'version': self.config.version,
            'weights': self.weights,
            'optimal_weights_found': self.optimal_weights_found,
            'xgboost_metadata': self.xgboost_model.get_metadata(),
            'lightgbm_metadata': self.lightgbm_model.get_metadata(),
        }
    
    def _ensemble_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions on feature matrix.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        # Get predictions from base models
        xgb_pred = self.xgboost_model.model.predict(
            __import__('xgboost').DMatrix(X, feature_names=self.xgboost_model.feature_names)
        )
        lgb_pred = self.lightgbm_model.model.predict(X)
        
        # Weighted ensemble
        ensemble_pred = self.weights[0] * xgb_pred + self.weights[1] * lgb_pred
        return ensemble_pred
    
    def _optimize_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Optimize ensemble weights on validation set.
        
        Uses grid search to find optimal weights that minimize RMSE.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        logger.info("Searching for optimal ensemble weights...")
        
        # Get base model predictions
        xgb_pred = self.xgboost_model.model.predict(
            __import__('xgboost').DMatrix(X_val, feature_names=self.xgboost_model.feature_names)
        )
        lgb_pred = self.lightgbm_model.model.predict(X_val)
        
        # Grid search over weights
        best_rmse = float('inf')
        best_weights = [0.5, 0.5]
        
        for xgb_weight in np.arange(0.0, 1.01, 0.1):
            lgb_weight = 1.0 - xgb_weight
            
            # Calculate ensemble prediction
            ensemble_pred = xgb_weight * xgb_pred + lgb_weight * lgb_pred
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((y_val - ensemble_pred) ** 2))
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = [xgb_weight, lgb_weight]
        
        self.weights = best_weights
        self.optimal_weights_found = True
        
        logger.info(f"Optimal weights found: XGBoost={best_weights[0]:.2f}, LightGBM={best_weights[1]:.2f}")
        logger.info(f"Validation RMSE with optimal weights: {best_rmse:.3f}s")
