"""
XGBoost Lap Time Prediction Model

Optimized for speed with <150ms inference latency.
Uses gradient boosting for fast predictions with minimal overhead.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import ValidationError

from .base import (
    BaseLapTimeModel,
    ModelConfig,
    PredictionInput,
    PredictionOutput,
    RaceCondition
)
from config.settings import Settings

logger = logging.getLogger(__name__)


class XGBoostLapTimeModel(BaseLapTimeModel):
    """
    XGBoost implementation of lap time prediction.
    
    Optimized for:
    - Speed: <150ms inference time
    - Efficiency: Minimal memory footprint
    - Robustness: Handles missing values and outliers well
    
    Attributes:
        model: XGBoost DMatrix model
        feature_names: Ordered list of feature names
        scaler: StandardScaler for feature normalization
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize XGBoost model.
        
        Args:
            config: Model configuration. If None, uses default settings.
        """
        super().__init__(config)
        self.model: Optional[xgb.Booster] = None
        self.feature_names: List[str] = []
        self.scaler = None
        self.feature_importance_: Dict[str, float] = {}
        
        # XGBoost hyperparameters optimized for speed
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.01,
            'reg_lambda': 1.0,
            'tree_method': 'hist',  # Fast histogram-based algorithm
            'predictor': 'cpu_predictor',  # CPU optimized for low latency
        }
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train XGBoost model on lap time data.
        
        Args:
            X_train: Training features
            y_train: Training targets (lap times in seconds)
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training XGBoost model on {len(X_train)} samples")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Override hyperparameters from kwargs
        params = {**self.xgb_params, **kwargs}
        n_estimators = params.pop('n_estimators', 200)
        
        # Create DMatrix for efficient training
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        
        # Setup validation
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, 'val'))
        
        # Train model with early stopping
        evals_result = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # Store feature importance
        importance = self.model.get_score(importance_type='gain')
        total_gain = sum(importance.values())
        self.feature_importance_ = {
            k: v / total_gain for k, v in importance.items()
        }
        
        # Calculate training metrics
        train_pred = self.model.predict(dtrain)
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        train_mae = np.mean(np.abs(y_train - train_pred))
        
        metrics = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'best_iteration': self.model.best_iteration,
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(dval)
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
            val_mae = np.mean(np.abs(y_val - val_pred))
            metrics.update({
                'val_rmse': val_rmse,
                'val_mae': val_mae,
            })
        
        logger.info(f"Training complete. Train RMSE: {train_rmse:.3f}s, MAE: {train_mae:.3f}s")
        if 'val_rmse' in metrics:
            logger.info(f"Validation RMSE: {metrics['val_rmse']:.3f}s, MAE: {metrics['val_mae']:.3f}s")
        
        return metrics
    
    def predict(self, input_data: PredictionInput) -> PredictionOutput:
        """
        Predict lap time for single input.
        
        Args:
            input_data: Prediction input with race conditions
            
        Returns:
            Prediction output with lap time and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")
        
        # Validate input
        self._validate_input(input_data)
        
        # Extract features
        features = self._extract_features(input_data)
        
        # Create feature vector in correct order
        X = pd.DataFrame([features])[self.feature_names]
        
        # Predict using DMatrix for consistency
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        prediction = self.model.predict(dmatrix)[0]
        
        # Decompose prediction into pace components
        pace_components = self._decompose_prediction(input_data, features, prediction)
        
        # Calculate confidence based on feature values
        confidence = self._calculate_confidence(input_data, features)
        
        return PredictionOutput(
            predicted_lap_time=float(prediction),
            confidence=confidence,
            pace_components=pace_components,
            metadata={
                'model_type': 'xgboost',
                'version': self.config.version,
                'feature_count': len(self.feature_names),
            }
        )
    
    def predict_batch(self, inputs: List[PredictionInput]) -> List[PredictionOutput]:
        """
        Predict lap times for multiple inputs efficiently.
        
        Args:
            inputs: List of prediction inputs
            
        Returns:
            List of prediction outputs
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")
        
        # Extract features for all inputs
        features_list = [self._extract_features(inp) for inp in inputs]
        X = pd.DataFrame(features_list)[self.feature_names]
        
        # Batch prediction
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        predictions = self.model.predict(dmatrix)
        
        # Create outputs
        outputs = []
        for i, (input_data, pred) in enumerate(zip(inputs, predictions)):
            pace_components = self._decompose_prediction(input_data, features_list[i], pred)
            confidence = self._calculate_confidence(input_data, features_list[i])
            
            outputs.append(PredictionOutput(
                predicted_lap_time=float(pred),
                confidence=confidence,
                pace_components=pace_components,
                metadata={
                    'model_type': 'xgboost',
                    'version': self.config.version,
                }
            ))
        
        return outputs
    
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory to save model files
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_path = path / "xgboost_model.json"
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance_,
            'config': self.config.__dict__,
            'xgb_params': self.xgb_params,
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path) -> None:
        """
        Load model from disk.
        
        Args:
            path: Directory containing model files
        """
        model_path = path / "xgboost_model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.feature_importance_ = metadata['feature_importance']
        self.xgb_params = metadata['xgb_params']
        
        # Update config
        if 'config' in metadata:
            self.config = ModelConfig(**metadata['config'])
        
        logger.info(f"Model loaded from {path}")
    
    def get_metadata(self) -> Dict:
        """
        Get model metadata.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': 'xgboost',
            'version': self.config.version,
            'trained': self.model is not None,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance_,
            'hyperparameters': self.xgb_params,
            'best_iteration': self.model.best_iteration if self.model else None,
        }
    
    def _decompose_prediction(
        self,
        input_data: PredictionInput,
        features: Dict,
        total_prediction: float
    ) -> Dict[str, float]:
        """
        Decompose prediction into interpretable pace components.
        
        Args:
            input_data: Original input data
            features: Extracted features
            total_prediction: Total predicted lap time
            
        Returns:
            Dictionary of pace components
        """
        # Estimate base pace (clean air, optimal conditions)
        base_pace = total_prediction
        
        # Estimate tire effect (from tire age and degradation)
        tire_effect = features.get('tire_age', 0) * 0.05  # ~0.05s per lap of tire age
        if 'degradation_slope' in features:
            tire_effect = features['degradation_slope'] * features.get('tire_age', 0)
        
        # Estimate fuel effect (from fuel load)
        fuel_effect = features.get('fuel_load', 0) * 0.03  # ~0.03s per kg
        
        # Estimate traffic penalty
        traffic_penalty = 0.0
        if input_data.traffic_state == RaceCondition.DIRTY_AIR:
            traffic_penalty = features.get('traffic_penalty', 0.4)
        
        # Estimate safety car factor
        safety_car_factor = 1.0
        if input_data.safety_car_active:
            safety_car_factor = 1.3  # 30% slower under SC
        
        # Estimate weather adjustment
        weather_adjustment = 0.0
        if input_data.weather_temp:
            # Hotter temps generally slower
            weather_adjustment = (input_data.weather_temp - 25) * 0.02
        
        # Calculate base pace by removing effects
        base_pace = total_prediction - tire_effect - fuel_effect - traffic_penalty - weather_adjustment
        base_pace = base_pace / safety_car_factor
        
        return {
            'base_pace': float(base_pace),
            'tire_effect': float(tire_effect),
            'fuel_effect': float(fuel_effect),
            'traffic_penalty': float(traffic_penalty),
            'weather_adjustment': float(weather_adjustment),
            'safety_car_factor': float(safety_car_factor),
        }
    
    def _calculate_confidence(
        self,
        input_data: PredictionInput,
        features: Dict
    ) -> float:
        """
        Calculate prediction confidence based on input characteristics.
        
        Args:
            input_data: Original input data
            features: Extracted features
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 1.0
        
        # Reduce confidence for extreme tire ages
        if features.get('tire_age', 0) > 30:
            confidence *= 0.9
        elif features.get('tire_age', 0) > 40:
            confidence *= 0.8
        
        # Reduce confidence for rare conditions
        if input_data.safety_car_active:
            confidence *= 0.85
        
        # Reduce confidence for extreme fuel loads
        if features.get('fuel_load', 0) > 100:
            confidence *= 0.9
        
        # Reduce confidence if features are missing
        if input_data.track_temp is None:
            confidence *= 0.95
        if input_data.gap_to_ahead is None and input_data.traffic_state == RaceCondition.DIRTY_AIR:
            confidence *= 0.9
        
        return max(0.5, min(1.0, confidence))
